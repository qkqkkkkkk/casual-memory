from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from causal_memory_control import CounterfactualAudit, InterventionArm, InterventionBuilder
from causal_memory_control.fever_backend import (
    FeverBranchRunner,
    build_fever_checkpoint,
    build_fever_event,
    load_existing_rows,
    placebo_candidate,
)
from causal_memory_control.run_fever_audit import summarize_audit
from causal_memory_control import run_fever_audit


class FakeClient:
    def __init__(self, *_args, **_kwargs):
        self.calls = 0
        self.cache_hits = 0

    def ask(self, messages, repeat_idx):
        self.calls += 1
        prompt = "\n".join(message["content"] for message in messages)
        verdict = "REFUTES" if "(no memory)" in prompt else "SUPPORTS"
        return {
            "memory_relevance": "HIGH",
            "memory_influence": "ADOPTED",
            "memory_assessment": "fixture",
            "evidence_ids": ["E1"],
            "verdict": verdict,
            "confidence": 0.9,
            "rationale": "fixture",
        }, False


class FakeIndex:
    def __init__(self, bank, *_args, **_kwargs):
        self.bank = bank

    def search_with_scores(self, _query, k=1, label_bias=None):
        return [(item, 0.8 - index * 0.1) for index, item in enumerate(self.bank[:k])]


def fixture():
    claim = {
        "id": "c1",
        "claim": "The sky is blue.",
        "label": "SUPPORTS",
        "evidence_bundle": [
            {"title": "Sky", "text": "The sky appears blue.", "is_gold": True}
        ],
    }
    memory = {
        "memory_id": "m1",
        "claim": "Water is wet.",
        "gold_label": "SUPPORTS",
        "evidence_bundle": [{"title": "Water", "text": "Water is wet."}],
        "historical_success": True,
        "memory_schema_version": "gmemory-fever-v2",
        "rationale_digest": "Format-matched unrelated FEVER precedent.",
        "retrieval_score": 0.7,
        "retrieval_rank": 1,
    }
    placebo = {
        **memory,
        "memory_id": "placebo-m1",
        "claim": "Mars is distant.",
        "placebo_valid": True,
    }
    candidates = {agent: [memory] for agent in ("A1", "A2", "A3")}
    return claim, memory, placebo, candidates


class FeverRunnerTests(unittest.TestCase):
    def test_real_p2_backend_runs_all_receiver_interventions_and_resumes(self):
        claim, memory, placebo, candidates = fixture()
        event = build_fever_event(
            claim,
            candidates,
            receiver="A1",
            target_id="m1",
            event_id="event-1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit_runs.jsonl"
            path.touch()
            client = FakeClient()
            raw_runner = FeverBranchRunner(
                client=client,
                claim=claim,
                candidates=candidates,
                placebo_item=placebo,
                log_path=path,
                design_hash="design",
                run_hash="run",
            )
            audit = CounterfactualAudit(
                raw_runner,
                intervention_builder=InterventionBuilder(
                    [placebo_candidate(placebo)]
                ),
            ).run(
                build_fever_checkpoint(event, "c1"),
                event,
                arms=(
                    InterventionArm.DROP,
                    InterventionArm.PLACEBO,
                    InterventionArm.GLOBAL_DROP,
                ),
                seeds=(10, 11),
            )
            # Two repeats: one shared USE plus three controls each.
            self.assertEqual(len(path.read_text().splitlines()), 8)
            self.assertEqual(audit.primary.team_utility.mean, 0.0)
            self.assertEqual(audit.primary.local_utility.mean, 1.0)
            summary = summarize_audit(audit)
            self.assertEqual(summary["memory_observation_count"], 3)
            self.assertEqual(summary["arms"]["drop"]["q_use"], 1.0)
            self.assertEqual(summary["arms"]["drop"]["q_control"], 1.0)

            calls = client.calls
            resumed = FeverBranchRunner(
                client=client,
                claim=claim,
                candidates=candidates,
                placebo_item=placebo,
                log_path=path,
                design_hash="design",
                run_hash="run",
                existing_rows=load_existing_rows(path),
            )
            CounterfactualAudit(
                resumed,
                intervention_builder=InterventionBuilder(
                    [placebo_candidate(placebo)]
                ),
            ).run(
                build_fever_checkpoint(event, "c1"),
                event,
                arms=(InterventionArm.DROP,),
                seeds=(10, 11),
            )
            self.assertEqual(client.calls, calls)
            self.assertEqual(len(path.read_text().splitlines()), 8)

    def test_cli_writes_complete_first_run_and_blocks_training_without_retest(self):
        claim, memory, _, _ = fixture()
        second_memory = {
            **memory,
            "memory_id": "m2",
            "source_example_id": "m2",
            "claim": "Mars looks reddish.",
            "evidence_bundle": [{"title": "Mars", "text": "Mars looks reddish."}],
        }
        memory["source_example_id"] = "m1"
        distractor = {
            **second_memory,
            "memory_id": "d1",
            "source_example_id": "d1",
            "claim": "Grass is green.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.jsonl"
            experience_path = root / "experience.jsonl"
            distractor_path = root / "distractor.jsonl"
            output = root / "output"
            test_path.write_text(json.dumps(claim) + "\n", encoding="utf-8")
            experience_path.write_text(
                "\n".join(json.dumps(row) for row in (memory, second_memory)) + "\n",
                encoding="utf-8",
            )
            distractor_path.write_text(json.dumps(distractor) + "\n", encoding="utf-8")
            with patch.object(run_fever_audit, "CachedChat", FakeClient), patch.object(
                run_fever_audit, "GMemorySemanticIndex", FakeIndex
            ):
                run_fever_audit.main(
                    (
                        "--test", str(test_path),
                        "--experience-bank", str(experience_path),
                        "--distractor-bank", str(distractor_path),
                        "--claims", "1",
                        "--repeats", "2",
                        "--arms", "drop,placebo",
                        "--output-dir", str(output),
                    )
                )
            self.assertEqual(len((output / "audit_runs.jsonl").read_text().splitlines()), 6)
            gate = json.loads((output / "pretraining_gate.json").read_text())
            estimator = json.loads((output / "estimator_evaluation.json").read_text())
            self.assertFalse(gate["can_train"])
            self.assertIn("independent_retest_required", gate["reasons"])
            self.assertEqual(estimator["status"], "blocked_by_pretraining_gate")

            retest_output = root / "retest"
            with patch.object(run_fever_audit, "CachedChat", FakeClient), patch.object(
                run_fever_audit, "GMemorySemanticIndex", FakeIndex
            ):
                run_fever_audit.main(
                    (
                        "--test", str(test_path),
                        "--experience-bank", str(experience_path),
                        "--distractor-bank", str(distractor_path),
                        "--claims", "1",
                        "--repeats", "2",
                        "--arms", "drop,placebo",
                        "--sample-seed-base", "2000",
                        "--retest-results", str(output),
                        "--output-dir", str(retest_output),
                    )
                )
            retest_gate = json.loads(
                (retest_output / "pretraining_gate.json").read_text()
            )
            self.assertEqual(
                retest_gate["sign_consistency_source"],
                "independent_run_level_retest",
            )
            self.assertEqual(retest_gate["independent_retest_coverage"], 1.0)
            self.assertNotIn("independent_retest_required", retest_gate["reasons"])


if __name__ == "__main__":
    unittest.main()
