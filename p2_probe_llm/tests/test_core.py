from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from p2_probe_llm.evidence import enrich_split
from p2_probe_llm.build_pools import write_pool
from p2_probe_llm.mas import _evidence_metrics, _prompt, run_episode
from p2_probe_llm.stats import bh_reject, effect
from p2_probe_llm.retrieval import BM25Index, GMemorySemanticIndex, role_query
from p2_probe_llm.run_experiment_e1 import memory_is_eligible, retrieve
from p2_probe_llm.diagnose_e1 import read_jsonl


class FakeClient:
    def __init__(self):
        self.cache_hits = 0
        self.calls = 0

    def ask(self, messages, repeat_idx):
        self.calls += 1
        return {"memory_relevance": "HIGH", "memory_influence": "ADOPTED", "memory_assessment": "The precedent is relevant.", "verdict": "SUPPORTS", "confidence": .8, "rationale": "fixture"}, False


class FakeEncoder:
    vectors = {
        "cat precedent": [1.0, 0.0],
        "space precedent": [0.0, 1.0],
        "feline claim": [0.9, 0.1],
    }

    def encode(self, texts, **_kwargs):
        return [self.vectors[text] for text in texts]


class RealProbeTests(unittest.TestCase):
    def test_resolves_official_fever_line_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); wiki = root / "wiki"; wiki.mkdir()
            (wiki / "wiki-001.jsonl").write_text(json.dumps({"id": "Test_Page", "lines": "0\tFirst sentence.\tlink\n1\tTarget sentence."}) + "\n", encoding="utf-8")
            source = root / "dev.jsonl"
            source.write_text(json.dumps({"id": 1, "claim": "A claim", "label": "SUPPORTS", "evidence": [[[1, 2, "Test_Page", 1]]]}) + "\n", encoding="utf-8")
            output = root / "enriched.jsonl"
            report = enrich_split(source, wiki, output)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["evidence_bundle"][0]["text"], "Target sentence.")
            self.assertEqual(report["binary_with_evidence"], 1)

    def test_two_round_team_and_solo(self):
        client = FakeClient()
        claim = {"id": "c1", "claim": "Claim", "label": "SUPPORTS", "evidence_bundle": [{"title": "T", "text": "Evidence"}]}
        memory = {"memory_id": "m1", "claim": "Prior", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Prior evidence"}], "rationale_digest": "prior"}
        candidates = {agent: [memory] for agent in ("A1", "A2", "A3")}
        run = run_episode(client, claim, candidates, 0, "treated", ("c1", "A1", "m1"), {"m1": memory})
        self.assertTrue(run.team_correct)
        self.assertTrue(run.per_agent_correct_solo["A1"])
        self.assertTrue(run.round1["A1"]["memory_considered"])
        self.assertEqual(client.calls, 7)

    def test_control_replaces_only_a1_memory(self):
        client = FakeClient()
        claim = {"id": "c1", "claim": "Claim", "label": "SUPPORTS", "evidence_bundle": [{"title": "T", "text": "Evidence"}]}
        memory = {"memory_id": "m1", "claim": "Prior", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Prior evidence"}], "rationale_digest": "prior"}
        placebo = {"memory_id": "placeholder", "claim": "Other", "gold_label": "REFUTES", "evidence_bundle": [{"text": "Other evidence"}], "rationale_digest": "unrelated"}
        candidates = {agent: [memory] for agent in ("A1", "A2", "A3")}
        run = run_episode(client, claim, candidates, 0, "control", ("c1", "A1", "m1"), {"m1": placebo})
        self.assertEqual(run.candidates["A1"], ["placebo-m1"])
        self.assertEqual(run.candidates["A2"], ["m1"])
        self.assertEqual(run.candidates["A3"], ["m1"])

    def test_build_pool_uses_gmemory_v2_task_main_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            rows = [{
                "id": 7,
                "claim": "Marie Curie won a Nobel Prize.",
                "label": "SUPPORTS",
                "evidence_bundle": [{"title": "Marie Curie", "text": "She won Nobel Prizes."}],
            }]
            write_pool(path, rows, "experience")
            memory = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(memory["memory_schema_version"], "gmemory-fever-v2")
            self.assertEqual(memory["task_main"], rows[0]["claim"])
            self.assertIn("Historical final verdict: SUPPORTS", memory["task_trajectory"])
            self.assertIn("key_steps", memory)

    def test_gmemory_prompt_requires_explicit_memory_assessment(self):
        claim = {"claim": "A claim", "evidence_bundle": [{"title": "T", "text": "Evidence"}]}
        memory = {"memory_id": "m1", "claim": "Prior", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Past evidence"}], "rationale_digest": "Compare carefully."}
        messages = _prompt("Evidence Analyst", claim, [memory])
        combined = "\n".join(message["content"] for message in messages)
        self.assertIn("MANDATORY TO CONSIDER", combined)
        self.assertIn("memory_assessment", combined)
        self.assertIn("PRIMARY REFERENCE", combined)

    def test_evidence_metrics_uses_explicit_gold_flags(self):
        bundle = [
            {"title": "Gold", "text": "supports", "is_gold": True},
            {"title": "Distractor from m2", "text": "unrelated", "is_gold": False},
        ]
        metrics = _evidence_metrics(["E1", "E2"], bundle)
        self.assertEqual(metrics["evidence_precision"], .5)
        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertEqual(metrics["evidence_f1"], 2 / 3)

    def test_evidence_metrics_marks_hidden_gold_as_not_estimable(self):
        bundle = [{"title": "Distractor from m2", "text": "unrelated", "is_gold": False}]
        metrics = _evidence_metrics(["E1"], bundle)
        self.assertIsNone(metrics["evidence_f1"])
        self.assertFalse(metrics["evidence_gold_available"])

    def test_e1_roles_share_claim_level_retrieval_query(self):
        claim = {"claim": "Shared query", "evidence_bundle": [{"text": "Ignored for retrieval"}]}
        queries = [role_query(claim, aid) for aid in ("A1", "A2", "A3")]
        self.assertEqual(queries, [("Shared query", None)] * 3)

    def test_gmemory_semantic_index_ranks_task_main_by_cosine(self):
        bank = [
            {"memory_id": "cat", "task_main": "cat precedent"},
            {"memory_id": "space", "task_main": "space precedent"},
        ]
        index = GMemorySemanticIndex(bank, threshold=.3, encoder=FakeEncoder())
        matches = index.search_with_scores("feline claim", k=2)
        self.assertEqual([item["memory_id"] for item, _ in matches], ["cat"])
        self.assertGreater(matches[0][1], .9)

    def test_stats(self):
        point, interval = effect([1, 1, 0, 1], 100, 42)
        self.assertEqual(point, .75)
        self.assertEqual(len(interval), 2)
        self.assertEqual(bh_reject([.001, .01, .9], .1), [True, True, False])

    def test_bm25_is_deterministic(self):
        bank = [
            {"memory_id": "a", "claim": "Marie Curie won a Nobel Prize", "evidence_bundle": []},
            {"memory_id": "b", "claim": "Alan Turing worked on computing", "evidence_bundle": []},
        ]
        index = BM25Index(bank)
        first = [x["memory_id"] for x in index.search("Marie Curie Nobel", k=2)]
        second = [x["memory_id"] for x in index.search("Marie Curie Nobel", k=2)]
        self.assertEqual(first, second)
        self.assertEqual(first[0], "a")

    def test_e1_retrieval_excludes_distractor_provenance_and_evidence(self):
        claim = {
            "id": "c1", "claim": "Marie Curie won a Nobel Prize",
            "label": "SUPPORTS",
            "evidence_bundle": [{"text": "Marie Curie won the Nobel Prize."}],
            "evidence_policy": {"distractor_source_ids": ["fever-train-bad"]},
        }
        bank = [
            {"memory_id": "fever-train-bad", "source_example_id": "bad", "claim": "Marie Curie Nobel", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Unrelated."}]},
            {"memory_id": "fever-train-overlap", "claim": "Marie Curie Nobel Prize", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Marie Curie won the Nobel Prize."}]},
            {"memory_id": "fever-train-good", "claim": "Alan Turing worked on computing", "gold_label": "SUPPORTS", "evidence_bundle": [{"text": "Turing worked on computing."}]},
        ]
        found = retrieve(claim, bank, "A1", 1)
        self.assertEqual([item["memory_id"] for item in found], ["fever-train-good"])
        self.assertFalse(memory_is_eligible(claim, bank[0]))
        self.assertFalse(memory_is_eligible(claim, bank[1]))

    def test_diagnostic_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"x": 1}\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"x": 1}])


if __name__ == "__main__":
    unittest.main()
