from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from p2_probe_llm.evidence import enrich_split
from p2_probe_llm.mas import run_episode
from p2_probe_llm.stats import bh_reject, effect
from p2_probe_llm.retrieval import BM25Index
from p2_probe_llm.run_experiment_e1 import memory_is_eligible, retrieve


class FakeClient:
    def __init__(self):
        self.cache_hits = 0
        self.calls = 0

    def ask(self, messages, repeat_idx):
        self.calls += 1
        return {"verdict": "SUPPORTS", "confidence": .8, "rationale": "fixture"}, False


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
        self.assertEqual(client.calls, 7)

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


if __name__ == "__main__":
    unittest.main()
