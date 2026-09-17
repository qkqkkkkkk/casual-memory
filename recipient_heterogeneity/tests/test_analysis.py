from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from recipient_heterogeneity.analysis import (
    AnalysisError,
    analyze_units,
    compare_independent_runs,
    write_analysis,
)


AGENTS = ("A1", "A2", "A3")


def unit(claim_id, receiver, use_samples, drop_samples, memory_id=None):
    memory_id = memory_id or f"memory-{claim_id}"
    utilities = [use - drop for use, drop in zip(use_samples, drop_samples)]
    return {
        "event_id": f"{claim_id}-{receiver}",
        "claim_id": claim_id,
        "receiver_agent_id": receiver,
        "memory_id": memory_id,
        "memory_observation_count": 3,
        "arms": {
            "drop": {
                "q_use": sum(use_samples) / len(use_samples),
                "q_control": sum(drop_samples) / len(drop_samples),
                "q_use_samples": use_samples,
                "q_control_samples": drop_samples,
                "team_utility": {"mean": sum(utilities) / len(utilities)},
                "team_utility_samples": utilities,
            }
        },
    }


def fixture_units():
    rows = []
    shared = [0.0, 1.0]
    drops = {
        "A1": [0.0, 0.0],
        "A2": [1.0, 1.0],
        "A3": [0.0, 1.0],
    }
    rows.extend(unit("c1", agent, shared, drops[agent]) for agent in AGENTS)
    rows.extend(
        unit("c2", agent, [1.0, 1.0], [0.0, 1.0]) for agent in AGENTS
    )
    return rows


class RecipientHeterogeneityTests(unittest.TestCase):
    def test_direct_flip_and_pairwise_metrics(self):
        result = analyze_units(fixture_units(), bootstrap_samples=200, seed=7)
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["recipient_event_count"], 6)
        self.assertEqual(
            result["direct_recipient_sign_flip_rate"]["estimate"], 0.5
        )
        self.assertEqual(
            result["any_recipient_sign_heterogeneity_rate"]["estimate"], 0.5
        )
        self.assertEqual(result["mean_within_event_utility_range"]["estimate"], 0.5)
        self.assertEqual(result["per_recipient"]["A1"]["positive_events"], 2)
        self.assertEqual(
            result["pairwise"]["A1_vs_A2"]["direct_opposite_sign_rate"][
                "estimate"
            ],
            0.5,
        )

    def test_independent_retest_reports_stable_direction(self):
        previous = analyze_units(fixture_units(), bootstrap_samples=200, seed=7)
        current = analyze_units(fixture_units(), bootstrap_samples=200, seed=8)
        retest = compare_independent_runs(
            current, previous, bootstrap_samples=200, seed=9
        )
        self.assertEqual(retest["shared_events"], 2)
        self.assertEqual(retest["stable_direct_flip_rate"]["estimate"], 0.5)
        self.assertEqual(
            retest["reproducible_directional_flip_rate"]["estimate"], 0.5
        )
        self.assertEqual(retest["recipient_sign_consistency"]["estimate"], 1.0)

    def test_incomplete_recipient_group_is_rejected(self):
        with self.assertRaisesRegex(AnalysisError, "incomplete recipient coverage"):
            analyze_units(fixture_units()[:-1], bootstrap_samples=200)

    def test_nonidentical_use_baseline_is_rejected(self):
        rows = fixture_units()
        rows[1]["arms"]["drop"]["q_use_samples"] = [1.0, 1.0]
        with self.assertRaisesRegex(AnalysisError, "USE baseline differs"):
            analyze_units(rows, bootstrap_samples=200)

    def test_write_analysis_creates_report_json_and_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"design_hash": "same-design"}
            units = {"design_hash": "same-design", "units": fixture_units()}
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "audit_units.json").write_text(
                json.dumps(units), encoding="utf-8"
            )
            output = write_analysis(root, bootstrap_samples=200)
            self.assertTrue(output.is_file())
            self.assertTrue((root / "recipient_matrix.csv").is_file())
            report = (root / "rq3_report.md").read_text(encoding="utf-8")
            self.assertIn("Direct +/− recipient sign-flip rate", report)
            self.assertIn("Not available", report)


if __name__ == "__main__":
    unittest.main()
