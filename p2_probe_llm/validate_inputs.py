#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .run_experiment_e1 import _evidence_texts, _memory_source_keys, load, placebo, retrieve, select_claims


def normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate enriched FEVER E1 inputs before spending LLM calls")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--experience-bank", type=Path, required=True)
    p.add_argument("--distractor-bank", type=Path, required=True)
    p.add_argument("--sample-claims", type=int, default=100)
    p.add_argument("--top-k", type=int, choices=(1,), default=1)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    raw_test = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    binary = [row for row in raw_test if row.get("label") in {"SUPPORTS", "REFUTES"}]
    test = load(args.test); experience = load(args.experience_bank, binary=False); distractors = load(args.distractor_bank, binary=False)
    test_keys = {normalize(row["claim"]) for row in test}
    experience_keys = {normalize(row["claim"]) for row in experience}
    distractor_keys = {normalize(row["claim"]) for row in distractors}
    experience_ids = {_memory_source_keys(row).intersection({str(row.get("memory_id")), str(row.get("source_example_id"))}) for row in experience}
    experience_id_values = {value for values in experience_ids for value in values}
    distractor_id_values = {value for row in distractors for value in _memory_source_keys(row)}
    sampled = select_claims(test, min(args.sample_claims, len(test)), 42)
    role_different = 0; quality = []; valid_candidates = 0; provenance_overlap = 0; exact_evidence_overlap = 0
    for claim in sampled:
        candidates = {aid: retrieve(claim, experience, aid, args.top_k) for aid in ("A1", "A2", "A3")}
        if any(not values for values in candidates.values()):
            continue
        valid_candidates += 1
        role_different += int(len({tuple(x["memory_id"] for x in values) for values in candidates.values()}) > 1)
        item = candidates["A1"][0]
        if claim.get("evidence_policy", {}).get("distractor_source_ids", []) and {str(x) for x in claim["evidence_policy"]["distractor_source_ids"]} & _memory_source_keys(item):
            provenance_overlap += 1
        if _evidence_texts(claim) & _evidence_texts(item):
            exact_evidence_overlap += 1
        replacement = placebo(item, experience, forbidden_evidence=_evidence_texts(claim))
        quality.append((replacement["placebo_similarity"], replacement["placebo_token_ratio"]))
    report = {
        "binary_examples": len(binary), "binary_with_evidence": len(test),
        "evidence_coverage": len(test) / max(1, len(binary)),
        "experience_items": len(experience), "distractor_items": len(distractors),
        "experience_supports": sum(x.get("gold_label") == "SUPPORTS" for x in experience),
        "experience_refutes": sum(x.get("gold_label") == "REFUTES" for x in experience),
        "exact_train_dev_overlap": len(test_keys & (experience_keys | distractor_keys)),
        "experience_distractor_id_overlap": len(experience_id_values & distractor_id_values),
        "valid_candidate_claims": valid_candidates,
        "retrieved_provenance_overlap": provenance_overlap,
        "retrieved_exact_evidence_overlap": exact_evidence_overlap,
        "role_candidate_difference_rate": role_different / max(1, valid_candidates),
        "placebo_similarity_max": max((x[0] for x in quality), default=1.0),
        "placebo_token_ratio_max": max((x[1] for x in quality), default=1.0),
    }
    report["pass"] = bool(
        report["evidence_coverage"] >= .95
        and report["exact_train_dev_overlap"] == 0
        and report["experience_distractor_id_overlap"] == 0
        and report["valid_candidate_claims"] == len(sampled)
        and report["retrieved_provenance_overlap"] == 0
        and report["retrieved_exact_evidence_overlap"] == 0
        and report["placebo_similarity_max"] < .15
        and report["placebo_token_ratio_max"] <= .10
    )
    text = json.dumps(report, indent=2)
    if args.output: args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["pass"]: raise SystemExit(2)


if __name__ == "__main__": main()
