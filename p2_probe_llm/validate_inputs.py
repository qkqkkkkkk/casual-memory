#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .retrieval import GMemorySemanticIndex
from .run_experiment_e1 import _evidence_texts, _memory_source_keys, load, placebo, retrieve, schema_coverage, select_claims


def normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate enriched FEVER E1 inputs before spending LLM calls")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--experience-bank", type=Path, required=True)
    p.add_argument("--distractor-bank", type=Path, required=True)
    p.add_argument("--sample-claims", type=int, default=100)
    p.add_argument("--top-k", type=int, choices=(1,), default=1)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--retrieval-threshold", type=float, default=0.3)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    raw_test = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    binary = [row for row in raw_test if row.get("label") in {"SUPPORTS", "REFUTES"}]
    test = load(args.test); experience = load(args.experience_bank, binary=False); distractors = load(args.distractor_bank, binary=False)
    test_keys = {normalize(row["claim"]) for row in test}
    experience_keys = {normalize(row["claim"]) for row in experience}
    distractor_keys = {normalize(row["claim"]) for row in distractors}
    # Flatten provenance IDs into one hashable set.  The previous version
    # accidentally created a set of sets, which raises ``TypeError``.
    experience_id_values = {value for row in experience for value in _memory_source_keys(row)}
    distractor_id_values = {value for row in distractors for value in _memory_source_keys(row)}
    index = GMemorySemanticIndex(experience, args.embedding_model, args.retrieval_threshold)
    sampled = select_claims(test, min(args.sample_claims, len(test)), 42)
    shared_count = 0; quality = []; retrieval_scores = []; valid_candidates = 0; provenance_overlap = 0; exact_evidence_overlap = 0
    for claim in sampled:
        shared = retrieve(claim, experience, "A1", args.top_k, index)
        candidates = {aid: list(shared) for aid in ("A1", "A2", "A3")}
        if any(not values for values in candidates.values()):
            continue
        valid_candidates += 1
        shared_count += int(len({tuple(x["memory_id"] for x in values) for values in candidates.values()}) == 1)
        item = candidates["A1"][0]
        retrieval_scores.append(float(item["retrieval_score"]))
        if claim.get("evidence_policy", {}).get("distractor_source_ids", []) and {str(x) for x in claim["evidence_policy"]["distractor_source_ids"]} & _memory_source_keys(item):
            provenance_overlap += 1
        if _evidence_texts(claim) & _evidence_texts(item):
            exact_evidence_overlap += 1
        replacement = placebo(item, experience, forbidden_evidence=_evidence_texts(claim))
        quality.append((replacement["placebo_similarity"], replacement["placebo_token_ratio"]))
    report = {
        "binary_examples": len(binary), "binary_with_evidence": len(test),
        "evidence_coverage": len(test) / max(1, len(binary)),
        "experience_items": len(experience), "distractor_items": len(distractors), "retrieval_method": "gmemory_semantic_claim", "embedding_model": args.embedding_model, "retrieval_threshold": args.retrieval_threshold,
        "experience_schema_coverage": schema_coverage(experience), "distractor_schema_coverage": schema_coverage(distractors),
        "experience_supports": sum(x.get("gold_label") == "SUPPORTS" for x in experience),
        "experience_refutes": sum(x.get("gold_label") == "REFUTES" for x in experience),
        "exact_train_dev_overlap": len(test_keys & (experience_keys | distractor_keys)),
        "experience_distractor_id_overlap": len(experience_id_values & distractor_id_values),
        "valid_candidate_claims": valid_candidates,
        "retrieved_provenance_overlap": provenance_overlap,
        "retrieved_exact_evidence_overlap": exact_evidence_overlap,
        "shared_candidate_rate": shared_count / max(1, valid_candidates),
        "retrieval_score_min": min(retrieval_scores, default=None),
        "retrieval_score_mean": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        "retrieval_score_max": max(retrieval_scores, default=None),
        "placebo_similarity_max": max((x[0] for x in quality), default=1.0),
        "placebo_token_ratio_max": max((x[1] for x in quality), default=1.0),
    }
    report["pass"] = bool(
        report["evidence_coverage"] >= .95
        and report["exact_train_dev_overlap"] == 0
        and report["experience_distractor_id_overlap"] == 0
        and report["experience_schema_coverage"] == 1.0
        and report["distractor_schema_coverage"] == 1.0
        and report["valid_candidate_claims"] == len(sampled)
        and report["shared_candidate_rate"] == 1.0
        and report["retrieved_provenance_overlap"] == 0
        and report["retrieved_exact_evidence_overlap"] == 0
        and report["placebo_similarity_max"] < .15
        and report["placebo_token_ratio_max"] <= .10
    )
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["pass"]: raise SystemExit(2)


if __name__ == "__main__": main()
