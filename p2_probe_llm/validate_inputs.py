#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .run_experiment import load, placebo, retrieve, select_claims


def normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate enriched FEVER inputs before spending LLM calls")
    p.add_argument("--test", type=Path, required=True); p.add_argument("--memory-bank", type=Path, required=True); p.add_argument("--sample-claims", type=int, default=100); p.add_argument("--top-k", type=int, choices=(1,), default=1, help="Number of memories injected per agent; fixed to top-1 for this experiment"); p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    raw_test = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    binary = [row for row in raw_test if row.get("label") in {"SUPPORTS", "REFUTES"}]
    test = load(args.test); bank = load(args.memory_bank, binary=False)
    test_keys = {normalize(row["claim"]) for row in test}; bank_keys = {normalize(row["claim"]) for row in bank}
    sampled = select_claims(test, min(args.sample_claims, len(test)), 42)
    role_different = 0; quality = []
    for claim in sampled:
        candidates = {aid: retrieve(claim, bank, aid, args.top_k) for aid in ("A1", "A2", "A3")}
        role_different += int(len({tuple(x["memory_id"] for x in values) for values in candidates.values()}) > 1)
        item = candidates["A1"][0]; replacement = placebo(item, bank)
        quality.append((replacement["placebo_similarity"], replacement["placebo_token_ratio"]))
    report = {
        "binary_examples": len(binary), "binary_with_evidence": len(test),
        "evidence_coverage": len(test) / max(1, len(binary)),
        "memory_items": len(bank),
        "memory_supports": sum(x.get("gold_label") == "SUPPORTS" for x in bank),
        "memory_refutes": sum(x.get("gold_label") == "REFUTES" for x in bank),
        "exact_train_dev_overlap": len(test_keys & bank_keys),
        "role_candidate_difference_rate": role_different / max(1, len(sampled)),
        "placebo_similarity_max": max((x[0] for x in quality), default=1.0),
        "placebo_token_ratio_max": max((x[1] for x in quality), default=1.0),
    }
    report["pass"] = bool(report["evidence_coverage"] >= .95 and report["exact_train_dev_overlap"] == 0 and report["role_candidate_difference_rate"] >= .8 and report["placebo_similarity_max"] < .15 and report["placebo_token_ratio_max"] <= .10)
    text = json.dumps(report, indent=2)
    if args.output: args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["pass"]: raise SystemExit(2)


if __name__ == "__main__": main()
