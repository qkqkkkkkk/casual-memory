#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Build a frozen FEVER memory bank from enriched train JSONL")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-claims-from", type=Path, default=None, help="Optional enriched dev split used to prevent exact claim overlap")
    args = p.parse_args(); rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        raw = json.loads(line)
        if raw.get("label") not in {"SUPPORTS", "REFUTES"} or not raw.get("evidence_bundle"): continue
        rows.append(raw)
    normalize = lambda value: re.sub(r"\W+", " ", str(value).lower()).strip()
    excluded = set()
    removed_overlap = 0
    if args.exclude_claims_from:
        excluded = {normalize(json.loads(line).get("claim", "")) for line in args.exclude_claims_from.read_text(encoding="utf-8").splitlines() if line.strip()}
        before = len(rows)
        rows = [row for row in rows if normalize(row.get("claim", "")) not in excluded]
        removed_overlap = before - len(rows)
    rng = random.Random(args.seed)
    buckets = {label: [row for row in rows if row["label"] == label] for label in ("SUPPORTS", "REFUTES")}
    for bucket in buckets.values(): rng.shuffle(bucket)
    rows = buckets["SUPPORTS"][:args.max_items // 2] + buckets["REFUTES"][:args.max_items // 2]
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = args.output.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing memory bank: {args.output}") from exc
    with out:
        for idx, raw in enumerate(rows):
            item = {
                "memory_id": f"fever-train-{raw.get('id', idx)}",
                "claim": raw["claim"], "gold_label": raw["label"],
                "evidence_bundle": raw["evidence_bundle"],
                "rationale_digest": f"Historical FEVER precedent label: {raw['label']}",
                "source_example_id": raw.get("id", idx), "is_synthetic": False,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    digest = hashlib.md5(args.output.read_bytes()).hexdigest()
    print(json.dumps({"items": len(rows), "supports": sum(x["label"] == "SUPPORTS" for x in rows), "refutes": sum(x["label"] == "REFUTES" for x in rows), "excluded_exact_overlap": removed_overlap, "md5": digest, "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
