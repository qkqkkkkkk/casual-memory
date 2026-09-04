#!/usr/bin/env python3
"""Build disjoint FEVER experience and distractor pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def norm(value: str) -> str:
    return re.sub(r"\W+", " ", str(value).lower()).strip()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_pool(path: Path, rows: list[dict], pool: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            source_id = row.get("id", idx)
            handle.write(json.dumps({
                "memory_id": f"fever-train-{source_id}",
                "claim": row["claim"],
                "gold_label": row["label"],
                "evidence_bundle": row["evidence_bundle"],
                "rationale_digest": f"Historical FEVER precedent label: {row['label']}",
                "source_example_id": source_id,
                "source_pool": pool,
                "is_synthetic": False,
            }, ensure_ascii=False) + "\n")
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disjoint FEVER experience and distractor pools")
    parser.add_argument("--input", type=Path, required=True, help="Enriched FEVER train JSONL")
    parser.add_argument("--exclude-claims-from", type=Path, required=True, help="Enriched dev JSONL")
    parser.add_argument("--experience-output", type=Path, required=True)
    parser.add_argument("--distractor-output", type=Path, required=True)
    parser.add_argument("--max-experience-items", type=int, default=2000)
    parser.add_argument("--max-distractor-items", type=int, default=4000)
    parser.add_argument("--distractor-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.distractor_fraction < 1.0:
        raise SystemExit("--distractor-fraction must be between 0 and 1")

    excluded = {norm(row.get("claim", "")) for row in read_rows(args.exclude_claims_from)}
    rows = [row for row in read_rows(args.input)
            if row.get("label") in {"SUPPORTS", "REFUTES"}
            and row.get("evidence_bundle")
            and norm(row.get("claim", "")) not in excluded]
    rows.sort(key=lambda row: hashlib.sha256(
        f"{args.seed}|{row.get('id', '')}|{row.get('claim', '')}".encode("utf-8")
    ).hexdigest())
    distractors = [row for row in rows if int(hashlib.sha256(
        f"{args.seed}|d|{row.get('id', '')}".encode("utf-8")
    ).hexdigest()[:8], 16) / 0xFFFFFFFF < args.distractor_fraction]
    distractor_ids = {str(row.get("id")) for row in distractors}
    experience = [row for row in rows if str(row.get("id")) not in distractor_ids]

    rng = random.Random(args.seed)
    selected: dict[str, list[dict]] = {}
    for name, pool, limit in (
        ("experience", experience, args.max_experience_items),
        ("distractor", distractors, args.max_distractor_items),
    ):
        buckets = {label: [row for row in pool if row["label"] == label] for label in ("SUPPORTS", "REFUTES")}
        for bucket in buckets.values():
            rng.shuffle(bucket)
        half = limit // 2
        chosen = buckets["SUPPORTS"][:half] + buckets["REFUTES"][:half]
        rng.shuffle(chosen)
        selected[name] = chosen

    exp_md5 = write_pool(args.experience_output, selected["experience"], "experience")
    dis_md5 = write_pool(args.distractor_output, selected["distractor"], "distractor")
    exp_ids = {str(row.get("id")) for row in selected["experience"]}
    dis_ids = {str(row.get("id")) for row in selected["distractor"]}
    report = {
        "source_rows_after_dev_exclusion": len(rows),
        "experience_items": len(selected["experience"]),
        "distractor_items": len(selected["distractor"]),
        "experience_supports": sum(x["label"] == "SUPPORTS" for x in selected["experience"]),
        "experience_refutes": sum(x["label"] == "REFUTES" for x in selected["experience"]),
        "distractor_supports": sum(x["label"] == "SUPPORTS" for x in selected["distractor"]),
        "distractor_refutes": sum(x["label"] == "REFUTES" for x in selected["distractor"]),
        "pool_id_overlap": len(exp_ids & dis_ids),
        "experience_md5": exp_md5,
        "distractor_md5": dis_md5,
        "seed": args.seed,
        "outputs": [str(args.experience_output), str(args.distractor_output)],
    }
    print(json.dumps(report, indent=2))
    if report["pool_id_overlap"]:
        raise SystemExit("Experience and distractor pools overlap")


if __name__ == "__main__":
    main()
