#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import CachedChat
from .difficulty import make_variant
from .mas import run_episode
from .run_experiment import load, placebo, retrieve, select_claims
from .retrieval import BM25Index


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate real-LLM FEVER evidence difficulty")
    p.add_argument("--test", type=Path, required=True); p.add_argument("--memory-bank", type=Path, required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1"); p.add_argument("--api-key", default=None); p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--claims", type=int, default=100); p.add_argument("--repeats", type=int, default=1); p.add_argument("--top-k", type=int, default=6); p.add_argument("--gold-recalls", default="0,0.3,0.5,0.7,1.0"); p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "difficulty_calibration.json"
    if report_path.exists(): raise SystemExit(f"Refusing to overwrite existing calibration: {report_path}")
    test = load(args.test); sampled = select_claims(test, args.claims, args.seed); bank = load(args.memory_bank, binary=False)
    recalls = [float(x) for x in args.gold_recalls.split(",")]
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    index = BM25Index(bank)
    placebo_cache = {}; rows = []; eligible_variants = {}; excluded_by_recall = {}
    for recall in recalls:
        correct = total = individual_correct = 0
        eligible_variants[recall] = []; excluded_by_recall[recall] = 0
        for claim in sampled:
            variant = make_variant(claim, bank, recall, args.seed)
            candidates = {aid: retrieve(variant, bank, aid, args.top_k, index) for aid in ("A1", "A2", "A3")}
            for item in {x["memory_id"]: x for values in candidates.values() for x in values}.values():
                placebo_cache.setdefault(item["memory_id"], placebo(item, bank))
            if any(not placebo_cache[item["memory_id"]].get("placebo_valid", False)
                   for values in candidates.values() for item in values):
                excluded_by_recall[recall] += 1
                continue
            candidates = {aid: [placebo_cache[item["memory_id"]] for item in values] for aid, values in candidates.items()}
            eligible_variants[recall].append(variant)
            for repeat in range(args.repeats):
                run = run_episode(client, variant, candidates, repeat, "placebo_all")
                correct += int(run.team_correct); individual_correct += sum(run.per_agent_correct_r1.values()); total += 1
        rows.append({"gold_recall": recall, "team_accuracy": correct / max(1, total), "individual_accuracy": individual_correct / max(1, 3 * total), "runs": total, "eligible_claims": len(eligible_variants[recall]), "excluded_invalid_placebo_claims": excluded_by_recall[recall]})
    eligible = [row for row in rows if .62 <= row["team_accuracy"] <= .80]
    selected = min(eligible, key=lambda row: abs(row["team_accuracy"] - .70)) if eligible else None
    report = {"experiment": "fever_difficulty_calibration", "model": args.model, "target_band": [.62, .80], "target_center": .70, "selected": selected, "conditions": rows, "llm_calls": client.calls, "cache_hits": client.cache_hits, "pass": selected is not None}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if selected:
        selected_path = args.output_dir / "fever_dev_selected_difficulty.jsonl"
        with selected_path.open("x", encoding="utf-8") as handle:
            for claim in eligible_variants[selected["gold_recall"]]:
                handle.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    if not selected: raise SystemExit("No gold-recall condition reached the target band; inspect the report before continuing.")


if __name__ == "__main__": main()
