#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .client import CachedChat
from .mas import run_episode
from .run_experiment import load, placebo, retrieve, select_claims
from .stats import phi
from .retrieval import BM25Index


def metrics(runs: list[dict], condition: str) -> dict:
    rows = [row for row in runs if row["arm"] == condition]
    agents = ("A1", "A2", "A3")
    individual = sum(sum(row["per_agent_correct_r1"].values()) for row in rows) / max(1, 3 * len(rows))
    errors = {a: [not row["per_agent_correct_r1"][a] for row in rows] for a in agents}
    correlations = [phi(errors[a], errors[b]) for idx, a in enumerate(agents) for b in agents[idx + 1:]]
    majority_r1 = [sum(row["per_agent_correct_r1"].values()) >= 2 for row in rows]
    return {
        "condition": condition,
        "n_runs": len(rows),
        "individual_accuracy": individual,
        "error_correlation": sum(correlations) / len(correlations),
        "disagreement_rate": sum(len({row["round1"][a]["verdict"] for a in agents}) > 1 for row in rows) / max(1, len(rows)),
        "oracle_accuracy": sum(any(row["per_agent_correct_r1"].values()) for row in rows) / max(1, len(rows)),
        "team_accuracy_round1": sum(majority_r1) / max(1, len(rows)),
        "team_accuracy_round2": sum(row["team_correct"] for row in rows) / max(1, len(rows)),
    }


def with_cluster_ci(runs: list[dict], condition: str, n_bootstrap: int, seed: int) -> dict:
    point = metrics(runs, condition)
    grouped = defaultdict(list)
    for row in runs:
        if row["arm"] == condition: grouped[row["claim_id"]].append(row)
    keys = list(grouped); rng = random.Random(seed)
    sampled_metrics = []
    for _ in range(n_bootstrap):
        sampled = [row for _ in keys for row in grouped[rng.choice(keys)]]
        sampled_metrics.append(metrics(sampled, condition))
    for key in ("individual_accuracy", "error_correlation", "disagreement_rate", "oracle_accuracy", "team_accuracy_round1", "team_accuracy_round2"):
        values = sorted(row[key] for row in sampled_metrics)
        point[key + "_ci"] = [values[int(.025 * len(values))], values[int(.975 * len(values))]]
    return point


def paired_delta_ci(runs: list[dict], n_bootstrap: int, seed: int) -> dict:
    """Cluster-bootstrap memory minus placebo using paired claim clusters."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in runs:
        grouped[row["claim_id"]][row["arm"]].append(row)
    keys = [key for key, arms in grouped.items()
            if arms.get("memory_all") and arms.get("placebo_all")]
    if not keys:
        return {}
    point_memory = metrics(runs, "memory_all")
    point_placebo = metrics(runs, "placebo_all")
    metric_names = ("individual_accuracy", "error_correlation", "disagreement_rate",
                    "oracle_accuracy", "team_accuracy_round1", "team_accuracy_round2")
    point = {name: point_memory[name] - point_placebo[name] for name in metric_names}
    rng = random.Random(seed)
    samples = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        sampled = []
        for _ in keys:
            key = rng.choice(keys)
            sampled.extend(grouped[key]["memory_all"])
            sampled.extend(grouped[key]["placebo_all"])
        mem = metrics(sampled, "memory_all")
        pla = metrics(sampled, "placebo_all")
        for name in metric_names:
            samples[name].append(mem[name] - pla[name])
    return {
        name: {"estimate": point[name], "ci": [sorted(samples[name])[int(.025 * n_bootstrap)],
                                                   sorted(samples[name])[int(.975 * n_bootstrap)]]}
        for name in metric_names
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Real-LLM FEVER memory-all/placebo-all diversity experiment")
    p.add_argument("--test", type=Path, required=True); p.add_argument("--memory-bank", type=Path, required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1"); p.add_argument("--api-key", default=None); p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--claims", type=int, default=20); p.add_argument("--repeats", type=int, default=5); p.add_argument("--top-k", type=int, default=6); p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    claims = select_claims(load(args.test), args.claims, args.seed); bank = load(args.memory_bank, binary=False)
    bank_md5 = hashlib.md5(args.memory_bank.read_bytes()).hexdigest()
    config_md5 = hashlib.md5(json.dumps({"model": args.model, "claims": args.claims, "repeats": args.repeats, "top_k": args.top_k, "seed": args.seed, "endpoint": args.endpoint}, sort_keys=True).encode()).hexdigest()
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    index = BM25Index(bank)
    log_path = args.output_dir / "diversity_runs.jsonl"
    try:
        log_path.open("x", encoding="utf-8").close()
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing log: {log_path}. Use a new --output-dir.") from exc
    placebo_cache = {}; runs = []; excluded_claims = 0
    for claim in claims:
        candidates = {a: retrieve(claim, bank, a, args.top_k, index) for a in ("A1", "A2", "A3")}
        for item in {x["memory_id"]: x for values in candidates.values() for x in values}.values():
            placebo_cache.setdefault(item["memory_id"], placebo(item, bank))
        if any(not placebo_cache[item["memory_id"]].get("placebo_valid", False)
               for values in candidates.values() for item in values):
            excluded_claims += 1
            continue
        placebo_candidates = {a: [placebo_cache[item["memory_id"]] for item in values] for a, values in candidates.items()}
        for repeat in range(args.repeats):
            pair = [{"config_md5": config_md5, "bank_md5": bank_md5, **run_episode(client, claim, candidates, repeat, "memory_all").__dict__}, {"config_md5": config_md5, "bank_md5": bank_md5, **run_episode(client, claim, placebo_candidates, repeat, "placebo_all").__dict__}]
            runs.extend(pair)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in pair) + "\n")
    result = {
        "experiment": "p2_probe_real_llm_diversity", "model": args.model,
        "retrieval_method": "bm25",
        "n_claims": len(claims) - excluded_claims, "n_claims_input": len(claims),
        "excluded_invalid_placebo_claims": excluded_claims, "repeats": args.repeats,
        "memory_bank_md5": bank_md5, "config_md5": config_md5,
        "conditions": [with_cluster_ci(runs, name, args.bootstrap, args.seed) for name in ("memory_all", "placebo_all")],
        "cache_hits": client.cache_hits, "llm_calls": client.calls,
    }
    result["paired_deltas"] = paired_delta_ci(runs, args.bootstrap, args.seed + 1)
    (args.output_dir / "diversity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs = [output for run in runs for section in ("round1", "round2") for output in run[section].values()]
    parse_rate = sum(bool(x.get("parse_fail")) for x in outputs) / max(1, len(outputs))
    memory, placebo_all = result["conditions"]
    used_memory_ids = {item_id for run in runs if run["arm"] == "memory_all" for values in run["candidates"].values() for item_id in values}
    placebo_quality = bool(used_memory_ids) and all(placebo_cache[item_id].get("placebo_valid", False) for item_id in used_memory_ids)
    memory_runs = [run for run in runs if run["arm"] == "memory_all"]
    role_diversity = sum(len({tuple(run["candidates"][aid]) for aid in ("A1", "A2", "A3")}) > 1 for run in memory_runs) / max(1, len(memory_runs))
    delta = result["paired_deltas"]
    gate = (
        "# Real-LLM FEVER Diversity Gate Report\n\n"
        f"- Structured-output parse failure rate <= 0.5%: {'PASS' if parse_rate <= .005 else 'FAIL'} ({parse_rate:.4%})\n"
        f"- Placebo similarity < 0.15 and token difference <= 10%: {'PASS' if placebo_quality else 'FAIL'}\n"
        f"- Role-specific candidate sets differ: {'PASS' if role_diversity >= .8 else 'FAIL'} ({role_diversity:.2%})\n"
        f"- Memory-all individual accuracy: {memory['individual_accuracy']:.4f}\n"
        f"- Placebo-all individual accuracy: {placebo_all['individual_accuracy']:.4f}\n"
        f"- Error-correlation delta (memory - placebo): {memory['error_correlation'] - placebo_all['error_correlation']:+.4f}\n"
        f"- Round-1 majority-accuracy delta: {memory['team_accuracy_round1'] - placebo_all['team_accuracy_round1']:+.4f}\n"
        f"- Paired error-correlation delta: {delta.get('error_correlation', {}).get('estimate', float('nan')):+.4f} (CI {delta.get('error_correlation', {}).get('ci', [])})\n"
        f"- Paired round-1 team-accuracy delta: {delta.get('team_accuracy_round1', {}).get('estimate', float('nan')):+.4f} (CI {delta.get('team_accuracy_round1', {}).get('ci', [])})\n"
        f"- Excluded claims with invalid placebo: {excluded_claims}\n"
    )
    (args.output_dir / "gate_report.md").write_text(gate, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
