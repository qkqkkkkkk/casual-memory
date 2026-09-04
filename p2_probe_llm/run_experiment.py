#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .client import CachedChat
from .mas import run_episode
from .stats import bh_reject, cluster_paired_effect, cluster_rate_ci, effect, paired_sign_pvalue
from .retrieval import BM25Index, role_query, tokenize


def load(path: Path, binary: bool = True) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [x for x in rows if (x.get("label") in {"SUPPORTS", "REFUTES"} if binary else True) and x.get("evidence_bundle")]


def lexical_similarity(a: str, b: str) -> float:
    aa, bb = set(tokenize(a)), set(tokenize(b))
    return len(aa & bb) / max(1, len(aa | bb))


def select_claims(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets = {label: [row for row in rows if row.get("label") == label] for label in ("SUPPORTS", "REFUTES")}
    for bucket in buckets.values(): rng.shuffle(bucket)
    chosen = buckets["SUPPORTS"][: n // 2] + buckets["REFUTES"][: n // 2]
    remainder = [row for row in rows if row not in chosen]; rng.shuffle(remainder)
    chosen.extend(remainder[: n - len(chosen)]); rng.shuffle(chosen)
    return chosen


def retrieve(claim: dict, bank: list[dict], aid: str, k: int, index: BM25Index | None = None) -> list[dict]:
    index = index or BM25Index(bank)
    query, label_bias = role_query(claim, aid)
    return index.search(query, k=k, label_bias=label_bias)


def placebo(item: dict, bank: list[dict], max_similarity: float = .15, max_token_ratio: float = .10) -> dict:
    """Build a format-matched unrelated memory, or mark it invalid.

    The old implementation preferred semantic dissimilarity and only then
    tried to match length, so it could silently violate the registered
    placebo thresholds.  Eligibility is now a hard constraint.
    """
    source_len = len(json.dumps(item, ensure_ascii=False).split())
    scored = []
    for candidate in bank:
        if candidate["memory_id"] == item["memory_id"]:
            continue
        similarity = lexical_similarity(item["claim"], candidate["claim"])
        candidate_len = len(json.dumps(candidate, ensure_ascii=False).split())
        token_ratio = abs(candidate_len - source_len) / max(1, source_len)
        scored.append((similarity, token_ratio, str(candidate["memory_id"]), candidate))
    if not scored:
        p = dict(item)
        p["memory_id"] = "placebo-" + item["memory_id"]
        p["rationale_digest"] = "Format-matched unrelated precedent unavailable."
        p["placebo_for"] = item["memory_id"]
        p["placebo_similarity"] = 1.0
        p["placebo_token_ratio"] = 1.0
        p["placebo_valid"] = False
        return p
    eligible = [row for row in scored if row[0] < max_similarity and row[1] <= max_token_ratio]
    if eligible:
        similarity, token_ratio, _, target = min(eligible, key=lambda row: (row[0], row[1], row[2]))
        p = dict(target)
        p["placebo_valid"] = True
    else:
        # Preserve diagnostics for the gate, but never use this item in an arm.
        similarity, token_ratio, _, target = min(scored, key=lambda row: (row[0], row[1], row[2]))
        p = dict(target)
        p["placebo_valid"] = False
    p["memory_id"] = "placebo-" + item["memory_id"]
    p["rationale_digest"] = "Format-matched unrelated precedent."
    p["placebo_for"] = item["memory_id"]
    p["placebo_similarity"] = float(similarity)
    p["placebo_token_ratio"] = float(token_ratio)
    return p


def _legacy_main() -> None:
    p = argparse.ArgumentParser(description="Real LLM FEVER local/team causal mismatch experiment")
    p.add_argument("--test", type=Path, required=True, help="Enriched FEVER dev JSONL")
    p.add_argument("--memory-bank", type=Path, required=True, help="Frozen bank JSONL from build_memory_bank.py")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--claims", type=int, default=20); p.add_argument("--repeats", type=int, default=5); p.add_argument("--top-k", type=int, choices=(1,), default=1, help="Number of memories injected per agent; fixed to top-1 for this experiment"); p.add_argument("--audit-top-n", type=int, choices=(1,), default=1)
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    claims = select_claims(load(args.test), args.claims, args.seed); bank = load(args.memory_bank, binary=False)
    bank_md5 = hashlib.md5(args.memory_bank.read_bytes()).hexdigest()
    config_md5 = hashlib.md5(json.dumps({"model": args.model, "claims": args.claims, "repeats": args.repeats, "top_k": args.top_k, "audit_top_n": args.audit_top_n, "seed": args.seed, "endpoint": args.endpoint}, sort_keys=True).encode()).hexdigest()
    placebo_lookup = {}
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    index = BM25Index(bank)
    log_path = args.output_dir / "episode_runs.jsonl"
    try:
        log_path.open("x", encoding="utf-8").close()
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing log: {log_path}. Use a new --output-dir.") from exc
    all_runs = []; summary = []; excluded_units = 0
    for claim in claims:
        candidates = {a: retrieve(claim, bank, a, args.top_k, index) for a in ("A1", "A2", "A3")}
        for item in {x["memory_id"]: x for values in candidates.values() for x in values}.values():
            placebo_lookup.setdefault(item["memory_id"], placebo(item, bank))
        # Start with the most relevant candidate; all retrieved candidates remain logged.
        for item in candidates["A1"][:args.audit_top_n]:
            if not placebo_lookup[item["memory_id"]].get("placebo_valid", False):
                excluded_units += 1
                continue
            unit = (str(claim["id"]), "A1", item["memory_id"])
            start = len(all_runs)
            for r in range(args.repeats):
                control = run_episode(client, claim, candidates, r, "control", unit, placebo_lookup)
                treated = run_episode(client, claim, candidates, r, "treated", unit, placebo_lookup)
                pair = [{"branch": "control", "config_md5": config_md5, "bank_md5": bank_md5, **control.__dict__}, {"branch": "treated", "config_md5": config_md5, "bank_md5": bank_md5, **treated.__dict__}]
                all_runs += pair
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(json.dumps(x, ensure_ascii=False) for x in pair) + "\n")
            tc = all_runs[start::2]; tt = all_runs[start + 1::2]
            ld = [int(t["per_agent_correct_r1"]["A1"]) - int(c["per_agent_correct_r1"]["A1"]) for t, c in zip(tt, tc)]
            ad = [int(t["per_agent_correct_solo"]["A1"]) - int(c["per_agent_correct_solo"]["A1"]) for t, c in zip(tt, tc)]
            td = [int(t["team_correct"]) - int(c["team_correct"]) for t, c in zip(tt, tc)]
            r1_team_t = [sum(r["per_agent_correct_r1"].values()) >= 2 for r in tt]
            r1_team_c = [sum(r["per_agent_correct_r1"].values()) >= 2 for r in tc]
            r2_delta = [int(t["team_correct"]) - int(t["round1"]["correct"]) if "correct" in t.get("round1", {}) else 0 for t in tt]
            lp, lci = effect(ld, args.bootstrap, args.seed); ap, aci = effect(ad, args.bootstrap, args.seed + 1); tp, tci = effect(td, args.bootstrap, args.seed + 2)
            classification = "local_positive_team_negative" if lp > 0 and tp < 0 else "local_negative_team_positive" if lp < 0 and tp > 0 else "other"
            summary.append({"claim_id": str(claim["id"]), "memory_id": item["memory_id"], "local_b": lp, "local_b_ci": lci, "local_a": ap, "local_a_ci": aci, "team": tp, "team_ci": tci, "round1_team": sum(r1_team_t) / max(1, len(r1_team_t)) - sum(r1_team_c) / max(1, len(r1_team_c)), "round2_team": tp - (sum(r1_team_t) / max(1, len(r1_team_t)) - sum(r1_team_c) / max(1, len(r1_team_c))), "local_p": paired_sign_pvalue(ld), "team_p": paired_sign_pvalue(td), "classification": classification})
    local_reject = bh_reject([x["local_p"] for x in summary]); team_reject = bh_reject([x["team_p"] for x in summary])
    for row, local_ok, team_ok in zip(summary, local_reject, team_reject):
        row["local_bh_reject"] = local_ok; row["team_bh_reject"] = team_ok
        row["confirmed"] = bool(local_ok and team_ok and ((row["local_b_ci"][0] > 0 and row["team_ci"][1] < 0) or (row["local_b_ci"][1] < 0 and row["team_ci"][0] > 0)))
    mismatches = [x for x in summary if x["confirmed"]]
    rate = len(mismatches) / len(summary) if summary else 0.0
    result = {"experiment": "p2_probe_real_llm", "benchmark": "FEVER_binary", "retrieval_method": "bm25_claim_only", "model": args.model, "n_claims": len(claims), "n_claims_input": len(claims), "n_audit_units": len(summary), "repeats": args.repeats, "fdr_q": 0.1, "mismatch_count": len(mismatches), "mismatch_rate": rate, "mismatch_rate_ci": cluster_rate_ci(summary, args.bootstrap, args.seed + 3), "direction_i_count": sum(x["confirmed"] and x["classification"] == "local_positive_team_negative" for x in summary), "direction_ii_count": sum(x["confirmed"] and x["classification"] == "local_negative_team_positive" for x in summary), "undetermined_count": len(summary) - len(mismatches), "cache_hits": client.cache_hits, "llm_calls": client.calls, "memory_bank_md5": bank_md5, "config_md5": config_md5, "units": summary}
    result["aggregate_effects"] = {
        "local_b": {"estimate": cluster_paired_effect([[row["local_b"]] for row in summary], args.bootstrap, args.seed + 10)[0], "ci": cluster_paired_effect([[row["local_b"]] for row in summary], args.bootstrap, args.seed + 10)[1]},
        "local_a": {"estimate": cluster_paired_effect([[row["local_a"]] for row in summary], args.bootstrap, args.seed + 11)[0], "ci": cluster_paired_effect([[row["local_a"]] for row in summary], args.bootstrap, args.seed + 11)[1]},
        "team": {"estimate": cluster_paired_effect([[row["team"]] for row in summary], args.bootstrap, args.seed + 12)[0], "ci": cluster_paired_effect([[row["team"]] for row in summary], args.bootstrap, args.seed + 12)[1]},
        "round1_team": {"estimate": cluster_paired_effect([[row["round1_team"]] for row in summary], args.bootstrap, args.seed + 13)[0], "ci": cluster_paired_effect([[row["round1_team"]] for row in summary], args.bootstrap, args.seed + 13)[1]},
        "round2_increment": {"estimate": cluster_paired_effect([[row["round2_team"]] for row in summary], args.bootstrap, args.seed + 14)[0], "ci": cluster_paired_effect([[row["round2_team"]] for row in summary], args.bootstrap, args.seed + 14)[1]},
    }
    (args.output_dir / "mismatch_rate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs = [output for run in all_runs for section in ("round1", "round2", "solo") for output in run.get(section, {}).values()]
    parse_rate = sum(bool(x.get("parse_fail")) for x in outputs) / max(1, len(outputs))
    unaffected_equal = all(control["round1"][aid] == treated["round1"][aid] for control, treated in zip(all_runs[::2], all_runs[1::2]) for aid in ("A2", "A3"))
    diagnostics = [value for run in all_runs if run["branch"] == "control" for value in run["placebo_diagnostics"].values()]
    placebo_ok = bool(diagnostics) and all(value["similarity"] < .15 and value["token_ratio"] <= .10 for value in diagnostics)
    treated_runs = [run for run in all_runs if run["branch"] == "treated"]
    role_diversity = sum(len({tuple(run["candidates"][aid]) for aid in ("A1", "A2", "A3")}) > 1 for run in treated_runs) / max(1, len(treated_runs))
    comparable = [row for row in summary if row["local_a"] != 0 or row["local_b"] != 0]
    sign_agreement = (sum((row["local_a"] > 0) == (row["local_b"] > 0) and (row["local_a"] < 0) == (row["local_b"] < 0) for row in comparable) / len(comparable)) if comparable else None
    sign_text = f"{sign_agreement:.2%}" if sign_agreement is not None else "NOT ESTIMABLE: no non-zero local effects"
    gate = (
        "# Real-LLM FEVER Gate Report\n\n"
        f"- Real evidence present for selected claims: {'PASS' if all(x.get('evidence_bundle') for x in claims) else 'FAIL'}\n"
        f"- Frozen memory-bank MD5 recorded: PASS (`{result['memory_bank_md5']}`)\n"
        f"- Unaffected A2/A3 round-1 outputs identical across paired arms: {'PASS' if unaffected_equal else 'FAIL'}\n"
        f"- Placebo similarity < 0.15 and token difference <= 10%: {'PASS' if placebo_ok else 'FAIL'}\n"
        f"- Role-specific candidate sets differ: {'PASS' if role_diversity >= .8 else 'FAIL'} ({role_diversity:.2%})\n"
        f"- Structured-output parse failure rate <= 0.5%: {'PASS' if parse_rate <= .005 else 'FAIL'} ({parse_rate:.4%})\n"
        f"- Local A/B sign agreement >= 80%: {'PASS' if sign_agreement is not None and sign_agreement >= .8 else 'FAIL'} ({sign_text})\n"
        f"- BH-FDR correction q=0.1 applied: PASS\n"
        f"- Confirmed mismatch: **{len(mismatches)} / {len(summary)} ({rate:.3%})**\n"
    )
    (args.output_dir / "gate_report.md").write_text(gate, encoding="utf-8")
    result["excluded_invalid_placebo_units"] = excluded_units
    result["n_claims_with_valid_audit"] = len({row["claim_id"] for row in summary})
    result["comparable_local_effect_units"] = len(comparable)
    result["local_sign_agreement"] = sign_agreement
    (args.output_dir / "mismatch_rate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_claims", "n_claims_with_valid_audit", "n_audit_units", "mismatch_count", "mismatch_rate", "undetermined_count", "excluded_invalid_placebo_units", "comparable_local_effect_units", "local_sign_agreement", "cache_hits", "llm_calls")}, indent=2))


def main() -> None:
    """Stable public entry point for the focused E1 experiment."""
    from .run_experiment_e1 import main as e1_main
    e1_main()


if __name__ == "__main__":
    main()
