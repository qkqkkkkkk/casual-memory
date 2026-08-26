#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from .client import CachedChat
from .mas import run_episode
from .stats import bh_reject, cluster_rate_ci, effect, paired_sign_pvalue


def sim(a: str, b: str) -> float:
    aa, bb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(aa & bb) / max(1, len(aa | bb))


def load(path: Path, binary: bool = True) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [x for x in rows if (x.get("label") in {"SUPPORTS", "REFUTES"} if binary else True) and x.get("evidence_bundle")]


def select_claims(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets = {label: [row for row in rows if row.get("label") == label] for label in ("SUPPORTS", "REFUTES")}
    for bucket in buckets.values(): rng.shuffle(bucket)
    chosen = buckets["SUPPORTS"][: n // 2] + buckets["REFUTES"][: n // 2]
    remainder = [row for row in rows if row not in chosen]; rng.shuffle(remainder)
    chosen.extend(remainder[: n - len(chosen)]); rng.shuffle(chosen)
    return chosen


def retrieve(claim: dict, bank: list[dict], aid: str, k: int) -> list[dict]:
    claim_text = claim["claim"]
    evidence_text = " ".join(str(x.get("text", "")) for x in claim.get("evidence_bundle", []))

    def score(item: dict) -> tuple[float, str]:
        if aid == "A1":
            memory_evidence = " ".join(str(x.get("text", "")) for x in item.get("evidence_bundle", []))
            value = sim(claim_text + " " + evidence_text, item["claim"] + " " + memory_evidence)
        elif aid == "A2":
            value = sim(claim_text, item["claim"]) + (0.05 if item.get("gold_label") == "REFUTES" else 0.0)
        else:
            value = sim(claim_text, item["claim"])
        return value, str(item["memory_id"])

    return sorted(bank, key=score, reverse=True)[:k]


def placebo(item: dict, bank: list[dict]) -> dict:
    candidates = sorted((x for x in bank if x["memory_id"] != item["memory_id"]), key=lambda x: (sim(item["claim"], x["claim"]), str(x["memory_id"])))
    dissimilar = candidates[:max(20, len(candidates) // 10)]
    target_len = len(json.dumps(item, ensure_ascii=False).split())
    target = min(dissimilar, key=lambda x: abs(len(json.dumps(x, ensure_ascii=False).split()) - target_len))
    replacement_len = len(json.dumps(target, ensure_ascii=False).split())
    p = dict(target); p["memory_id"] = "placebo-" + item["memory_id"]; p["rationale_digest"] = "Format-matched unrelated precedent."
    p["placebo_for"] = item["memory_id"]; p["placebo_similarity"] = sim(item["claim"], target["claim"]); p["placebo_token_ratio"] = abs(replacement_len - target_len) / max(1, target_len)
    return p


def main() -> None:
    p = argparse.ArgumentParser(description="Real LLM FEVER local/team causal mismatch experiment")
    p.add_argument("--test", type=Path, required=True, help="Enriched FEVER dev JSONL")
    p.add_argument("--memory-bank", type=Path, required=True, help="Frozen bank JSONL from build_memory_bank.py")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--claims", type=int, default=20); p.add_argument("--repeats", type=int, default=5); p.add_argument("--top-k", type=int, default=6); p.add_argument("--audit-top-n", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=42); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    claims = select_claims(load(args.test), args.claims, args.seed); bank = load(args.memory_bank, binary=False)
    bank_md5 = hashlib.md5(args.memory_bank.read_bytes()).hexdigest()
    config_md5 = hashlib.md5(json.dumps({"model": args.model, "claims": args.claims, "repeats": args.repeats, "top_k": args.top_k, "audit_top_n": args.audit_top_n, "seed": args.seed, "endpoint": args.endpoint}, sort_keys=True).encode()).hexdigest()
    placebo_lookup = {}
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    log_path = args.output_dir / "episode_runs.jsonl"
    try:
        log_path.open("x", encoding="utf-8").close()
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing log: {log_path}. Use a new --output-dir.") from exc
    all_runs = []; summary = []
    for claim in claims:
        candidates = {a: retrieve(claim, bank, a, args.top_k) for a in ("A1", "A2", "A3")}
        for item in {x["memory_id"]: x for values in candidates.values() for x in values}.values():
            placebo_lookup.setdefault(item["memory_id"], placebo(item, bank))
        # Start with the most relevant candidate; all retrieved candidates remain logged.
        for item in candidates["A1"][:args.audit_top_n]:
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
            lp, lci = effect(ld, args.bootstrap, args.seed); ap, aci = effect(ad, args.bootstrap, args.seed + 1); tp, tci = effect(td, args.bootstrap, args.seed + 2)
            classification = "local_positive_team_negative" if lp > 0 and tp < 0 else "local_negative_team_positive" if lp < 0 and tp > 0 else "other"
            summary.append({"claim_id": str(claim["id"]), "memory_id": item["memory_id"], "local_b": lp, "local_b_ci": lci, "local_a": ap, "local_a_ci": aci, "team": tp, "team_ci": tci, "local_p": paired_sign_pvalue(ld), "team_p": paired_sign_pvalue(td), "classification": classification})
    local_reject = bh_reject([x["local_p"] for x in summary]); team_reject = bh_reject([x["team_p"] for x in summary])
    for row, local_ok, team_ok in zip(summary, local_reject, team_reject):
        row["local_bh_reject"] = local_ok; row["team_bh_reject"] = team_ok
        row["confirmed"] = bool(local_ok and team_ok and ((row["local_b_ci"][0] > 0 and row["team_ci"][1] < 0) or (row["local_b_ci"][1] < 0 and row["team_ci"][0] > 0)))
    mismatches = [x for x in summary if x["confirmed"]]
    rate = len(mismatches) / len(summary) if summary else 0.0
    result = {"experiment": "p2_probe_real_llm", "benchmark": "FEVER_binary", "model": args.model, "n_claims": len(claims), "n_audit_units": len(summary), "repeats": args.repeats, "fdr_q": 0.1, "mismatch_count": len(mismatches), "mismatch_rate": rate, "mismatch_rate_ci": cluster_rate_ci(summary, args.bootstrap, args.seed + 3), "direction_i_count": sum(x["confirmed"] and x["classification"] == "local_positive_team_negative" for x in summary), "direction_ii_count": sum(x["confirmed"] and x["classification"] == "local_negative_team_positive" for x in summary), "undetermined_count": len(summary) - len(mismatches), "cache_hits": client.cache_hits, "llm_calls": client.calls, "memory_bank_md5": bank_md5, "config_md5": config_md5, "units": summary}
    (args.output_dir / "mismatch_rate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs = [output for run in all_runs for section in ("round1", "round2", "solo") for output in run.get(section, {}).values()]
    parse_rate = sum(bool(x.get("parse_fail")) for x in outputs) / max(1, len(outputs))
    unaffected_equal = all(control["round1"][aid] == treated["round1"][aid] for control, treated in zip(all_runs[::2], all_runs[1::2]) for aid in ("A2", "A3"))
    diagnostics = [value for run in all_runs if run["branch"] == "control" for value in run["placebo_diagnostics"].values()]
    placebo_ok = bool(diagnostics) and all(value["similarity"] < .15 and value["token_ratio"] <= .10 for value in diagnostics)
    treated_runs = [run for run in all_runs if run["branch"] == "treated"]
    role_diversity = sum(len({tuple(run["candidates"][aid]) for aid in ("A1", "A2", "A3")}) > 1 for run in treated_runs) / max(1, len(treated_runs))
    comparable = [row for row in summary if row["local_a"] != 0 or row["local_b"] != 0]
    sign_agreement = sum((row["local_a"] > 0) == (row["local_b"] > 0) and (row["local_a"] < 0) == (row["local_b"] < 0) for row in comparable) / max(1, len(comparable))
    gate = (
        "# Real-LLM FEVER Gate Report\n\n"
        f"- Real evidence present for selected claims: {'PASS' if all(x.get('evidence_bundle') for x in claims) else 'FAIL'}\n"
        f"- Frozen memory-bank MD5 recorded: PASS (`{result['memory_bank_md5']}`)\n"
        f"- Unaffected A2/A3 round-1 outputs identical across paired arms: {'PASS' if unaffected_equal else 'FAIL'}\n"
        f"- Placebo similarity < 0.15 and token difference <= 10%: {'PASS' if placebo_ok else 'FAIL'}\n"
        f"- Role-specific candidate sets differ: {'PASS' if role_diversity >= .8 else 'FAIL'} ({role_diversity:.2%})\n"
        f"- Structured-output parse failure rate <= 0.5%: {'PASS' if parse_rate <= .005 else 'FAIL'} ({parse_rate:.4%})\n"
        f"- Local A/B sign agreement >= 80%: {'PASS' if sign_agreement >= .8 else 'FAIL'} ({sign_agreement:.2%})\n"
        f"- BH-FDR correction q=0.1 applied: PASS\n"
        f"- Confirmed mismatch: **{len(mismatches)} / {len(summary)} ({rate:.3%})**\n"
    )
    (args.output_dir / "gate_report.md").write_text(gate, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_claims", "n_audit_units", "mismatch_count", "mismatch_rate", "undetermined_count", "cache_hits", "llm_calls")}, indent=2))


if __name__ == "__main__": main()
