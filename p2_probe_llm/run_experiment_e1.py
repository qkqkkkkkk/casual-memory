#!/usr/bin/env python3
"""Real-LLM FEVER E1: unilateral top-1 memory intervention.

Only A1's retrieved memory is replaced by a placebo in the control arm.
A2/A3 receive exactly the same memories in both paired arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .client import CachedChat
from .mas import _render_memory, run_episode
from .retrieval import BM25Index, GMemorySemanticIndex, role_query, tokenize
from .stats import bh_reject, cluster_paired_effect, cluster_rate_ci, effect, paired_sign_pvalue

AGENTS = ("A1", "A2", "A3")
MEMORY_SCHEMA = "gmemory-fever-v2"


def load(path: Path, binary: bool = True) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if (row.get("label") in {"SUPPORTS", "REFUTES"} if binary else True) and row.get("evidence_bundle")]


def lexical_similarity(a: str, b: str) -> float:
    aa, bb = set(tokenize(a)), set(tokenize(b))
    return len(aa & bb) / max(1, len(aa | bb))


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _memory_source_keys(item: dict[str, Any]) -> set[str]:
    return {str(item[field]) for field in ("memory_id", "source_example_id", "source_id") if item.get(field) is not None}


def _evidence_texts(item: dict[str, Any]) -> set[str]:
    return {_norm_text(entry.get("text", "")) for entry in item.get("evidence_bundle", []) if _norm_text(entry.get("text", ""))}


def _claim_key(item: dict[str, Any]) -> str:
    return _norm_text(item.get("claim", ""))


def schema_coverage(bank: list[dict[str, Any]]) -> float:
    return sum(item.get("memory_schema_version") == MEMORY_SCHEMA for item in bank) / max(1, len(bank))


def memory_is_eligible(claim: dict[str, Any], item: dict[str, Any]) -> bool:
    """Reject a memory that supplied a distractor or exact evidence sentence."""
    forbidden = {str(value) for value in claim.get("evidence_policy", {}).get("distractor_source_ids", [])}
    return not (forbidden & _memory_source_keys(item) or _evidence_texts(claim) & _evidence_texts(item))


def select_claims(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    n = min(max(0, n), len(rows))
    buckets = {label: [row for row in rows if row.get("label") == label] for label in ("SUPPORTS", "REFUTES")}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen = buckets["SUPPORTS"][: n // 2] + buckets["REFUTES"][: n // 2]
    remainder = [row for row in rows if row not in chosen]
    rng.shuffle(remainder)
    chosen.extend(remainder[: n - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def retrieve(claim: dict[str, Any], bank: list[dict[str, Any]], aid: str, k: int, index: BM25Index | GMemorySemanticIndex | None = None) -> list[dict[str, Any]]:
    if not bank or k <= 0:
        return []
    index = index or BM25Index(bank)
    query, label_bias = role_query(claim, aid)
    ranked = index.search_with_scores(query, k=len(bank), label_bias=label_bias)
    eligible = [
        (rank, item, score)
        for rank, (item, score) in enumerate(ranked, 1)
        if memory_is_eligible(claim, item)
    ]
    return [
        dict(item, retrieval_score=score, retrieval_rank=rank)
        for rank, item, score in eligible[:k]
    ]


def placebo(item: dict[str, Any], bank: list[dict[str, Any]], max_similarity: float = .15, max_token_ratio: float = .10, forbidden_evidence: set[str] | None = None) -> dict[str, Any]:
    """Choose an unrelated, length-matched bank item; invalid matches are never used."""
    source_len = len(_render_memory(item).split())
    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for candidate in bank:
        if candidate.get("memory_id") == item.get("memory_id"):
            continue
        if forbidden_evidence and (_evidence_texts(candidate) & forbidden_evidence):
            continue
        similarity = lexical_similarity(item.get("claim", ""), candidate.get("claim", ""))
        rendered_candidate = dict(candidate)
        rendered_candidate.update({
            "memory_id": "placebo-" + str(item.get("memory_id")),
            "rationale_digest": "Format-matched unrelated FEVER precedent.",
        })
        length = len(_render_memory(rendered_candidate).split())
        ratio = abs(length - source_len) / max(1, source_len)
        scored.append((similarity, ratio, str(candidate.get("memory_id", "")), rendered_candidate))
    if not scored:
        replacement = dict(item)
        replacement.update({"memory_id": "placebo-" + str(item.get("memory_id")), "rationale_digest": "Unrelated precedent unavailable.", "placebo_valid": False, "placebo_similarity": 1.0, "placebo_token_ratio": 1.0})
        return replacement
    eligible = [row for row in scored if row[0] < max_similarity and row[1] <= max_token_ratio]
    similarity, ratio, _, replacement = min(eligible or scored, key=lambda row: (row[0], row[1], row[2]))
    replacement = dict(replacement)
    replacement.update({
        "placebo_for": item.get("memory_id"),
        "placebo_similarity": float(similarity), "placebo_token_ratio": float(ratio),
        "placebo_valid": bool(eligible),
    })
    return replacement


def _paired_effects(control: list[dict[str, Any]], treated: list[dict[str, Any]]) -> dict[str, list[int]]:
    if len(control) != len(treated):
        raise ValueError("Control and treated repeats are not paired")
    local_b = [int(t["per_agent_correct_r1"]["A1"]) - int(c["per_agent_correct_r1"]["A1"]) for c, t in zip(control, treated)]
    local_a = [int(t["per_agent_correct_solo"]["A1"]) - int(c["per_agent_correct_solo"]["A1"]) for c, t in zip(control, treated)]
    team = [int(t["team_correct"]) - int(c["team_correct"]) for c, t in zip(control, treated)]
    r1_control = [sum(c["per_agent_correct_r1"].values()) >= 2 for c in control]
    r1_treated = [sum(t["per_agent_correct_r1"].values()) >= 2 for t in treated]
    round1_team = [int(t) - int(c) for c, t in zip(r1_control, r1_treated)]
    increment = [team_delta - r1_delta for team_delta, r1_delta in zip(team, round1_team)]
    return {"local_b": local_b, "local_a": local_a, "team": team, "round1_team": round1_team, "round2_increment": increment}


def _effect(values: list[int], bootstrap: int, seed: int) -> dict[str, Any]:
    estimate, ci = effect(values, bootstrap, seed)
    return {"estimate": estimate, "ci": ci}


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-LLM FEVER E1 unilateral-memory mismatch experiment")
    parser.add_argument("--test", type=Path, required=True, help="Selected enriched FEVER test JSONL")
    parser.add_argument("--experience-bank", type=Path, required=True, help="Frozen retrieval memory bank JSONL")
    parser.add_argument("--distractor-bank", type=Path, required=True, help="Disjoint evidence-distractor pool JSONL")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Same SentenceTransformer model used by G-Memory retrieval")
    parser.add_argument("--retrieval-threshold", type=float, default=0.3, help="Cosine threshold matching G-Memory retrieval")
    parser.add_argument("--claims", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, choices=(1,), default=1, help="E1 is fixed to top-1")
    parser.add_argument("--audit-top-n", type=int, choices=(1,), default=1, help="E1 audits only A1 top-1")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be at least 1")

    claims = select_claims(load(args.test), args.claims, args.seed)
    experience_bank = load(args.experience_bank, binary=False)
    distractor_bank = load(args.distractor_bank, binary=False)
    if not claims:
        raise SystemExit("No eligible claims found in --test")
    if not experience_bank or not distractor_bank:
        raise SystemExit("Both experience and distractor banks must contain evidence-bearing items")
    if schema_coverage(experience_bank) != 1.0 or schema_coverage(distractor_bank) != 1.0:
        raise SystemExit("Banks use the old memory schema; rebuild both with the current build_pools command")
    experience_sources = {source for item in experience_bank for source in _memory_source_keys(item)}
    distractor_sources = {source for item in distractor_bank for source in _memory_source_keys(item)}
    if experience_sources & distractor_sources:
        raise SystemExit("Experience and distractor banks overlap in provenance; rebuild them with build_pools")
    test_claim_keys = {_claim_key(claim) for claim in claims}
    if test_claim_keys & ({_claim_key(item) for item in experience_bank} | {_claim_key(item) for item in distractor_bank}):
        raise SystemExit("A test claim is present in an experience/distractor bank; rebuild pools excluding the test split")

    experience_md5 = hashlib.md5(args.experience_bank.read_bytes()).hexdigest()
    distractor_md5 = hashlib.md5(args.distractor_bank.read_bytes()).hexdigest()
    config_md5 = hashlib.md5(json.dumps({"model": args.model, "claims": args.claims, "repeats": args.repeats, "top_k": args.top_k, "audit_top_n": args.audit_top_n, "seed": args.seed, "endpoint": args.endpoint, "retrieval": "gmemory_semantic_claim", "embedding_model": args.embedding_model, "retrieval_threshold": args.retrieval_threshold, "experiment": "E1"}, sort_keys=True).encode()).hexdigest()
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    index = GMemorySemanticIndex(experience_bank, args.embedding_model, args.retrieval_threshold)
    log_path = args.output_dir / "episode_runs.jsonl"
    try:
        log_path.open("x", encoding="utf-8").close()
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing log: {log_path}. Use a new --output-dir.") from exc

    all_runs: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    excluded_invalid_placebo = excluded_no_eligible_memory = excluded_exact_evidence = 0
    candidate_diagnostics: list[dict[str, Any]] = []
    for claim in claims:
        shared_candidates = retrieve(claim, experience_bank, "A1", args.top_k, index)
        candidates = {aid: list(shared_candidates) for aid in AGENTS}
        if any(not candidates[aid] for aid in AGENTS):
            excluded_no_eligible_memory += 1
            continue
        for aid in AGENTS:
            candidate_diagnostics.append({
                "claim_id": str(claim["id"]),
                "agent_id": aid,
                "memory_id": candidates[aid][0]["memory_id"],
                "retrieval_score": candidates[aid][0].get("retrieval_score"),
                "retrieval_rank": candidates[aid][0].get("retrieval_rank"),
                "eligible": memory_is_eligible(claim, candidates[aid][0]),
            })
        item = candidates["A1"][0]
        if not memory_is_eligible(claim, item):
            excluded_exact_evidence += 1
            continue
        # A placebo is claim-specific because its evidence must also be
        # disjoint from this claim's evidence bundle.
        placebo_lookup = {
            str(item["memory_id"]): placebo(
                item, experience_bank, forbidden_evidence=_evidence_texts(claim)
            )
        }
        if not placebo_lookup[str(item["memory_id"])].get("placebo_valid", False):
            excluded_invalid_placebo += 1
            continue

        audit_unit = (str(claim["id"]), "A1", str(item["memory_id"]))
        control_runs: list[dict[str, Any]] = []
        treated_runs: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            control = run_episode(client, claim, candidates, repeat, "control", audit_unit, placebo_lookup)
            treated = run_episode(client, claim, candidates, repeat, "treated", audit_unit, placebo_lookup)
            control_row = {"branch": "control", "config_md5": config_md5, "experience_bank_md5": experience_md5, "distractor_bank_md5": distractor_md5, **control.__dict__}
            treated_row = {"branch": "treated", "config_md5": config_md5, "experience_bank_md5": experience_md5, "distractor_bank_md5": distractor_md5, **treated.__dict__}
            control_runs.append(control_row); treated_runs.append(treated_row); all_runs.extend((control_row, treated_row))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(control_row, ensure_ascii=False) + "\n")
                handle.write(json.dumps(treated_row, ensure_ascii=False) + "\n")

        diffs = _paired_effects(control_runs, treated_runs)
        ordinal = len(summary) * 10
        b = _effect(diffs["local_b"], args.bootstrap, args.seed + ordinal)
        a = _effect(diffs["local_a"], args.bootstrap, args.seed + ordinal + 1)
        team = _effect(diffs["team"], args.bootstrap, args.seed + ordinal + 2)
        r1 = _effect(diffs["round1_team"], args.bootstrap, args.seed + ordinal + 3)
        inc = _effect(diffs["round2_increment"], args.bootstrap, args.seed + ordinal + 4)
        classification = "local_positive_team_negative" if b["estimate"] > 0 and team["estimate"] < 0 else "local_negative_team_positive" if b["estimate"] < 0 and team["estimate"] > 0 else "other"
        summary.append({
            "claim_id": str(claim["id"]), "agent_id": "A1", "memory_id": str(item["memory_id"]),
            "local_b": b["estimate"], "local_b_ci": b["ci"], "local_a": a["estimate"], "local_a_ci": a["ci"],
            "team": team["estimate"], "team_ci": team["ci"], "round1_team": r1["estimate"], "round1_team_ci": r1["ci"],
            "round2_increment": inc["estimate"], "round2_increment_ci": inc["ci"],
            "local_b_diffs": diffs["local_b"], "local_a_diffs": diffs["local_a"], "team_diffs": diffs["team"],
            "round1_team_diffs": diffs["round1_team"], "round2_increment_diffs": diffs["round2_increment"],
            "local_p": paired_sign_pvalue(diffs["local_b"]), "team_p": paired_sign_pvalue(diffs["team"]), "classification": classification,
        })

    local_reject = bh_reject([row["local_p"] for row in summary])
    team_reject = bh_reject([row["team_p"] for row in summary])
    for row, local_ok, team_ok in zip(summary, local_reject, team_reject):
        opposite = (row["local_b_ci"][0] > 0 and row["team_ci"][1] < 0) or (row["local_b_ci"][1] < 0 and row["team_ci"][0] > 0)
        row["local_bh_reject"] = local_ok; row["team_bh_reject"] = team_ok; row["confirmed"] = bool(local_ok and team_ok and opposite)
    mismatches = [row for row in summary if row["confirmed"]]
    aggregate = {key: dict(zip(("estimate", "ci"), cluster_paired_effect([row[f"{key}_diffs"] for row in summary], args.bootstrap, args.seed + offset))) for key, offset in (("local_b", 10), ("local_a", 11), ("team", 12), ("round1_team", 13), ("round2_increment", 14))}
    comparable = [row for row in summary if row["local_a"] != 0 and row["local_b"] != 0]
    sign_agreement = sum((row["local_a"] > 0) == (row["local_b"] > 0) for row in comparable) / len(comparable) if comparable else None
    a1_memory_use = {}
    for branch in ("control", "treated"):
        values = [run["round1"]["A1"] for run in all_runs if run["branch"] == branch]
        a1_memory_use[branch] = {
            "runs": len(values),
            "completion_rate": sum(bool(value.get("memory_considered")) for value in values) / max(1, len(values)),
            "high_relevance_rate": sum(value.get("memory_relevance") == "HIGH" for value in values) / max(1, len(values)),
            "adopted_or_partial_rate": sum(value.get("memory_influence") in {"ADOPTED", "PARTIAL"} for value in values) / max(1, len(values)),
            "rejected_rate": sum(value.get("memory_influence") == "REJECTED" for value in values) / max(1, len(values)),
        }
    retrieval_scores = [
        float(row["retrieval_score"])
        for row in candidate_diagnostics
        if row["agent_id"] == "A1" and row.get("retrieval_score") is not None
    ]
    retrieval_score_summary = {
        "min": min(retrieval_scores, default=None),
        "mean": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        "max": max(retrieval_scores, default=None),
    }
    eligible_memory_coverage = len({row["claim_id"] for row in summary}) / max(1, len(claims))
    result: dict[str, Any] = {
        "experiment": "p2_probe_real_llm_E1", "benchmark": "FEVER_binary", "retrieval_method": "gmemory_semantic_claim", "embedding_model": args.embedding_model, "retrieval_threshold": args.retrieval_threshold, "model": args.model,
        "n_claims": len(claims), "n_claims_input": len(claims), "n_claims_with_valid_audit": len({row["claim_id"] for row in summary}), "n_audit_units": len(summary), "repeats": args.repeats,
        "fdr_q": .1, "mismatch_count": len(mismatches), "mismatch_rate": len(mismatches) / len(summary) if summary else 0.0,
        "mismatch_rate_ci": cluster_rate_ci(summary, args.bootstrap, args.seed + 3) if summary else [0.0, 0.0], "direction_i_count": sum(row["confirmed"] and row["classification"] == "local_positive_team_negative" for row in summary), "direction_ii_count": sum(row["confirmed"] and row["classification"] == "local_negative_team_positive" for row in summary), "undetermined_count": len(summary) - len(mismatches),
        "eligible_memory_coverage": eligible_memory_coverage, "excluded_invalid_placebo_units": excluded_invalid_placebo, "excluded_no_eligible_memory_units": excluded_no_eligible_memory, "excluded_exact_evidence_units": excluded_exact_evidence,
        "comparable_local_effect_units": len(comparable), "local_sign_agreement": sign_agreement, "cache_hits": client.cache_hits, "llm_calls": client.calls,
        "experience_bank_md5": experience_md5, "distractor_bank_md5": distractor_md5, "config_md5": config_md5, "candidate_diagnostics": candidate_diagnostics, "aggregate_effects": aggregate, "units": summary,
        "a1_round1_memory_use": a1_memory_use, "retrieval_score_summary": retrieval_score_summary,
    }

    outputs = [output for run in all_runs for section in ("round1", "round2", "solo") for output in run.get(section, {}).values()]
    parse_rate = sum(bool(output.get("parse_fail")) for output in outputs) / max(1, len(outputs))
    memory_considered_rate = sum(bool(output.get("memory_considered")) for output in outputs) / max(1, len(outputs))
    paired = list(zip(all_runs[::2], all_runs[1::2]))
    unaffected_equal = bool(paired) and all(control["round1"][aid] == treated["round1"][aid] for control, treated in paired for aid in ("A2", "A3"))
    shared_retrieval = bool(all_runs) and all(len({tuple(run["candidates"][aid]) for aid in AGENTS}) == 1 for run in all_runs if run["branch"] == "treated")
    result["memory_considered_rate"] = memory_considered_rate
    result["shared_retrieval_profile"] = shared_retrieval
    diagnostics = [value for run in all_runs if run["branch"] == "control" for value in run["placebo_diagnostics"].values()]
    placebo_ok = bool(diagnostics) and all(value["similarity"] < .15 and value["token_ratio"] <= .10 for value in diagnostics)
    sign_text = f"{sign_agreement:.2%}" if sign_agreement is not None else "no units with both non-zero local effects"
    gate = (
        "# Real-LLM FEVER E1 Gate Report\n\n"
        f"- Real evidence present for selected claims: {'PASS' if all(row.get('evidence_bundle') for row in claims) else 'FAIL'}\n"
        f"- Frozen experience-bank MD5 recorded: PASS (`{experience_md5}`)\n"
        f"- Frozen distractor-bank MD5 recorded: PASS (`{distractor_md5}`)\n"
        f"- Eligible semantic top-1 coverage >= 95%: {'PASS' if eligible_memory_coverage >= .95 else 'FAIL'} ({eligible_memory_coverage:.2%}; no candidate {excluded_no_eligible_memory})\n"
        f"- Retrieved memory evidence has no exact claim-evidence overlap: {'PASS' if excluded_exact_evidence == 0 else 'FAIL'} (excluded {excluded_exact_evidence})\n"
        f"- Unaffected A2/A3 round-1 outputs identical across paired arms: {'PASS' if unaffected_equal else 'FAIL'}\n"
        f"- One shared claim-level retrieval profile before intervention: {'PASS' if shared_retrieval else 'FAIL'}\n"
        f"- Placebo similarity < 0.15 and token difference <= 10%: {'PASS' if placebo_ok else 'FAIL'}\n"
        f"- Structured-output parse failure rate <= 0.5%: {'PASS' if parse_rate <= .005 else 'FAIL'} ({parse_rate:.4%})\n"
        f"- Structured memory-use report completion rate >= 95%: {'PASS' if memory_considered_rate >= .95 else 'FAIL'} ({memory_considered_rate:.2%})\n"
        f"- A1 treated memory adopted/partially adopted: {a1_memory_use['treated']['adopted_or_partial_rate']:.2%}\n"
        f"- A1 control placebo adopted/partially adopted: {a1_memory_use['control']['adopted_or_partial_rate']:.2%}\n"
        f"- Retrieved top-1 cosine score summary: `{retrieval_score_summary}`\n"
        f"- Local A/B sign agreement >= 80%: {'PASS' if sign_agreement is not None and sign_agreement >= .8 else 'NOT ESTIMABLE' if sign_agreement is None else 'FAIL'} ({sign_text})\n"
        "- BH-FDR correction q=0.1 applied: PASS\n"
        f"- Confirmed mismatch: **{len(mismatches)} / {len(summary)} ({result['mismatch_rate']:.3%})**\n"
        f"- Aggregate local B effect: `{aggregate['local_b']}`\n"
        f"- Aggregate team effect: `{aggregate['team']}`\n"
    )
    (args.output_dir / "gate_report.md").write_text(gate, encoding="utf-8")
    (args.output_dir / "mismatch_rate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("n_claims", "n_claims_with_valid_audit", "n_audit_units", "mismatch_count", "mismatch_rate", "undetermined_count", "excluded_invalid_placebo_units", "excluded_no_eligible_memory_units", "excluded_exact_evidence_units", "comparable_local_effect_units", "local_sign_agreement", "cache_hits", "llm_calls")}, indent=2))


if __name__ == "__main__":
    main()
