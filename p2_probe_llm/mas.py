from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .client import CachedChat


ROLES = {
    "A1": "Evidence Analyst: compare each evidence sentence with the claim literally; do not fill missing facts.",
    "A2": "Skeptic: actively search for contradictions and evidence gaps; do not accept a claim without support.",
    "A3": "Precedent Reasoner: transfer the retrieved precedent carefully and resolve any conflict with current evidence.",
}


@dataclass
class Run:
    claim_id: str; arm: str; repeat_idx: int; audit_unit: tuple[str, str, str] | None
    candidates: dict[str, list[str]]; placebo_map: dict[str, str]; placebo_diagnostics: dict[str, dict[str, float]]
    round1: dict[str, dict[str, Any]]; round2: dict[str, dict[str, Any]]; solo: dict[str, dict[str, Any]]
    team_verdict: str; team_correct: bool
    per_agent_correct_r1: dict[str, bool]; per_agent_correct_r2: dict[str, bool]
    per_agent_correct_solo: dict[str, bool]
    cache_hits: int = 0; llm_calls: int = 0


def _parse_verdict(obj: dict[str, Any]) -> str:
    value = str(obj.get("verdict", "")).upper().strip()
    obj["_parse_fail"] = value not in {"SUPPORTS", "REFUTES"}
    if value not in {"SUPPORTS", "REFUTES"}:
        text = json.dumps(obj).upper()
        value = "SUPPORTS" if "SUPPORTS" in text and "REFUTES" not in text else "REFUTES"
    return value


def _render_memory(item: dict[str, Any]) -> str:
    evidence = "\n".join(
        f"- {x.get('title', 'unknown')}: {x.get('text', '')}"
        for x in item.get("evidence_bundle", [])
    )
    description = item.get("task_description") or f"Claim: {item.get('claim', '')}"
    key_steps = item.get("key_steps") or "Compare the historical evidence with its claim."
    trajectory = item.get("task_trajectory") or f"Historical evidence reviewed:\n{evidence[:2400]}"
    return (
        f"### Reference case {item['memory_id']}\n"
        f"Task description:\n{description}\n\n"
        f"Key steps:\n{key_steps}\n\n"
        f"Detailed trajectory:\n{trajectory}\n\n"
        f"Recorded result: {item['gold_label']}\n"
        f"Transfer note: {item.get('rationale_digest', '')}"
    )


def _jaccard(a: str, b: str) -> float:
    aa, bb = set(a.lower().split()), set(b.lower().split())
    return len(aa & bb) / max(1, len(aa | bb))


def _make_placebo(target: dict[str, Any], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pool = [x for values in candidates.values() for x in values
            if x.get("memory_id") != target.get("memory_id") and x.get("claim") != target.get("claim")]
    if not pool:
        replacement = dict(target)
        replacement["evidence_bundle"] = []
    else:
        replacement = dict(min(pool, key=lambda x: (_jaccard(target["claim"], x.get("claim", "")), str(x.get("memory_id")))))
    replacement["memory_id"] = f"placebo-{target['memory_id']}"
    replacement["rationale_digest"] = "Format-matched unrelated FEVER precedent."
    return replacement


def _prompt(role: str, claim: dict[str, Any], memories: list[dict[str, Any]], peers: str = "", context_heading: str = "Other agents' round-1 statements") -> list[dict[str, str]]:
    evidence = "\n".join(f"- {x.get('title','unknown')}: {x.get('text','')}" for x in claim.get("evidence_bundle", [])) or "(no resolved evidence)"
    memory_text = "\n".join(_render_memory(x) for x in memories) or "(no memory)"
    system = (
        f"You are {role}. Return JSON only: "
        "{\"memory_relevance\": \"HIGH\" or \"MEDIUM\" or \"LOW\", "
        "\"memory_influence\": \"ADOPTED\" or \"PARTIAL\" or \"REJECTED\", "
        "\"memory_assessment\": string, \"verdict\": \"SUPPORTS\" or \"REFUTES\", "
        "\"confidence\": number, \"rationale\": string}. "
        "All three memory fields are required."
    )
    user = (
        "## Successful historical example (PRIMARY REFERENCE; MANDATORY TO CONSIDER)\n"
        "Start your analysis from this retrieved case. Compare its claim relation, evidence pattern, key steps, and recorded result with the current task. "
        "Use it as a strong but defeasible prior: transfer it when the fact pattern is analogous, and explicitly reject or limit it when the current evidence conflicts. "
        "Do not ignore it, but do not copy its verdict merely because its wording is similar.\n"
        f"{memory_text}\n\n"
        f"## Your current task\nClaim: {claim['claim']}\nEvidence:\n{evidence}\n"
    )
    if peers: user += f"{context_heading}:\n{peers}\n"
    user += (
        "First fill memory_relevance, memory_influence, and memory_assessment. "
        "Then decide whether the current claim is supported or refuted. In the rationale, state exactly how the reference case affected or failed to affect your decision."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _agent_output(obj: dict[str, Any], gold_label: str) -> dict[str, Any]:
    verdict = _parse_verdict(obj)
    assessment = str(obj.get("memory_assessment", "")).strip()
    relevance = str(obj.get("memory_relevance", "")).upper().strip()
    influence = str(obj.get("memory_influence", "")).upper().strip()
    considered = (
        bool(assessment)
        and relevance in {"HIGH", "MEDIUM", "LOW"}
        and influence in {"ADOPTED", "PARTIAL", "REJECTED"}
    )
    return {
        "verdict": verdict,
        "confidence": obj.get("confidence"),
        "rationale": str(obj.get("rationale", "")),
        "memory_relevance": relevance,
        "memory_influence": influence,
        "memory_assessment": assessment,
        "memory_considered": considered,
        "correct": verdict == gold_label,
        "parse_fail": obj["_parse_fail"],
    }


def run_episode(client: CachedChat, claim: dict[str, Any], candidates: dict[str, list[dict[str, Any]]], repeat_idx: int, arm: str, audit_unit: tuple[str, str, str] | None = None, placebo_lookup: dict[str, dict[str, Any]] | None = None) -> Run:
    actual = {a: list(ms) for a, ms in candidates.items()}; placebo_map = {}; placebo_diagnostics = {}
    if arm == "control" and audit_unit:
        aid, mid = audit_unit[1], audit_unit[2]
        for idx, item in enumerate(actual[aid]):
            if item["memory_id"] == mid:
                replacement = dict((placebo_lookup or {}).get(mid) or _make_placebo(item, candidates))
                replacement["memory_id"] = f"placebo-{mid}"
                actual[aid][idx] = replacement; placebo_map[mid] = replacement["memory_id"]
                placebo_diagnostics[mid] = {"similarity": float(replacement.get("placebo_similarity", _jaccard(item["claim"], replacement.get("claim", "")))), "token_ratio": float(replacement.get("placebo_token_ratio", 0.0))}
    r1 = {}; hits0 = client.cache_hits; calls0 = client.calls
    for aid in ("A1", "A2", "A3"):
        obj, _ = client.ask(_prompt(ROLES[aid], claim, actual[aid]), repeat_idx)
        r1[aid] = _agent_output(obj, claim["label"])
    r2 = {}
    for aid in ("A1", "A2", "A3"):
        peers = "\n".join(f"{a}: {r1[a]['verdict']} ({r1[a]['rationale']})" for a in ("A1", "A2", "A3") if a != aid)
        obj, _ = client.ask(_prompt(ROLES[aid], claim, actual[aid], peers), repeat_idx)
        r2[aid] = _agent_output(obj, claim["label"])
    solo = {}
    if audit_unit:
        aid = audit_unit[1]
        own = f"Your own initial statement: {r1[aid]['verdict']} ({r1[aid]['rationale']})"
        obj, _ = client.ask(_prompt(ROLES[aid], claim, actual[aid], own, "Self-refinement context"), repeat_idx)
        solo[aid] = _agent_output(obj, claim["label"])
    votes = [r2[a]["verdict"] for a in ("A1", "A2", "A3")]; team = max(set(votes), key=votes.count)
    return Run(
        claim_id=str(claim["id"]), arm=arm, repeat_idx=repeat_idx,
        audit_unit=audit_unit,
        candidates={a: [m["memory_id"] for m in ms] for a, ms in actual.items()},
        placebo_map=placebo_map, placebo_diagnostics=placebo_diagnostics, round1=r1, round2=r2, solo=solo,
        team_verdict=team, team_correct=team == claim["label"],
        per_agent_correct_r1={a: r1[a]["correct"] for a in r1},
        per_agent_correct_r2={a: r2[a]["correct"] for a in r2},
        per_agent_correct_solo={a: solo[a]["correct"] for a in solo},
        cache_hits=client.cache_hits - hits0, llm_calls=client.calls - calls0,
    )
