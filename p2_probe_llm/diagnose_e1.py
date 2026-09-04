#!/usr/bin/env python3
"""Diagnose a completed E1 run without making any LLM calls."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def similarity(a: str, b: str) -> float:
    words_a = set(re.findall(r"[a-z0-9]+", str(a).lower()))
    words_b = set(re.findall(r"[a-z0-9]+", str(b).lower()))
    return len(words_a & words_b) / max(1, len(words_a | words_b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect E1 memory injection and answer changes")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--experience-bank", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None, help="Selected difficulty JSONL, for claim/memory relevance diagnostics")
    parser.add_argument("--limit", type=int, default=20, help="Number of example pairs to print")
    args = parser.parse_args()

    rows = read_jsonl(args.results_dir / "episode_runs.jsonl")
    bank = {str(row.get("memory_id")): row for row in read_jsonl(args.experience_bank)}
    claims = {}
    if args.test:
        claims = {str(row.get("id")): row for row in read_jsonl(args.test)}
    pairs: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        pairs[(str(row.get("claim_id")), int(row.get("repeat_idx", 0)))][row["branch"]] = row
    complete = [pair for pair in pairs.values() if "control" in pair and "treated" in pair]

    stats = Counter()
    examples = []
    memory_rows = []
    for pair in complete:
        control, treated = pair["control"], pair["treated"]
        c_id = control.get("candidates", {}).get("A1", [None])[0]
        t_id = treated.get("candidates", {}).get("A1", [None])[0]
        c_r1 = control.get("round1", {}).get("A1", {})
        t_r1 = treated.get("round1", {}).get("A1", {})
        c_solo = control.get("solo", {}).get("A1", {})
        t_solo = treated.get("solo", {}).get("A1", {})
        stats["a1_memory_id_changed"] += c_id != t_id
        stats["a1_round1_verdict_changed"] += c_r1.get("verdict") != t_r1.get("verdict")
        stats["a1_solo_verdict_changed"] += c_solo.get("verdict") != t_solo.get("verdict")
        stats["a1_round1_rationale_changed"] += c_r1.get("rationale") != t_r1.get("rationale")
        stats["a1_round1_confidence_changed"] += c_r1.get("confidence") != t_r1.get("confidence")
        stats["control_a1_memory_assessed"] += bool(c_r1.get("memory_assessment"))
        stats["treated_a1_memory_assessed"] += bool(t_r1.get("memory_assessment"))
        stats[f"control_a1_influence_{str(c_r1.get('memory_influence', 'missing')).lower()}"] += 1
        stats[f"treated_a1_influence_{str(t_r1.get('memory_influence', 'missing')).lower()}"] += 1
        stats[f"control_a1_relevance_{str(c_r1.get('memory_relevance', 'missing')).lower()}"] += 1
        stats[f"treated_a1_relevance_{str(t_r1.get('memory_relevance', 'missing')).lower()}"] += 1
        stats["a1_round2_verdict_changed"] += control.get("round2", {}).get("A1", {}).get("verdict") != treated.get("round2", {}).get("A1", {}).get("verdict")
        stats["team_verdict_changed"] += control.get("team_verdict") != treated.get("team_verdict")
        stats["a2_a3_round1_changed"] += any(control.get("round1", {}).get(a) != treated.get("round1", {}).get(a) for a in ("A2", "A3"))
        if len(examples) < args.limit and c_id != t_id:
            examples.append({
                "claim_id": control.get("claim_id"), "repeat_idx": control.get("repeat_idx"),
                "control_memory": c_id, "treated_memory": t_id,
                "control_r1": c_r1.get("verdict"), "treated_r1": t_r1.get("verdict"),
                "control_solo": c_solo.get("verdict"), "treated_solo": t_solo.get("verdict"),
                "control_rationale": c_r1.get("rationale", ""), "treated_rationale": t_r1.get("rationale", ""),
                "control_memory_assessment": c_r1.get("memory_assessment", ""), "treated_memory_assessment": t_r1.get("memory_assessment", ""),
            })

    # Deduplicate audited treated memories and show their actual text/label.
    for pair in complete:
        treated = pair["treated"]
        memory_id = treated.get("audit_unit", [None, None, None])[-1]
        if memory_id in {row.get("memory_id") for row in memory_rows}:
            continue
        item = bank.get(str(memory_id), {})
        claim = claims.get(str(treated.get("claim_id")), {})
        memory_rows.append({
            "memory_id": memory_id,
            "claim_id": treated.get("claim_id"),
            "claim_text": claim.get("claim"),
            "memory_claim": item.get("claim"), "gold_label": item.get("gold_label"),
            "evidence": " ".join(str(x.get("text", "")) for x in item.get("evidence_bundle", []))[:500],
            "rationale_digest": item.get("rationale_digest"),
            "memory_label_matches_claim": bool(claim) and item.get("gold_label") == claim.get("label"),
            "claim_memory_similarity": similarity(claim.get("claim", ""), item.get("claim", "")) if claim else None,
        })

    report = {"paired_runs": len(complete), "counts": dict(stats), "examples": examples, "audited_memories": memory_rows[:args.limit]}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
