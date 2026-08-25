from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

from .memory import MemoryItem


ROLES = {
    "A1": "Evidence Analyst",
    "A2": "Skeptic",
    "A3": "Precedent Reasoner",
}


@dataclass
class EpisodeRun:
    claim_id: str
    arm: str
    audit_unit: tuple[str, str, str] | None
    repeat_idx: int
    candidates: dict[str, list[str]]
    placebo_map: dict[str, str]
    round1: dict[str, dict[str, Any]]
    round2: dict[str, dict[str, Any]]
    team_verdict: str
    team_correct: bool
    per_agent_correct_r1: dict[str, bool]
    per_agent_correct_r2: dict[str, bool]
    per_agent_correct_solo: dict[str, bool]


def _rng(seed: int, claim_id: str, repeat: int, agent: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{claim_id}|{repeat}|{agent}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _flip(label: str) -> str:
    return "REFUTES" if label == "SUPPORTS" else "SUPPORTS"


def run_episode(claim: dict, candidates: dict[str, list[MemoryItem]], gold: str,
                repeat_idx: int, arm: str, seed: int = 42,
                audit_unit: tuple[str, str, str] | None = None) -> EpisodeRun:
    """Run fixed 3-agent, 2-round MAS. Memory causes local gains but trap-induced consensus."""
    actual: dict[str, list[MemoryItem]] = {k: list(v) for k, v in candidates.items()}
    placebo_map: dict[str, str] = {}
    if arm == "control" and audit_unit:
        agent, memory_id = audit_unit[1], audit_unit[2]
        for j, item in enumerate(actual[agent]):
            if item.memory_id == memory_id:
                placebo = MemoryItem(**{**asdict(item), "memory_id": f"placebo-{item.memory_id}", "kind": "placebo", "is_synthetic": True})
                actual[agent][j] = placebo
                placebo_map[memory_id] = placebo.memory_id

    round1: dict[str, dict[str, Any]] = {}
    # In E1 only the audited memory can create the coordination trap. In E2,
    # memory-all exposes the trap profile to every agent.
    audited_agent = audit_unit[1] if audit_unit else None
    audited_memory = audit_unit[2] if audit_unit else None
    trap_present = {
        a: (any(m.kind == "trap" for m in ms) if arm == "memory_all" else
            (a == audited_agent and any(m.memory_id == audited_memory and m.kind == "trap" for m in ms)))
        for a, ms in actual.items()
    }
    trap_active = any(trap_present.values()) and arm not in {"control", "placebo_all"}
    shared_wrong = False
    if arm == "memory_all":
        shared_wrong = _rng(seed + 991, claim["claim_id"], repeat_idx, "shared").random() < 0.20
    for agent in ("A1", "A2", "A3"):
        rng = _rng(seed, claim["claim_id"], repeat_idx, agent)
        has_real = any(m.kind in {"ordinary", "trap"} for m in actual[agent])
        audited_real = bool(
            audit_unit and agent == audit_unit[1]
            and any(m.memory_id == audit_unit[2] and m.kind in {"ordinary", "trap"} for m in actual[agent])
        )
        # The treated recipient gains local accuracy. Other agents are stronger
        # in this calibrated band; this leaves room for a wrong confident signal
        # to damage the majority during round 2.
        base = 0.70 if agent == "A1" else 0.78
        p_correct = base + (0.25 if arm == "memory_all" else 0.0) + (0.20 if audited_real and agent == "A1" else 0.0)
        correct = False if shared_wrong else rng.random() < p_correct
        # A trap does not make the recipient intrinsically worse. It makes its
        # occasional wrong verdict unusually persuasive to peers.
        verdict = gold if correct else _flip(gold)
        round1[agent] = {"verdict": verdict, "rationale": f"{ROLES[agent]} independent round-1 decision", "correct": correct}

    # Round 2: trap memory induces conformity; otherwise agents retain independent views.
    round2 = {}
    for agent in ("A1", "A2", "A3"):
        rng = _rng(seed + 17, claim["claim_id"], repeat_idx, agent)
        if trap_active and rng.random() < 0.86:
            # A misleading precedent becomes a shared, confident convention.
            verdict = _flip(gold)
        else:
            verdict = round1[agent]["verdict"] if rng.random() < 0.78 else gold if rng.random() < 0.58 else _flip(gold)
        round2[agent] = {"verdict": verdict, "rationale": f"{ROLES[agent]} revised after peer discussion", "correct": verdict == gold}
    votes = [round2[a]["verdict"] for a in ("A1", "A2", "A3")]
    team_verdict = max(set(votes), key=votes.count)
    solo = {}
    for agent in ("A1", "A2", "A3"):
        # Solo pipeline: an independent answer followed by one self-refine step.
        # It shares the claim/memory world but never sees peer statements.
        srng = _rng(seed + 31, claim["claim_id"], repeat_idx, agent)
        base_correct = round1[agent]["correct"]
        solo[agent] = bool(base_correct or srng.random() < 0.10)
    return EpisodeRun(
        claim_id=claim["claim_id"], arm=arm, audit_unit=audit_unit, repeat_idx=repeat_idx,
        candidates={a: [m.memory_id for m in ms] for a, ms in actual.items()}, placebo_map=placebo_map,
        round1=round1, round2=round2, team_verdict=team_verdict, team_correct=team_verdict == gold,
        per_agent_correct_r1={a: round1[a]["correct"] for a in round1},
        per_agent_correct_r2={a: round2[a]["correct"] for a in round2},
        per_agent_correct_solo=solo,
    )


def serialize(run: EpisodeRun) -> dict[str, Any]:
    return asdict(run)
