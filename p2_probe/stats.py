from __future__ import annotations

import math
import random
from collections import defaultdict


def paired_effect(treated: list[dict], control: list[dict], outcome: str, repeats: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    xs = [float(x) - float(y) for x, y in zip(treated, control)]
    point = sum(xs) / len(xs) if xs else 0.0
    rng = random.Random(seed)
    boots = [sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(repeats)] if xs else [0.0]
    boots.sort()
    return point, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def phi(xs: list[bool], ys: list[bool]) -> float:
    a = sum(x and y for x, y in zip(xs, ys)); b = sum(x and not y for x, y in zip(xs, ys))
    c = sum((not x) and y for x, y in zip(xs, ys)); d = sum((not x) and (not y) for x, y in zip(xs, ys))
    den = math.sqrt(max(1, (a+b)*(c+d)*(a+c)*(b+d)))
    return (a*d-b*c) / den


def mismatch_label(local: float, team: float, margin: float = 0.0) -> str:
    if local > margin and team < -margin: return "local_positive_team_negative"
    if local < -margin and team > margin: return "local_negative_team_positive"
    if abs(local) <= margin and abs(team) <= margin: return "both_neutral"
    return "aligned_or_unclassified"


def paired_sign_pvalue(treated: list[bool], control: list[bool]) -> float:
    """Two-sided paired sign-test p-value, adequate for binary outcomes."""
    diffs = [int(x) - int(y) for x, y in zip(treated, control) if int(x) != int(y)]
    n = len(diffs)
    if n == 0:
        return 1.0
    k = min(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bh_reject(pvalues: list[float], q: float = 0.1) -> list[bool]:
    """Benjamini-Hochberg rejection mask, preserving input order."""
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    threshold_idx = -1
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= q * rank / max(1, len(pvalues)):
            threshold_idx = rank
    cutoff = indexed[threshold_idx - 1][1] if threshold_idx >= 0 else -1.0
    return [p <= cutoff for p in pvalues]


def diversity(runs: list[dict], condition: str) -> dict:
    rows = [r for r in runs if r["arm"] == condition]
    if not rows: return {"condition": condition, "individual_accuracy": 0, "team_accuracy": 0, "error_correlation": 0, "disagreement_rate": 0, "oracle_accuracy": 0}
    acc = [sum(r["per_agent_correct_r1"].values()) / 3 for r in rows]
    errors = {a: [not r["per_agent_correct_r1"][a] for r in rows] for a in ("A1", "A2", "A3")}
    corr = sum(phi(errors[a], errors[b]) for i, a in enumerate(errors) for b in list(errors)[i+1:]) / 3
    # E2 is a diversity diagnostic: aggregate the independent round-1 votes,
    # before round-2 communication can create a second source of dependence.
    team_r1 = [sum(r["per_agent_correct_r1"].values()) >= 2 for r in rows]
    return {"condition": condition, "individual_accuracy": sum(acc)/len(acc), "team_accuracy": sum(team_r1)/len(team_r1), "error_correlation": corr, "disagreement_rate": sum(len({r["round1"][a]["verdict"] for a in errors}) > 1 for r in rows)/len(rows), "oracle_accuracy": sum(any(r["per_agent_correct_r1"].values()) for r in rows)/len(rows)}
