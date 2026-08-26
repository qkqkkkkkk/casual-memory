from __future__ import annotations

import math
import random
from collections import defaultdict


def effect(differences: list[int], n_bootstrap: int, seed: int) -> tuple[float, list[float]]:
    point = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(n_bootstrap))
    return point, [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def paired_sign_pvalue(differences: list[int]) -> float:
    nonzero = [x for x in differences if x]
    if not nonzero:
        return 1.0
    k = min(sum(x > 0 for x in nonzero), sum(x < 0 for x in nonzero))
    tail = sum(math.comb(len(nonzero), j) for j in range(k + 1)) / (2 ** len(nonzero))
    return min(1.0, 2 * tail)


def bh_reject(pvalues: list[float], q: float = 0.1) -> list[bool]:
    ordered = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    accepted_rank = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= q * rank / max(1, len(pvalues)):
            accepted_rank = rank
    cutoff = ordered[accepted_rank - 1][1] if accepted_rank else -1.0
    return [value <= cutoff for value in pvalues]


def cluster_rate_ci(rows: list[dict], n_bootstrap: int, seed: int) -> list[float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["claim_id"]].append(row)
    keys = list(grouped)
    rng = random.Random(seed)
    rates = []
    for _ in range(n_bootstrap):
        sampled = [row for _ in keys for row in grouped[rng.choice(keys)]]
        rates.append(sum(bool(row["confirmed"]) for row in sampled) / max(1, len(sampled)))
    rates.sort()
    return [rates[int(0.025 * len(rates))], rates[int(0.975 * len(rates))]]


def phi(xs: list[bool], ys: list[bool]) -> float:
    a = sum(x and y for x, y in zip(xs, ys)); b = sum(x and not y for x, y in zip(xs, ys))
    c = sum((not x) and y for x, y in zip(xs, ys)); d = sum((not x) and (not y) for x, y in zip(xs, ys))
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denominator if denominator else 0.0
