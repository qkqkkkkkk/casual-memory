"""Noise-aware analysis for recipient-specific causal memory utility.

The collection runner creates one receiver-level USE-vs-DROP audit for each
agent while holding the claim, retrieved memory, candidate contexts, and
sampling seeds fixed.  This module groups those audits back into one
``(claim, memory)`` experimental unit and measures whether the resulting team
utilities differ across recipients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


ANALYSIS_SCHEMA = "recipient-heterogeneity-fever-v1"
DEFAULT_RECIPIENTS = ("A1", "A2", "A3")


class AnalysisError(ValueError):
    """Raised when collected audit units do not satisfy the RQ3 design."""


def _sign(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0


def _sign_label(value: int) -> str:
    return {1: "positive", 0: "neutral", -1: "negative"}[value]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AnalysisError("a percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    numeric = tuple(float(value) for value in values)
    if not numeric:
        raise AnalysisError("a bootstrap estimate requires at least one event")
    point = sum(numeric) / len(numeric)
    rng = random.Random(seed)
    draws = [
        sum(numeric[rng.randrange(len(numeric))] for _ in numeric) / len(numeric)
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": point,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "event_count": len(numeric),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("mean requires at least one value")
    return sum(float(value) for value in values) / len(values)


def _load_payload(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = results_dir / "run_manifest.json"
    units_path = results_dir / "audit_units.json"
    if not manifest_path.is_file() or not units_path.is_file():
        raise AnalysisError(
            f"{results_dir} must contain run_manifest.json and audit_units.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(units_path.read_text(encoding="utf-8"))
    units = payload.get("units")
    if not isinstance(units, list):
        raise AnalysisError(f"{units_path} has no units list")
    if payload.get("design_hash") != manifest.get("design_hash"):
        raise AnalysisError(f"manifest/audit design_hash mismatch in {results_dir}")
    return manifest, units


def _group_units(
    units: Sequence[Mapping[str, Any]],
    *,
    recipients: Sequence[str],
    arm: str,
) -> list[dict[str, Any]]:
    expected = tuple(recipients)
    expected_set = set(expected)
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for unit in units:
        receiver = str(unit.get("receiver_agent_id", ""))
        if receiver not in expected_set:
            continue
        claim_id = str(unit.get("claim_id", ""))
        memory_id = str(unit.get("memory_id", ""))
        if not claim_id or not memory_id:
            raise AnalysisError("every audit unit needs claim_id and memory_id")
        if arm not in unit.get("arms", {}):
            raise AnalysisError(
                f"unit {(claim_id, memory_id, receiver)} has no {arm!r} arm"
            )
        key = (claim_id, memory_id)
        receiver_units = grouped.setdefault(key, {})
        if receiver in receiver_units:
            raise AnalysisError(f"duplicate receiver unit: {key + (receiver,)}")
        receiver_units[receiver] = unit

    result = []
    for (claim_id, memory_id), receiver_units in sorted(grouped.items()):
        actual = set(receiver_units)
        if actual != expected_set:
            raise AnalysisError(
                f"incomplete recipient coverage for {(claim_id, memory_id)}: "
                f"expected {sorted(expected_set)}, got {sorted(actual)}"
            )
        reference_use: tuple[float, ...] | None = None
        recipients_payload: dict[str, Any] = {}
        for receiver in expected:
            outcomes = receiver_units[receiver]["arms"][arm]
            use_samples = tuple(float(value) for value in outcomes["q_use_samples"])
            drop_samples = tuple(float(value) for value in outcomes["q_control_samples"])
            if not use_samples or len(use_samples) != len(drop_samples):
                raise AnalysisError(
                    f"unpaired samples for {(claim_id, memory_id, receiver)}"
                )
            if reference_use is None:
                reference_use = use_samples
            elif use_samples != reference_use:
                raise AnalysisError(
                    "USE baseline differs across recipients for "
                    f"{(claim_id, memory_id)}; the frozen-prefix design was violated"
                )
            utility_samples = tuple(
                use_value - drop_value
                for use_value, drop_value in zip(use_samples, drop_samples)
            )
            stored_samples = tuple(
                float(value) for value in outcomes.get("team_utility_samples", ())
            )
            if stored_samples and stored_samples != utility_samples:
                raise AnalysisError(
                    f"stored utility samples are inconsistent for "
                    f"{(claim_id, memory_id, receiver)}"
                )
            recipients_payload[receiver] = {
                "event_id": str(receiver_units[receiver]["event_id"]),
                "q_use": _mean(use_samples),
                "q_drop": _mean(drop_samples),
                "utility": _mean(utility_samples),
                "utility_samples": list(utility_samples),
            }
        result.append(
            {
                "claim_id": claim_id,
                "memory_id": memory_id,
                "repeat_count": len(reference_use or ()),
                "memory_observation_count": int(
                    receiver_units[expected[0]].get("memory_observation_count", 0)
                ),
                "recipients": recipients_payload,
            }
        )
    if not result:
        raise AnalysisError("no complete recipient groups were found")
    return result


def _decorate_events(
    grouped: Sequence[Mapping[str, Any]],
    *,
    recipients: Sequence[str],
    neutral_epsilon: float,
) -> list[dict[str, Any]]:
    decorated = []
    for group in grouped:
        recipient_payload = {
            receiver: dict(group["recipients"][receiver]) for receiver in recipients
        }
        signs = {
            receiver: _sign(
                float(recipient_payload[receiver]["utility"]), neutral_epsilon
            )
            for receiver in recipients
        }
        utilities = [
            float(recipient_payload[receiver]["utility"]) for receiver in recipients
        ]
        for receiver in recipients:
            recipient_payload[receiver]["sign"] = _sign_label(signs[receiver])
        decorated.append(
            {
                "claim_id": str(group["claim_id"]),
                "memory_id": str(group["memory_id"]),
                "repeat_count": int(group["repeat_count"]),
                "memory_observation_count": int(group["memory_observation_count"]),
                "recipients": recipient_payload,
                "recipient_sign_vector": {
                    receiver: _sign_label(signs[receiver]) for receiver in recipients
                },
                "direct_sign_flip": 1 in signs.values() and -1 in signs.values(),
                "any_sign_heterogeneity": len(set(signs.values())) > 1,
                "utility_range": max(utilities) - min(utilities),
            }
        )
    return decorated


def analyze_units(
    units: Sequence[Mapping[str, Any]],
    *,
    recipients: Sequence[str] = DEFAULT_RECIPIENTS,
    arm: str = "drop",
    neutral_epsilon: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze one collection run using event/claim-level bootstrap clusters."""

    if neutral_epsilon < 0:
        raise AnalysisError("neutral_epsilon must be non-negative")
    if bootstrap_samples < 100:
        raise AnalysisError("bootstrap_samples must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise AnalysisError("confidence_level must be in (0, 1)")
    recipients = tuple(recipients)
    if len(recipients) < 2 or len(set(recipients)) != len(recipients):
        raise AnalysisError("at least two unique recipients are required")

    grouped = _group_units(units, recipients=recipients, arm=arm)
    events = _decorate_events(
        grouped, recipients=recipients, neutral_epsilon=neutral_epsilon
    )
    metric_seed = seed

    def estimate(values: Sequence[float]) -> dict[str, Any]:
        nonlocal metric_seed
        metric_seed += 1
        return _bootstrap_mean(
            values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=metric_seed,
        )

    direct_flip = estimate([float(event["direct_sign_flip"]) for event in events])
    any_heterogeneity = estimate(
        [float(event["any_sign_heterogeneity"]) for event in events]
    )
    utility_range = estimate([float(event["utility_range"]) for event in events])

    per_recipient = {}
    for receiver in recipients:
        values = [float(event["recipients"][receiver]["utility"]) for event in events]
        signs = [_sign(value, neutral_epsilon) for value in values]
        per_recipient[receiver] = {
            "mean_utility": estimate(values),
            "positive_events": signs.count(1),
            "neutral_events": signs.count(0),
            "negative_events": signs.count(-1),
        }

    pairwise = {}
    for left, right in combinations(recipients, 2):
        differences = []
        absolute_differences = []
        sign_disagreements = []
        opposite_signs = []
        for event in events:
            left_value = float(event["recipients"][left]["utility"])
            right_value = float(event["recipients"][right]["utility"])
            left_sign = _sign(left_value, neutral_epsilon)
            right_sign = _sign(right_value, neutral_epsilon)
            difference = left_value - right_value
            differences.append(difference)
            absolute_differences.append(abs(difference))
            sign_disagreements.append(float(left_sign != right_sign))
            opposite_signs.append(float(left_sign * right_sign == -1))
        pairwise[f"{left}_vs_{right}"] = {
            "difference_definition": f"U({left})-U({right})",
            "mean_difference": estimate(differences),
            "mean_absolute_difference": estimate(absolute_differences),
            "sign_disagreement_rate": estimate(sign_disagreements),
            "direct_opposite_sign_rate": estimate(opposite_signs),
        }

    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "estimand": "U_team(memory, recipient)=Y(use_all)-Y(drop_recipient_only)",
        "arm": arm,
        "recipients": list(recipients),
        "neutral_epsilon": neutral_epsilon,
        "bootstrap_unit": "claim_memory_event",
        "event_count": len(events),
        "recipient_event_count": len(events) * len(recipients),
        "repeat_counts": dict(
            sorted(
                {
                    str(count): sum(
                        int(event["repeat_count"]) == count for event in events
                    )
                    for count in {int(event["repeat_count"]) for event in events}
                }.items()
            )
        ),
        "direct_recipient_sign_flip_rate": direct_flip,
        "any_recipient_sign_heterogeneity_rate": any_heterogeneity,
        "mean_within_event_utility_range": utility_range,
        "per_recipient": per_recipient,
        "pairwise": pairwise,
        "events": events,
    }


def _event_map(analysis: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(event["claim_id"]), str(event["memory_id"])): event
        for event in analysis["events"]
    }


def _positive_negative_orientations(
    event: Mapping[str, Any], recipients: Sequence[str]
) -> set[tuple[str, str]]:
    signs = event["recipient_sign_vector"]
    return {
        (left, right)
        for left in recipients
        for right in recipients
        if left != right and signs[left] == "positive" and signs[right] == "negative"
    }


def compare_independent_runs(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 142,
) -> dict[str, Any]:
    """Measure whether recipient heterogeneity replicates across seed ranges."""

    recipients = tuple(str(value) for value in current["recipients"])
    if recipients != tuple(str(value) for value in previous["recipients"]):
        raise AnalysisError("independent runs use different recipient sets")
    if float(current["neutral_epsilon"]) != float(previous["neutral_epsilon"]):
        raise AnalysisError("independent runs use different neutral thresholds")
    current_events = _event_map(current)
    previous_events = _event_map(previous)
    shared = sorted(set(current_events) & set(previous_events))
    if not shared:
        raise AnalysisError("independent runs have no shared claim-memory events")

    event_exact_agreement = []
    stable_flip = []
    reproducible_directional_flip = []
    per_event_sign_agreement = []
    per_event_direct_reversal = []
    per_event_neutral_transition = []
    previous_flip = []
    current_flip = []
    per_recipient_values: dict[str, list[tuple[int, int]]] = {
        receiver: [] for receiver in recipients
    }
    for key in shared:
        left = previous_events[key]
        right = current_events[key]
        previous_signs = left["recipient_sign_vector"]
        current_signs = right["recipient_sign_vector"]
        pairs = [
            (
                {"positive": 1, "neutral": 0, "negative": -1}[previous_signs[r]],
                {"positive": 1, "neutral": 0, "negative": -1}[current_signs[r]],
            )
            for r in recipients
        ]
        for receiver, pair in zip(recipients, pairs):
            per_recipient_values[receiver].append(pair)
        event_exact_agreement.append(float(all(a == b for a, b in pairs)))
        per_event_sign_agreement.append(
            sum(a == b for a, b in pairs) / len(recipients)
        )
        per_event_direct_reversal.append(
            sum(a * b == -1 for a, b in pairs) / len(recipients)
        )
        per_event_neutral_transition.append(
            sum((a == 0) != (b == 0) for a, b in pairs) / len(recipients)
        )
        left_flip = bool(left["direct_sign_flip"])
        right_flip = bool(right["direct_sign_flip"])
        previous_flip.append(float(left_flip))
        current_flip.append(float(right_flip))
        stable_flip.append(float(left_flip and right_flip))
        shared_orientations = _positive_negative_orientations(
            left, recipients
        ) & _positive_negative_orientations(right, recipients)
        reproducible_directional_flip.append(float(bool(shared_orientations)))

    metric_seed = seed

    def estimate(values: Sequence[float]) -> dict[str, Any]:
        nonlocal metric_seed
        metric_seed += 1
        return _bootstrap_mean(
            values,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=metric_seed,
        )

    both = int(sum(stable_flip))
    either = sum(
        int(bool(left) or bool(right))
        for left, right in zip(previous_flip, current_flip)
    )
    per_recipient = {}
    for receiver, values in per_recipient_values.items():
        per_recipient[receiver] = {
            "strict_sign_agreement": estimate(
                [float(left == right) for left, right in values]
            ),
            "direct_opposite_sign_rate": estimate(
                [float(left * right == -1) for left, right in values]
            ),
            "nonzero_neutral_transition_rate": estimate(
                [float((left == 0) != (right == 0)) for left, right in values]
            ),
        }

    return {
        "shared_events": len(shared),
        "current_event_coverage": len(shared) / len(current_events),
        "previous_event_coverage": len(shared) / len(previous_events),
        "previous_direct_flip_rate": estimate(previous_flip),
        "current_direct_flip_rate": estimate(current_flip),
        "stable_direct_flip_rate": estimate(stable_flip),
        "reproducible_directional_flip_rate": estimate(
            reproducible_directional_flip
        ),
        "direct_flip_event_jaccard": both / either if either else None,
        "exact_recipient_sign_vector_agreement": estimate(event_exact_agreement),
        "recipient_sign_consistency": estimate(per_event_sign_agreement),
        "recipient_direct_opposite_sign_transition_rate": estimate(
            per_event_direct_reversal
        ),
        "recipient_nonzero_neutral_transition_rate": estimate(
            per_event_neutral_transition
        ),
        "per_recipient": per_recipient,
    }


def _format_estimate(metric: Mapping[str, Any], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{float(metric['estimate']) * scale:.2f}{suffix} "
        f"[{float(metric['ci_low']) * scale:.2f}, "
        f"{float(metric['ci_high']) * scale:.2f}]"
    )


def _write_csv(path: Path, analysis: Mapping[str, Any], run_label: str) -> None:
    recipients = tuple(str(value) for value in analysis["recipients"])
    fieldnames = ["run", "claim_id", "memory_id", "repeat_count"]
    for receiver in recipients:
        fieldnames.extend(
            [
                f"{receiver}_q_use",
                f"{receiver}_q_drop",
                f"{receiver}_utility",
                f"{receiver}_sign",
            ]
        )
    fieldnames.extend(
        ["direct_sign_flip", "any_sign_heterogeneity", "utility_range"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in analysis["events"]:
            row: dict[str, Any] = {
                "run": run_label,
                "claim_id": event["claim_id"],
                "memory_id": event["memory_id"],
                "repeat_count": event["repeat_count"],
                "direct_sign_flip": event["direct_sign_flip"],
                "any_sign_heterogeneity": event["any_sign_heterogeneity"],
                "utility_range": event["utility_range"],
            }
            for receiver in recipients:
                values = event["recipients"][receiver]
                row.update(
                    {
                        f"{receiver}_q_use": values["q_use"],
                        f"{receiver}_q_drop": values["q_drop"],
                        f"{receiver}_utility": values["utility"],
                        f"{receiver}_sign": values["sign"],
                    }
                )
            writer.writerow(row)


def _render_report(payload: Mapping[str, Any]) -> str:
    current = payload["current_run"]
    previous = payload.get("previous_run")
    retest = payload.get("independent_retest")
    confidence_percent = round(
        100
        * float(
            current["direct_recipient_sign_flip_rate"]["confidence_level"]
        )
    )
    lines = [
        "# RQ3: Recipient heterogeneity",
        "",
        f"- Complete claim-memory events: {current['event_count']}",
        f"- Recipient-level audit units: {current['recipient_event_count']}",
        f"- Recipients: {', '.join(current['recipients'])}",
        f"- Estimand: `{current['estimand']}`",
        f"- Neutral epsilon: {current['neutral_epsilon']}",
        "- Bootstrap unit: claim-memory event",
        "",
        "## Current run",
        "",
        f"| Metric | Estimate [{confidence_percent}% CI] |",
        "|---|---:|",
        "| Direct +/− recipient sign-flip rate | "
        + _format_estimate(current["direct_recipient_sign_flip_rate"], percent=True)
        + " |",
        "| Any recipient sign heterogeneity rate | "
        + _format_estimate(
            current["any_recipient_sign_heterogeneity_rate"], percent=True
        )
        + " |",
        "| Mean within-event utility range | "
        + _format_estimate(current["mean_within_event_utility_range"])
        + " |",
        "",
        "## Per-recipient team utility",
        "",
        f"| Recipient | Mean utility [{confidence_percent}% CI] | Positive | Neutral | Negative |",
        "|---|---:|---:|---:|---:|",
    ]
    for receiver in current["recipients"]:
        row = current["per_recipient"][receiver]
        lines.append(
            f"| {receiver} | {_format_estimate(row['mean_utility'])} | "
            f"{row['positive_events']} | {row['neutral_events']} | "
            f"{row['negative_events']} |"
        )
    if retest is None:
        lines.extend(
            [
                "",
                "## Independent retest",
                "",
                "Not available. Run the same design with a disjoint inference-seed "
                "range and pass `--retest-results` before interpreting sign flips as "
                "replicated recipient heterogeneity.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Independent retest",
                "",
                f"- Shared events: {retest['shared_events']}",
                "- Previous direct sign-flip rate: "
                + _format_estimate(
                    retest["previous_direct_flip_rate"], percent=True
                ),
                "- Current direct sign-flip rate: "
                + _format_estimate(retest["current_direct_flip_rate"], percent=True),
                "- Stable direct sign-flip rate (flip in both runs): "
                + _format_estimate(retest["stable_direct_flip_rate"], percent=True),
                "- Reproducible directional flip rate (same positive/negative "
                "recipient orientation): "
                + _format_estimate(
                    retest["reproducible_directional_flip_rate"], percent=True
                ),
                "- Recipient sign consistency: "
                + _format_estimate(retest["recipient_sign_consistency"], percent=True),
                "- Direct opposite-sign transition rate across runs: "
                + _format_estimate(
                    retest["recipient_direct_opposite_sign_transition_rate"],
                    percent=True,
                ),
            ]
        )
        if previous is not None:
            lines.extend(
                [
                    "",
                    "A direct sign flip observed only in one run is treated as "
                    "sampling-sensitive. The strongest RQ3 evidence is the "
                    "reproducible directional flip rate, not the single-run rate.",
                ]
            )
    return "\n".join(lines) + "\n"


def write_analysis(
    results_dir: Path,
    *,
    retest_results: Path | None = None,
    neutral_epsilon: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Path:
    """Analyze a run, optionally compare an independent run, and write artifacts."""

    results_dir = Path(results_dir)
    manifest, units = _load_payload(results_dir)
    current = analyze_units(
        units,
        neutral_epsilon=neutral_epsilon,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "design_hash": manifest.get("design_hash"),
        "current_results": str(results_dir),
        "current_run": current,
        "previous_results": None,
        "previous_run": None,
        "independent_retest": None,
    }
    if retest_results is not None:
        retest_results = Path(retest_results)
        previous_manifest, previous_units = _load_payload(retest_results)
        if previous_manifest.get("design_hash") != manifest.get("design_hash"):
            raise AnalysisError(
                "independent retest design_hash does not match the current run"
            )
        previous = analyze_units(
            previous_units,
            neutral_epsilon=neutral_epsilon,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 10_000,
        )
        payload["previous_results"] = str(retest_results)
        payload["previous_run"] = previous
        payload["independent_retest"] = compare_independent_runs(
            current,
            previous,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 20_000,
        )

    output = results_dir / "rq3_analysis.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(results_dir / "recipient_matrix.csv", current, results_dir.name)
    (results_dir / "rq3_report.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--neutral-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    path = write_analysis(
        args.results,
        retest_results=args.retest_results,
        neutral_epsilon=args.neutral_epsilon,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    current = payload["current_run"]
    retest = payload.get("independent_retest")
    print(
        json.dumps(
            {
                "analysis": str(path),
                "events": current["event_count"],
                "direct_sign_flip_rate": current[
                    "direct_recipient_sign_flip_rate"
                ]["estimate"],
                "stable_direct_sign_flip_rate": (
                    retest["stable_direct_flip_rate"]["estimate"]
                    if retest is not None
                    else None
                ),
                "reproducible_directional_flip_rate": (
                    retest["reproducible_directional_flip_rate"]["estimate"]
                    if retest is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return path


if __name__ == "__main__":
    main()
