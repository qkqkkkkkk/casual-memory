#!/usr/bin/env python3
"""Run the full event-level causal-memory pipeline on the real FEVER/P2 MAS.

The first invocation collects a frozen-prefix USE/DROP audit.  A second
invocation with a different ``--sample-seed-base`` and ``--retest-results``
provides the independent sign-consistency gate.  Predictor fitting is blocked
until both the oracle-headroom and independent-retest gates pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Mapping, Sequence

from p2_probe_llm.client import CachedChat
from p2_probe_llm.retrieval import GMemorySemanticIndex
from p2_probe_llm.run_experiment_e1 import (
    _claim_key,
    _evidence_texts,
    _memory_source_keys,
    load,
    memory_is_eligible,
    placebo,
    retrieve,
    schema_coverage,
    select_claims,
)

from .audit import CounterfactualAudit
from .diagnostics import (
    PivotalityObservation,
    UtilityRetest,
    evaluate_pretraining_gate,
    observation_count_distribution,
    oracle_noise_floor,
    stratify_pivotality,
)
from .estimator import AmortizedUtilityEstimator, PotentialOutcomeExample
from .features import EventFeatureBuilder
from .fever_backend import (
    AGENTS,
    FeverBranchRunner,
    build_fever_checkpoint,
    build_fever_event,
    load_existing_rows,
    placebo_candidate,
)
from .interventions import InterventionBuilder
from .oracle import OracleControllability, OracleExample
from .stats import summarize
from .controller import RelianceController
from .types import (
    CounterfactualAuditResult,
    InterventionArm,
    MemoryUseEvent,
    RelianceAction,
)


RUNNER_SCHEMA = "causal-memory-control-fever-v1"
DEFAULT_PRIMARY_ARM = InterventionArm.DROP


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--experience-bank", type=Path, required=True)
    parser.add_argument("--distractor-bank", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--retrieval-threshold", type=float, default=0.3)
    parser.add_argument("--claims", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--receivers",
        default="A1",
        help="Comma-separated subset of A1,A2,A3",
    )
    parser.add_argument(
        "--arms",
        default="drop,placebo",
        help="Comma-separated controls: drop,placebo,global_drop; drop is required",
    )
    parser.add_argument(
        "--primary-arm",
        choices=(InterventionArm.DROP.value, InterventionArm.GLOBAL_DROP.value),
        default=DEFAULT_PRIMARY_ARM.value,
        help=(
            "Analysis/training estimand. This does not change collection or the run "
            "hash, so completed outputs can be reanalysed with --resume."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--sample-seed-base",
        type=int,
        default=1000,
        help="First inference seed; change this for the independent retest",
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=1.96)
    parser.add_argument("--minimum-sign-consistency", type=float, default=0.8)
    parser.add_argument("--minimum-oracle-gain", type=float, default=0.0)
    parser.add_argument("--oracle-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--oracle-confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-retest-coverage", type=float, default=0.95)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument(
        "--cache-seed-from",
        type=Path,
        default=None,
        help="Copy an existing p2_probe_llm SQLite cache into a new output directory",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-train-samples", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _parse_receivers(value: str) -> tuple[str, ...]:
    receivers = tuple(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))
    if not receivers or any(receiver not in AGENTS for receiver in receivers):
        raise SystemExit("--receivers must be a comma-separated subset of A1,A2,A3")
    return receivers


def _parse_arms(value: str) -> tuple[InterventionArm, ...]:
    try:
        arms = tuple(
            dict.fromkeys(
                InterventionArm(part.strip().lower())
                for part in value.split(",")
                if part.strip()
            )
        )
    except ValueError as exc:
        raise SystemExit(
            "--arms supports only drop,placebo,global_drop"
        ) from exc
    if DEFAULT_PRIMARY_ARM not in arms:
        raise SystemExit("--arms must include drop because USE-vs-DROP is the primary estimand")
    if InterventionArm.USE in arms:
        raise SystemExit("do not include use in --arms; USE is added automatically")
    return arms


def _validate_args(args: argparse.Namespace) -> None:
    if args.claims < 1:
        raise SystemExit("--claims must be at least 1")
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 for sign diagnostics")
    if args.delta < 0 or args.kappa < 0:
        raise SystemExit("--delta and --kappa must be non-negative")
    if args.minimum_oracle_gain < 0:
        raise SystemExit("--minimum-oracle-gain must be non-negative")
    if args.oracle_bootstrap_samples < 100:
        raise SystemExit("--oracle-bootstrap-samples must be at least 100")
    if not 0.0 < args.oracle_confidence_level < 1.0:
        raise SystemExit("--oracle-confidence-level must be in (0, 1)")
    for name in (
        "minimum_sign_consistency",
        "minimum_retest_coverage",
        "train_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.min_train_samples < 2:
        raise SystemExit("--min-train-samples must be at least 2")
    if args.ensemble_size < 2:
        raise SystemExit("--ensemble-size must be at least 2")


def _md5(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"input file does not exist: {path}")
    return hashlib.md5(path.read_bytes()).hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _event_id(claim_id: str, receiver: str, memory_id: str) -> str:
    return "fever-" + hashlib.sha256(
        f"{claim_id}\x1f{receiver}\x1f{memory_id}".encode("utf-8")
    ).hexdigest()[:24]


def _validate_inputs(
    claims: Sequence[dict[str, Any]],
    experience_bank: Sequence[dict[str, Any]],
    distractor_bank: Sequence[dict[str, Any]],
) -> None:
    if not claims:
        raise SystemExit("no eligible evidence-bearing binary claims were selected")
    if not experience_bank or not distractor_bank:
        raise SystemExit("experience and distractor banks must not be empty")
    if schema_coverage(list(experience_bank)) != 1.0 or schema_coverage(list(distractor_bank)) != 1.0:
        raise SystemExit("banks use an old schema; rebuild them with p2_probe_llm.build_pools")
    experience_sources = {
        source for item in experience_bank for source in _memory_source_keys(item)
    }
    distractor_sources = {
        source for item in distractor_bank for source in _memory_source_keys(item)
    }
    if experience_sources & distractor_sources:
        raise SystemExit("experience and distractor provenance overlaps")
    test_keys = {_claim_key(claim) for claim in claims}
    bank_keys = {_claim_key(item) for item in (*experience_bank, *distractor_bank)}
    if test_keys & bank_keys:
        raise SystemExit("a selected test claim is present in a memory bank")


def _prepare_output(
    args: argparse.Namespace,
    design: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> tuple[Path, dict[tuple[str, int, str], dict[str, Any]]]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    log_path = output / "audit_runs.jsonl"
    manifest = {
        "runner_schema": RUNNER_SCHEMA,
        "design_hash": _stable_hash(design),
        "run_hash": _stable_hash(run_config),
        "design": dict(design),
        "run": dict(run_config),
    }
    if args.resume:
        if not manifest_path.is_file() or not log_path.is_file():
            raise SystemExit("--resume requires run_manifest.json and audit_runs.jsonl")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_hash") != manifest["run_hash"]:
            raise SystemExit("resume arguments do not match the existing run manifest")
        return log_path, load_existing_rows(log_path)
    if manifest_path.exists() or log_path.exists():
        raise SystemExit(
            f"refusing to overwrite {output}; use a new --output-dir or --resume"
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path.touch(exist_ok=False)
    return log_path, {}


def _estimate_dict(estimate: Any) -> dict[str, Any]:
    return asdict(estimate)


def summarize_audit(result: CounterfactualAuditResult) -> dict[str, Any]:
    arms = {}
    for arm_result in result.arms:
        arms[arm_result.arm.value] = {
            "q_use": sum(pair.use.team_reward for pair in arm_result.pairs)
            / len(arm_result.pairs),
            "q_control": sum(pair.control.team_reward for pair in arm_result.pairs)
            / len(arm_result.pairs),
            "q_use_samples": [pair.use.team_reward for pair in arm_result.pairs],
            "q_control_samples": [pair.control.team_reward for pair in arm_result.pairs],
            "team_utility": _estimate_dict(arm_result.team_utility),
            "team_utility_samples": [pair.team_utility for pair in arm_result.pairs],
            "behavior_effect": _estimate_dict(arm_result.behavior_effect),
            "behavior_effect_samples": [pair.behavior_distance for pair in arm_result.pairs],
            "local_utility": (
                _estimate_dict(arm_result.local_utility)
                if arm_result.local_utility is not None
                else None
            ),
            "local_utility_samples": [pair.local_utility for pair in arm_result.pairs],
            "sample_seeds": [pair.seed for pair in arm_result.pairs],
        }
    causal_effects: dict[str, Any] = {}
    receiver = arms.get(InterventionArm.DROP.value)
    exposure_set = arms.get(InterventionArm.GLOBAL_DROP.value)
    if receiver is not None:
        causal_effects["receiver_marginal"] = {
            "definition": "Y(use_all)-Y(drop_receiver_only)",
            **receiver["team_utility"],
            "samples": receiver["team_utility_samples"],
        }
    if exposure_set is not None:
        causal_effects["exposure_set_total"] = {
            "definition": "Y(use_all)-Y(drop_from_all_observers)",
            **exposure_set["team_utility"],
            "samples": exposure_set["team_utility_samples"],
        }
    if receiver is not None and exposure_set is not None:
        receiver_samples = receiver["team_utility_samples"]
        global_samples = exposure_set["team_utility_samples"]
        if len(receiver_samples) != len(global_samples):
            raise ValueError("DROP and GLOBAL_DROP sample counts must match")
        spillover_samples = [
            float(global_value) - float(receiver_value)
            for receiver_value, global_value in zip(
                receiver_samples, global_samples
            )
        ]
        causal_effects["spillover_redundancy"] = {
            "definition": (
                "exposure_set_total-receiver_marginal="
                "Y(drop_receiver_only)-Y(drop_from_all_observers)"
            ),
            **_estimate_dict(summarize(spillover_samples)),
            "samples": spillover_samples,
        }
    event = result.event
    return {
        "event_id": event.event_id,
        "claim_id": str(event.task_metadata["claim_id"]),
        "receiver_agent_id": event.receiver_agent_id,
        "memory_id": event.memory.memory_id,
        "memory_observation_count": event.observation_count,
        "retrieval_score": event.retrieval.similarity if event.retrieval else None,
        "retrieval_rank": event.retrieval.rank if event.retrieval else None,
        "reliability_prior": event.reliability_prior,
        "arms": arms,
        "causal_effects": causal_effects,
    }


def _sign(value: float, delta: float = 0.0) -> int:
    return 1 if value > delta else -1 if value < -delta else 0


def _pivotality_rows(
    audits: Sequence[CounterfactualAuditResult],
) -> list[PivotalityObservation]:
    rows = []
    for audit in audits:
        receiver = audit.event.receiver_agent_id
        others = tuple(agent for agent in AGENTS if agent != receiver)
        for pair in audit.primary.pairs:
            use = pair.use.metadata["raw_run"]
            control = pair.control.metadata["raw_run"]
            answers = tuple(
                str(use["round1"][agent]["verdict"]) == "SUPPORTS"
                for agent in others
            )
            local = float(pair.local_utility or 0.0)
            mismatch = _sign(local) * _sign(pair.team_utility) == -1
            rows.append(
                PivotalityObservation(
                    event_id=f"{audit.event.event_id}:{pair.repeat_index}",
                    other_agent_answers=answers,
                    mismatch=mismatch,
                )
            )
            # Frozen-prefix invariant for DROP: unaffected agents' first round
            # must be byte-identical because the cached prompt and seed match.
            if any(use["round1"][agent] != control["round1"][agent] for agent in others):
                raise RuntimeError(
                    f"frozen-prefix violation for {audit.event.event_id}: "
                    "an unaffected agent changed in round 1"
                )
    return rows


def _noise_report_from_audits(
    audits: Sequence[CounterfactualAuditResult],
    arm: InterventionArm,
    *,
    neutral_epsilon: float,
) -> Any:
    observations = [
        UtilityRetest(audit.event.event_id, pair.team_utility)
        for audit in audits
        for result in audit.arms
        if result.arm == arm
        for pair in result.pairs
    ]
    if not observations:
        return None
    return oracle_noise_floor(observations, neutral_epsilon=neutral_epsilon)


def _load_retest_units(
    current_design_hash: str,
    previous_dir: Path,
) -> list[dict[str, Any]]:
    manifest_path = previous_dir / "run_manifest.json"
    units_path = previous_dir / "audit_units.json"
    if not manifest_path.is_file() or not units_path.is_file():
        raise SystemExit(
            "--retest-results must contain run_manifest.json and audit_units.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("design_hash") != current_design_hash:
        raise SystemExit("independent retest design_hash does not match this run")
    payload = json.loads(units_path.read_text(encoding="utf-8"))
    return list(payload["units"])


def _sign_label(value: float, delta: float = 0.0) -> str:
    return {1: "positive", 0: "neutral", -1: "negative"}[_sign(value, delta)]


def _independent_arm_diagnostics(
    current_units: Sequence[Mapping[str, Any]],
    previous_units: Sequence[Mapping[str, Any]],
    arm: InterventionArm,
    *,
    neutral_epsilon: float,
) -> tuple[Any | None, float, dict[str, Any]]:
    previous = {
        str(unit["event_id"]): unit
        for unit in previous_units
        if arm.value in unit.get("arms", {})
    }
    current = {
        str(unit["event_id"]): unit
        for unit in current_units
        if arm.value in unit.get("arms", {})
    }
    shared = sorted(set(previous) & set(current))
    if not current or not shared:
        return None, 0.0, {
            "shared_events": 0,
            "transition_counts": {},
            "strict_sign_agreement": 0.0,
            "direct_opposite_sign_rate": 0.0,
            "nonzero_neutral_transition_rate": 0.0,
            "stable_nonzero_rate": 0.0,
        }
    observations = []
    transition_counts: dict[str, int] = {}
    strict_agreements = 0
    direct_reversals = 0
    neutral_transitions = 0
    stable_nonzero = 0
    for event_id in shared:
        values = []
        for source in (previous[event_id], current[event_id]):
            value = float(
                source["arms"][arm.value]["team_utility"]["mean"]
            )
            values.append(value)
            observations.append(
                UtilityRetest(event_id, value)
            )
        left_sign = _sign(values[0], neutral_epsilon)
        right_sign = _sign(values[1], neutral_epsilon)
        transition = (
            f"{_sign_label(values[0], neutral_epsilon)}_to_"
            f"{_sign_label(values[1], neutral_epsilon)}"
        )
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        strict_agreements += int(left_sign == right_sign)
        direct_reversals += int(left_sign * right_sign == -1)
        neutral_transitions += int((left_sign == 0) != (right_sign == 0))
        stable_nonzero += int(left_sign == right_sign and left_sign != 0)
    count = len(shared)
    transitions = {
        "shared_events": count,
        "transition_counts": dict(sorted(transition_counts.items())),
        "strict_sign_agreement": strict_agreements / count,
        "direct_opposite_sign_rate": direct_reversals / count,
        "nonzero_neutral_transition_rate": neutral_transitions / count,
        "stable_nonzero_rate": stable_nonzero / count,
    }
    return (
        oracle_noise_floor(observations, neutral_epsilon=neutral_epsilon),
        count / len(current),
        transitions,
    )


def _units_for_arm(
    units: Sequence[Mapping[str, Any]], arm: InterventionArm
) -> list[Mapping[str, Any]]:
    return [unit for unit in units if arm.value in unit.get("arms", {})]


def _oracle_evaluation(
    units: Sequence[Mapping[str, Any]],
    arm: InterventionArm,
    delta: float,
) -> Any:
    examples = []
    for unit in _units_for_arm(units, arm):
        outcomes = unit["arms"][arm.value]
        scores = {}
        if unit.get("retrieval_score") is not None:
            scores["similarity"] = float(unit["retrieval_score"])
        if unit.get("reliability_prior") is not None:
            scores["reliability"] = float(unit["reliability_prior"])
        examples.append(
            OracleExample(
                event_id=str(unit["event_id"]),
                q_use=float(outcomes["q_use"]),
                q_drop=float(outcomes["q_control"]),
                scores=scores,
            )
        )
    if not examples:
        return None
    return OracleControllability(delta=delta).evaluate(examples)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _oracle_gain_bootstrap(
    units: Sequence[Mapping[str, Any]],
    arm: InterventionArm,
    oracle: Any,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    arm_units = _units_for_arm(units, arm)
    if oracle is None or not arm_units:
        return {
            "status": "unavailable",
            "reason": "arm_has_no_audit_units",
        }
    oracle_policy = oracle.policy("oracle")
    strongest = max(
        (policy for policy in oracle.policies if policy.policy != "oracle"),
        key=lambda policy: (policy.mean_reward, policy.policy),
    )
    differences = []
    for unit, oracle_use, baseline_use in zip(
        arm_units, oracle_policy.decisions, strongest.decisions
    ):
        outcomes = unit["arms"][arm.value]
        q_use = float(outcomes["q_use"])
        q_control = float(outcomes["q_control"])
        oracle_reward = q_use if oracle_use else q_control
        baseline_reward = q_use if baseline_use else q_control
        differences.append(oracle_reward - baseline_reward)
    point_gain = sum(differences) / len(differences)
    arm_seed = sum((index + 1) * ord(char) for index, char in enumerate(arm.value))
    rng = random.Random(seed + arm_seed)
    bootstrapped = []
    for _ in range(samples):
        bootstrapped.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "ok",
        "method": "event_percentile_bootstrap_with_fixed_policy_decisions",
        "comparison_policy": strongest.policy,
        "point_gain": point_gain,
        "confidence_level": confidence_level,
        "ci_low": _percentile(bootstrapped, alpha),
        "ci_high": _percentile(bootstrapped, 1.0 - alpha),
        "bootstrap_samples": samples,
        "event_count": len(differences),
        "nonzero_event_differences": sum(value != 0.0 for value in differences),
    }


def _arm_gate(
    *,
    oracle: Any,
    within_noise: Any,
    independent_noise: Any | None,
    retest_coverage: float,
    transition_diagnostics: Mapping[str, Any] | None,
    bootstrap: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_noise = independent_noise or within_noise
    base_gate = evaluate_pretraining_gate(
        oracle,
        gate_noise,
        minimum_sign_consistency=args.minimum_sign_consistency,
        minimum_oracle_gain=args.minimum_oracle_gain,
    )
    ci_low = bootstrap.get("ci_low")
    oracle_ci_passed = (
        ci_low is not None and float(ci_low) > args.minimum_oracle_gain
    )
    oracle_passed = bool(base_gate.oracle_passed and oracle_ci_passed)
    reasons = list(base_gate.reasons)
    if base_gate.oracle_passed and not oracle_ci_passed:
        reasons.append("oracle_gain_ci_not_above_minimum")
    can_train = bool(oracle_passed and base_gate.noise_floor_passed)
    if independent_noise is None:
        can_train = False
        reasons.append("independent_retest_required")
    elif retest_coverage < args.minimum_retest_coverage:
        can_train = False
        reasons.append("independent_retest_coverage_below_threshold")
    return {
        **asdict(base_gate),
        "can_train": can_train,
        "oracle_passed": oracle_passed,
        "oracle_point_passed": base_gate.oracle_passed,
        "oracle_ci_passed": oracle_ci_passed,
        "oracle_gain_bootstrap": dict(bootstrap),
        "minimum_oracle_gain": args.minimum_oracle_gain,
        "reasons": reasons,
        "sign_consistency_source": (
            "independent_run_level_retest"
            if independent_noise is not None
            else "within_run_repeats_diagnostic_only"
        ),
        "independent_retest_coverage": retest_coverage,
        "minimum_retest_coverage": args.minimum_retest_coverage,
        "sign_transition_diagnostics": (
            dict(transition_diagnostics)
            if transition_diagnostics is not None
            else None
        ),
    }


def _normalize_confidence(value: Any) -> tuple[float | None, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, "missing_or_non_numeric"
    if not math.isfinite(numeric):
        return None, "non_finite"
    if 0.0 <= numeric <= 1.0:
        return numeric, "unit_interval"
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0, "percentage"
    return None, "out_of_range"


def _confidence_diagnostics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    scale_counts: dict[str, int] = {}
    normalized_values = []
    parse_failures = 0
    parsed_outputs = 0
    branch_scores: dict[str, list[float]] = {}
    branch_support: dict[str, list[float]] = {}
    for row in rows:
        for round_name in ("round1", "round2"):
            for output in row.get(round_name, {}).values():
                parsed_outputs += 1
                parse_failures += int(bool(output.get("parse_fail")))
                normalized, scale = _normalize_confidence(output.get("confidence"))
                scale_counts[scale] = scale_counts.get(scale, 0) + 1
                if normalized is not None:
                    normalized_values.append(normalized)
        correctness_probabilities = []
        support_probabilities = []
        for output in row.get("round2", {}).values():
            normalized, _ = _normalize_confidence(output.get("confidence"))
            if normalized is None:
                continue
            correctness_probabilities.append(
                normalized if bool(output.get("correct")) else 1.0 - normalized
            )
            support_probabilities.append(
                normalized
                if str(output.get("verdict")) == "SUPPORTS"
                else 1.0 - normalized
            )
        branch = str(row.get("branch", "unknown"))
        if correctness_probabilities:
            branch_scores.setdefault(branch, []).append(
                sum(correctness_probabilities) / len(correctness_probabilities)
            )
        if support_probabilities:
            branch_support.setdefault(branch, []).append(
                sum(support_probabilities) / len(support_probabilities)
            )
    return {
        "status": "auxiliary_uncalibrated_self_report_only",
        "normalization": "values in (1,100] divided by 100; [0,1] unchanged",
        "parsed_outputs": parsed_outputs,
        "parse_failures": parse_failures,
        "confidence_scale_counts": dict(sorted(scale_counts.items())),
        "normalized_confidence_count": len(normalized_values),
        "normalized_confidence_min": (
            min(normalized_values) if normalized_values else None
        ),
        "normalized_confidence_max": (
            max(normalized_values) if normalized_values else None
        ),
        "normalized_confidence_mean": (
            sum(normalized_values) / len(normalized_values)
            if normalized_values
            else None
        ),
        "branch_mean_team_correctness_score": {
            branch: sum(values) / len(values)
            for branch, values in sorted(branch_scores.items())
        },
        "branch_mean_support_probability": {
            branch: sum(values) / len(values)
            for branch, values in sorted(branch_support.items())
        },
    }


def _causal_effect_summary(
    units: Sequence[Mapping[str, Any]], *, delta: float
) -> dict[str, Any]:
    names = (
        "receiver_marginal",
        "exposure_set_total",
        "spillover_redundancy",
    )
    result = {}
    for name in names:
        values = [
            float(unit["causal_effects"][name]["mean"])
            for unit in units
            if name in unit.get("causal_effects", {})
        ]
        if not values:
            continue
        signs = [_sign(value, delta) for value in values]
        result[name] = {
            "event_count": len(values),
            "mean_of_event_means": sum(values) / len(values),
            "positive_events": signs.count(1),
            "neutral_events": signs.count(0),
            "negative_events": signs.count(-1),
        }
    return result


def _split_claims(
    units: Sequence[Mapping[str, Any]], train_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    claim_ids = sorted({str(unit["claim_id"]) for unit in units})
    random.Random(seed).shuffle(claim_ids)
    if len(claim_ids) < 2:
        return set(claim_ids), set()
    train_count = min(len(claim_ids) - 1, max(1, round(len(claim_ids) * train_fraction)))
    return set(claim_ids[:train_count]), set(claim_ids[train_count:])


def _rmse(expected: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(expected, predicted))
        / len(expected)
    )


def _fit_and_evaluate(
    *,
    units: Sequence[Mapping[str, Any]],
    events: Mapping[str, MemoryUseEvent],
    primary_arm: InterventionArm,
    args: argparse.Namespace,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if not gate["can_train"]:
        return {
            "status": "blocked_by_pretraining_gate",
            "primary_arm": primary_arm.value,
            "reasons": gate["reasons"],
        }
    train_claims, test_claims = _split_claims(
        units, args.train_fraction, args.selection_seed
    )
    train_units = [unit for unit in units if str(unit["claim_id"]) in train_claims]
    test_units = [unit for unit in units if str(unit["claim_id"]) in test_claims]
    if len(train_units) < args.min_train_samples or not test_units:
        return {
            "status": "insufficient_split",
            "primary_arm": primary_arm.value,
            "train_units": len(train_units),
            "test_units": len(test_units),
            "minimum_train_samples": args.min_train_samples,
        }
    builder = EventFeatureBuilder()
    examples = []
    for unit in train_units:
        event = events[str(unit["event_id"])]
        outcomes = unit["arms"][primary_arm.value]
        examples.append(
            PotentialOutcomeExample(
                features=builder.build(event),
                q_use=float(outcomes["q_use"]),
                q_drop=float(outcomes["q_control"]),
                event_id=event.event_id,
            )
        )
    estimator = AmortizedUtilityEstimator(
        ensemble_size=args.ensemble_size,
        min_samples=args.min_train_samples,
        utility_threshold=args.delta,
        seed=args.selection_seed + 97,
    )
    if not estimator.fit(examples):
        return {
            "status": "estimator_fit_failed",
            "primary_arm": primary_arm.value,
            "train_units": len(train_units),
        }
    controller = RelianceController(delta=args.delta, kappa=args.kappa)
    predictions = []
    true_use = []
    true_drop = []
    true_utility = []
    predicted_use = []
    predicted_drop = []
    predicted_utility = []
    controller_rewards = []
    oracle_rewards = []
    interval_hits = []
    for unit in test_units:
        event = events[str(unit["event_id"])]
        outcomes = unit["arms"][primary_arm.value]
        q_use = float(outcomes["q_use"])
        q_drop = float(outcomes["q_control"])
        utility = q_use - q_drop
        prediction = estimator.predict(builder.build(event))
        decision = controller.decide(prediction)
        selected_use = decision.action == RelianceAction.ACCEPT
        selected_reward = q_use if selected_use else q_drop
        oracle_use = utility > args.delta
        predictions.append(
            {
                "event_id": event.event_id,
                "claim_id": unit["claim_id"],
                "receiver_agent_id": event.receiver_agent_id,
                "memory_id": event.memory.memory_id,
                "observed_q_use": q_use,
                "observed_q_drop": q_drop,
                "observed_utility": utility,
                "prediction": asdict(prediction),
                "decision": {
                    **asdict(decision),
                    "action": decision.action.value,
                    "prediction": asdict(prediction),
                },
                "verify_policy": "abstain_as_drop",
                "selected_reward": selected_reward,
            }
        )
        true_use.append(q_use)
        true_drop.append(q_drop)
        true_utility.append(utility)
        predicted_use.append(prediction.q_use)
        predicted_drop.append(prediction.q_drop)
        predicted_utility.append(prediction.utility)
        controller_rewards.append(selected_reward)
        oracle_rewards.append(q_use if oracle_use else q_drop)
        interval_hits.append(
            abs(prediction.utility - utility) <= args.kappa * prediction.uncertainty
        )
    action_counts = {
        action.value: sum(row["decision"]["action"] == action.value for row in predictions)
        for action in RelianceAction
    }
    return {
        "status": "fitted_and_evaluated",
        "primary_arm": primary_arm.value,
        "split_unit": "claim_id",
        "train_claims": sorted(train_claims),
        "test_claims": sorted(test_claims),
        "train_units": len(train_units),
        "test_units": len(test_units),
        "q_use_rmse": _rmse(true_use, predicted_use),
        "q_drop_rmse": _rmse(true_drop, predicted_drop),
        "utility_rmse": _rmse(true_utility, predicted_utility),
        "utility_sign_accuracy": sum(
            _sign(left, args.delta) == _sign(right, args.delta)
            for left, right in zip(true_utility, predicted_utility)
        )
        / len(test_units),
        "uncertainty_interval_coverage": sum(interval_hits) / len(interval_hits),
        "action_counts": action_counts,
        "controller_mean_reward": sum(controller_rewards) / len(controller_rewards),
        "always_use_mean_reward": sum(true_use) / len(true_use),
        "never_use_mean_reward": sum(true_drop) / len(true_drop),
        "oracle_mean_reward": sum(oracle_rewards) / len(oracle_rewards),
        "predictions": predictions,
    }


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _validate_args(args)
    if args.cache_seed_from is not None and not args.cache_seed_from.is_file():
        raise SystemExit(f"cache seed file does not exist: {args.cache_seed_from}")
    receivers = _parse_receivers(args.receivers)
    requested_arms = _parse_arms(args.arms)
    primary_arm = InterventionArm(args.primary_arm)
    if primary_arm not in requested_arms:
        raise SystemExit(
            f"--primary-arm {primary_arm.value} must also be present in --arms"
        )

    input_hashes = {
        "test_md5": _md5(args.test),
        "experience_bank_md5": _md5(args.experience_bank),
        "distractor_bank_md5": _md5(args.distractor_bank),
    }
    selected_claims = select_claims(
        load(args.test), args.claims, args.selection_seed
    )
    experience_bank = load(args.experience_bank, binary=False)
    distractor_bank = load(args.distractor_bank, binary=False)
    _validate_inputs(selected_claims, experience_bank, distractor_bank)

    design = {
        "runner_schema": RUNNER_SCHEMA,
        "benchmark": "FEVER_binary",
        "model": args.model,
        "embedding_model": args.embedding_model,
        "retrieval_threshold": args.retrieval_threshold,
        "claims": args.claims,
        "repeats": args.repeats,
        "receivers": receivers,
        "arms": tuple(arm.value for arm in requested_arms),
        "selection_seed": args.selection_seed,
        "temperature": 0.7,
        "top_p": 0.8,
        **input_hashes,
    }
    run_config = {
        **design,
        "endpoint": args.endpoint,
        "sample_seed_base": args.sample_seed_base,
        "cache_seed_from": (
            str(args.cache_seed_from.resolve()) if args.cache_seed_from else None
        ),
    }
    design_hash = _stable_hash(design)
    run_hash = _stable_hash(run_config)
    log_path, existing_rows = _prepare_output(args, design, run_config)

    cache_path = args.output_dir / "llm_cache.sqlite"
    if args.cache_seed_from is not None and not cache_path.exists():
        shutil.copy2(args.cache_seed_from, cache_path)

    client = CachedChat(
        args.endpoint,
        args.model,
        cache_path,
        api_key=args.api_key,
    )
    index = GMemorySemanticIndex(
        experience_bank,
        args.embedding_model,
        args.retrieval_threshold,
    )
    audits: list[CounterfactualAuditResult] = []
    events: dict[str, MemoryUseEvent] = {}
    excluded_no_memory = 0
    excluded_invalid_placebo = 0
    sample_seeds = tuple(
        args.sample_seed_base + repeat for repeat in range(args.repeats)
    )

    for claim in selected_claims:
        shared = retrieve(claim, experience_bank, "A1", 1, index)
        if not shared:
            excluded_no_memory += 1
            continue
        candidates = {agent: list(shared) for agent in AGENTS}
        item = shared[0]
        if not memory_is_eligible(claim, item):
            excluded_no_memory += 1
            continue
        replacement = None
        if InterventionArm.PLACEBO in requested_arms:
            replacement = placebo(
                item,
                experience_bank,
                forbidden_evidence=_evidence_texts(claim),
            )
            if not replacement.get("placebo_valid", False):
                replacement = None
                excluded_invalid_placebo += len(receivers)

        for receiver in receivers:
            target_id = str(item["memory_id"])
            event_id = _event_id(str(claim["id"]), receiver, target_id)
            event = build_fever_event(
                claim,
                candidates,
                receiver=receiver,
                target_id=target_id,
                event_id=event_id,
            )
            events[event_id] = event
            controls = tuple(
                arm
                for arm in requested_arms
                if arm != InterventionArm.PLACEBO or replacement is not None
            )
            replacement_candidate = (
                placebo_candidate(replacement) if replacement is not None else None
            )
            builder = InterventionBuilder(
                (replacement_candidate,) if replacement_candidate else ()
            )
            branch_runner = FeverBranchRunner(
                client=client,
                claim=claim,
                candidates=candidates,
                placebo_item=replacement,
                log_path=log_path,
                design_hash=design_hash,
                run_hash=run_hash,
                existing_rows=existing_rows,
            )
            audit = CounterfactualAudit(
                branch_runner,
                intervention_builder=builder,
                repeats=args.repeats,
            ).run(
                build_fever_checkpoint(event, str(claim["id"])),
                event,
                arms=controls,
                seeds=sample_seeds,
            )
            audits.append(audit)

    if not audits:
        raise SystemExit("no valid audit units were produced")
    units = [summarize_audit(audit) for audit in audits]
    units_payload = {
        "runner_schema": RUNNER_SCHEMA,
        "design_hash": design_hash,
        "run_hash": run_hash,
        "units": units,
    }
    _json_dump(args.output_dir / "audit_units.json", units_payload)

    previous_units = None
    if args.retest_results is not None:
        previous_units = _load_retest_units(
            design_hash, args.retest_results
        )
    arm_evaluations: dict[str, Any] = {}
    arm_objects: dict[str, Any] = {}
    for arm in requested_arms:
        arm_units = _units_for_arm(units, arm)
        if not arm_units:
            arm_evaluations[arm.value] = {
                "status": "unavailable",
                "reason": "arm_has_no_valid_audit_units",
                "audit_units": 0,
            }
            continue
        oracle = _oracle_evaluation(units, arm, args.delta)
        within_noise = _noise_report_from_audits(
            audits, arm, neutral_epsilon=args.delta
        )
        independent_noise = None
        retest_coverage = 0.0
        transitions = None
        if previous_units is not None:
            independent_noise, retest_coverage, transitions = (
                _independent_arm_diagnostics(
                    units,
                    previous_units,
                    arm,
                    neutral_epsilon=args.delta,
                )
            )
        bootstrap = _oracle_gain_bootstrap(
            units,
            arm,
            oracle,
            samples=args.oracle_bootstrap_samples,
            confidence_level=args.oracle_confidence_level,
            seed=args.selection_seed,
        )
        gate = _arm_gate(
            oracle=oracle,
            within_noise=within_noise,
            independent_noise=independent_noise,
            retest_coverage=retest_coverage,
            transition_diagnostics=transitions,
            bootstrap=bootstrap,
            args=args,
        )
        arm_objects[arm.value] = oracle
        arm_evaluations[arm.value] = {
            "status": "ok",
            "audit_units": len(arm_units),
            "oracle": asdict(oracle),
            "within_run_repeat_noise": asdict(within_noise),
            "independent_retest_noise": (
                asdict(independent_noise)
                if independent_noise is not None
                else None
            ),
            "gate": gate,
        }
    primary_evaluation = arm_evaluations.get(primary_arm.value)
    if not primary_evaluation or primary_evaluation.get("status") != "ok":
        raise SystemExit(f"primary arm {primary_arm.value} has no valid audit units")
    if args.retest_results is not None and primary_evaluation["gate"].get(
        "independent_retest_coverage", 0.0
    ) == 0.0:
        raise SystemExit("independent retest has no matching primary-arm event IDs")
    gate_payload = {
        "primary_arm": primary_arm.value,
        **primary_evaluation["gate"],
    }
    oracle_payload = {
        "primary_arm": primary_arm.value,
        **asdict(arm_objects[primary_arm.value]),
        "gain_bootstrap": gate_payload["oracle_gain_bootstrap"],
    }
    _json_dump(args.output_dir / "arm_evaluations.json", {
        "runner_schema": RUNNER_SCHEMA,
        "primary_arm": primary_arm.value,
        "arms": arm_evaluations,
    })
    _json_dump(args.output_dir / "oracle_evaluation.json", oracle_payload)
    _json_dump(args.output_dir / "pretraining_gate.json", gate_payload)

    pivotality = stratify_pivotality(_pivotality_rows(audits))
    primary_within_noise = primary_evaluation["within_run_repeat_noise"]
    primary_independent_noise = primary_evaluation["independent_retest_noise"]
    raw_rows = list(existing_rows.values())
    diagnostics = {
        "primary_arm": primary_arm.value,
        "n_selected_claims": len(selected_claims),
        "n_audit_units": len(units),
        "excluded_no_eligible_memory": excluded_no_memory,
        "excluded_invalid_placebo_arms": excluded_invalid_placebo,
        "memory_observation_count_distribution": observation_count_distribution(
            events.values()
        ),
        "pivotality_bins": [asdict(row) for row in pivotality],
        "pivotality_estimand": "receiver_marginal_drop",
        "within_run_repeat_noise": primary_within_noise,
        "independent_retest_noise": primary_independent_noise,
        "per_arm_gate_summary": {
            name: evaluation.get("gate")
            for name, evaluation in arm_evaluations.items()
            if evaluation.get("status") == "ok"
        },
        "causal_effect_summary": _causal_effect_summary(
            units, delta=args.delta
        ),
        "confidence_diagnostics": _confidence_diagnostics(raw_rows),
        "llm_calls": int(client.calls),
        "cache_hits": int(client.cache_hits),
        "logged_collection_llm_calls": sum(
            int(row.get("branch_llm_calls", 0)) for row in raw_rows
        ),
        "logged_collection_cache_hits": sum(
            int(row.get("branch_cache_hits", 0)) for row in raw_rows
        ),
    }
    _json_dump(args.output_dir / "diagnostics.json", diagnostics)

    estimator = _fit_and_evaluate(
        units=units,
        events=events,
        primary_arm=primary_arm,
        args=args,
        gate=gate_payload,
    )
    _json_dump(args.output_dir / "estimator_evaluation.json", estimator)
    report = (
        "# FEVER causal-memory audit\n\n"
        f"- Primary arm: {primary_arm.value}\n"
        f"- Audit units: {len(units)}\n"
        f"- LLM calls in this process: {client.calls}\n"
        f"- Cache hits in this process: {client.cache_hits}\n"
        f"- |O(m)| distribution: {diagnostics['memory_observation_count_distribution']}\n"
        f"- Oracle point-estimate gate: "
        f"{'PASS' if gate_payload['oracle_point_passed'] else 'FAIL'}\n"
        f"- Oracle bootstrap-CI gate: "
        f"{'PASS' if gate_payload['oracle_ci_passed'] else 'FAIL'} "
        f"(CI={gate_payload['oracle_gain_bootstrap'].get('ci_low')}, "
        f"{gate_payload['oracle_gain_bootstrap'].get('ci_high')})\n"
        f"- Sign consistency: {gate_payload['sign_consistency']:.3f} "
        f"({gate_payload['sign_consistency_source']})\n"
        f"- Direct opposite-sign rate: "
        f"{(gate_payload.get('sign_transition_diagnostics') or {}).get('direct_opposite_sign_rate')}\n"
        f"- Nonzero/neutral transition rate: "
        f"{(gate_payload.get('sign_transition_diagnostics') or {}).get('nonzero_neutral_transition_rate')}\n"
        f"- Stable nonzero rate: "
        f"{(gate_payload.get('sign_transition_diagnostics') or {}).get('stable_nonzero_rate')}\n"
        f"- Pretraining gate: {'PASS' if gate_payload['can_train'] else 'BLOCKED'}\n"
        f"- Gate reasons: {gate_payload['reasons'] or ['none']}\n"
        f"- Estimator: {estimator['status']}\n"
    )
    (args.output_dir / "run_report.md").write_text(report, encoding="utf-8")
    if hasattr(client, "db"):
        client.db.close()
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_audit_units": len(units),
                "llm_calls": diagnostics["llm_calls"],
                "cache_hits": diagnostics["cache_hits"],
                "primary_arm": primary_arm.value,
                "oracle_point_passed": gate_payload["oracle_point_passed"],
                "oracle_ci_passed": gate_payload["oracle_ci_passed"],
                "oracle_passed": gate_payload["oracle_passed"],
                "sign_consistency": gate_payload["sign_consistency"],
                "pretraining_can_train": gate_payload["can_train"],
                "estimator_status": estimator["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return args.output_dir


if __name__ == "__main__":
    main()
