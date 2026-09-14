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
from .controller import RelianceController
from .types import (
    CounterfactualAuditResult,
    InterventionArm,
    MemoryUseEvent,
    RelianceAction,
)


RUNNER_SCHEMA = "causal-memory-control-fever-v1"
PRIMARY_ARM = InterventionArm.DROP


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
    if PRIMARY_ARM not in arms:
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
) -> Any:
    return oracle_noise_floor(
        UtilityRetest(audit.event.event_id, pair.team_utility)
        for audit in audits
        for pair in audit.primary.pairs
    )


def _load_independent_noise(
    current_units: Sequence[Mapping[str, Any]],
    current_design_hash: str,
    previous_dir: Path,
) -> tuple[Any, float]:
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
    previous = {unit["event_id"]: unit for unit in payload["units"]}
    current = {unit["event_id"]: unit for unit in current_units}
    shared = sorted(set(previous) & set(current))
    if not shared:
        raise SystemExit("independent retest has no matching event IDs")
    observations = []
    for event_id in shared:
        for source in (previous[event_id], current[event_id]):
            observations.append(
                UtilityRetest(
                    event_id,
                    float(source["arms"][PRIMARY_ARM.value]["team_utility"]["mean"]),
                )
            )
    return oracle_noise_floor(observations), len(shared) / len(current)


def _oracle_evaluation(units: Sequence[Mapping[str, Any]], delta: float) -> Any:
    examples = []
    for unit in units:
        drop = unit["arms"][PRIMARY_ARM.value]
        scores = {}
        if unit.get("retrieval_score") is not None:
            scores["similarity"] = float(unit["retrieval_score"])
        if unit.get("reliability_prior") is not None:
            scores["reliability"] = float(unit["reliability_prior"])
        examples.append(
            OracleExample(
                event_id=str(unit["event_id"]),
                q_use=float(drop["q_use"]),
                q_drop=float(drop["q_control"]),
                scores=scores,
            )
        )
    return OracleControllability(delta=delta).evaluate(examples)


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
    args: argparse.Namespace,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if not gate["can_train"]:
        return {"status": "blocked_by_pretraining_gate", "reasons": gate["reasons"]}
    train_claims, test_claims = _split_claims(
        units, args.train_fraction, args.selection_seed
    )
    train_units = [unit for unit in units if str(unit["claim_id"]) in train_claims]
    test_units = [unit for unit in units if str(unit["claim_id"]) in test_claims]
    if len(train_units) < args.min_train_samples or not test_units:
        return {
            "status": "insufficient_split",
            "train_units": len(train_units),
            "test_units": len(test_units),
            "minimum_train_samples": args.min_train_samples,
        }
    builder = EventFeatureBuilder()
    examples = []
    for unit in train_units:
        event = events[str(unit["event_id"])]
        drop = unit["arms"][PRIMARY_ARM.value]
        examples.append(
            PotentialOutcomeExample(
                features=builder.build(event),
                q_use=float(drop["q_use"]),
                q_drop=float(drop["q_control"]),
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
        return {"status": "estimator_fit_failed", "train_units": len(train_units)}
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
        drop = unit["arms"][PRIMARY_ARM.value]
        q_use = float(drop["q_use"])
        q_drop = float(drop["q_control"])
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

    oracle = _oracle_evaluation(units, args.delta)
    oracle_payload = asdict(oracle)
    _json_dump(args.output_dir / "oracle_evaluation.json", oracle_payload)
    within_noise = _noise_report_from_audits(audits)
    independent_noise = None
    retest_coverage = 0.0
    if args.retest_results is not None:
        independent_noise, retest_coverage = _load_independent_noise(
            units, design_hash, args.retest_results
        )
    gate_noise = independent_noise or within_noise
    base_gate = evaluate_pretraining_gate(
        oracle,
        gate_noise,
        minimum_sign_consistency=args.minimum_sign_consistency,
        minimum_oracle_gain=args.minimum_oracle_gain,
    )
    gate_reasons = list(base_gate.reasons)
    can_train = base_gate.can_train
    if independent_noise is None:
        can_train = False
        gate_reasons.append("independent_retest_required")
    elif retest_coverage < args.minimum_retest_coverage:
        can_train = False
        gate_reasons.append("independent_retest_coverage_below_threshold")
    gate_payload = {
        **asdict(base_gate),
        "can_train": can_train,
        "reasons": gate_reasons,
        "sign_consistency_source": (
            "independent_run_level_retest"
            if independent_noise is not None
            else "within_run_repeats_diagnostic_only"
        ),
        "independent_retest_coverage": retest_coverage,
        "minimum_retest_coverage": args.minimum_retest_coverage,
    }
    _json_dump(args.output_dir / "pretraining_gate.json", gate_payload)

    pivotality = stratify_pivotality(_pivotality_rows(audits))
    diagnostics = {
        "n_selected_claims": len(selected_claims),
        "n_audit_units": len(units),
        "excluded_no_eligible_memory": excluded_no_memory,
        "excluded_invalid_placebo_arms": excluded_invalid_placebo,
        "memory_observation_count_distribution": observation_count_distribution(
            events.values()
        ),
        "pivotality_bins": [asdict(row) for row in pivotality],
        "within_run_repeat_noise": asdict(within_noise),
        "independent_retest_noise": (
            asdict(independent_noise) if independent_noise is not None else None
        ),
        "llm_calls": int(client.calls),
        "cache_hits": int(client.cache_hits),
    }
    _json_dump(args.output_dir / "diagnostics.json", diagnostics)

    estimator = _fit_and_evaluate(
        units=units,
        events=events,
        args=args,
        gate=gate_payload,
    )
    _json_dump(args.output_dir / "estimator_evaluation.json", estimator)
    report = (
        "# FEVER causal-memory audit\n\n"
        f"- Audit units: {len(units)}\n"
        f"- LLM calls in this process: {client.calls}\n"
        f"- Cache hits in this process: {client.cache_hits}\n"
        f"- |O(m)| distribution: {diagnostics['memory_observation_count_distribution']}\n"
        f"- Oracle headroom gate: {'PASS' if base_gate.oracle_passed else 'FAIL'}\n"
        f"- Sign consistency: {gate_payload['sign_consistency']:.3f} "
        f"({gate_payload['sign_consistency_source']})\n"
        f"- Pretraining gate: {'PASS' if can_train else 'BLOCKED'}\n"
        f"- Gate reasons: {gate_reasons or ['none']}\n"
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
                "oracle_passed": base_gate.oracle_passed,
                "sign_consistency": gate_payload["sign_consistency"],
                "pretraining_can_train": can_train,
                "estimator_status": estimator["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return args.output_dir


if __name__ == "__main__":
    main()
