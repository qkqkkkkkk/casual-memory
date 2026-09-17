#!/usr/bin/env python3
"""Collect and analyze the complete FEVER RQ3 recipient audit.

This is a thin, fail-closed wrapper around the existing real FEVER/P2 runner.
It fixes the scientific design to the same Top-1 memory in A1/A2/A3 and to a
receiver-level DROP arm, then runs the recipient-heterogeneity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal_memory_control.run_fever_audit import main as run_audit

from .analysis import write_analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--experience-bank", type=Path, required=True)
    parser.add_argument("--distractor-bank", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--retrieval-threshold", type=float, default=0.3)
    parser.add_argument("--claims", type=int, default=87)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sample-seed-base", type=int, default=0)
    parser.add_argument("--neutral-epsilon", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--retest-results", type=Path, default=None)
    parser.add_argument("--cache-seed-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--test",
        str(args.test),
        "--experience-bank",
        str(args.experience_bank),
        "--distractor-bank",
        str(args.distractor_bank),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--embedding-model",
        args.embedding_model,
        "--retrieval-threshold",
        str(args.retrieval_threshold),
        "--claims",
        str(args.claims),
        "--repeats",
        str(args.repeats),
        "--receivers",
        "A1,A2,A3",
        "--arms",
        "drop",
        "--primary-arm",
        "drop",
        "--selection-seed",
        str(args.selection_seed),
        "--sample-seed-base",
        str(args.sample_seed_base),
        "--delta",
        str(args.neutral_epsilon),
        "--oracle-bootstrap-samples",
        str(args.bootstrap_samples),
        "--oracle-confidence-level",
        str(args.confidence_level),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.api_key is not None:
        forwarded.extend(("--api-key", args.api_key))
    if args.retest_results is not None:
        forwarded.extend(("--retest-results", str(args.retest_results)))
    if args.cache_seed_from is not None:
        forwarded.extend(("--cache-seed-from", str(args.cache_seed_from)))
    if args.resume:
        forwarded.append("--resume")
    return forwarded


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.retest_results is not None:
        if args.retest_results.resolve() == args.output_dir.resolve():
            raise SystemExit("--retest-results and --output-dir must differ")
    run_audit(_forwarded_args(args))
    analysis_path = write_analysis(
        args.output_dir,
        retest_results=args.retest_results,
        neutral_epsilon=args.neutral_epsilon,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.selection_seed,
    )
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    current = payload["current_run"]
    retest = payload.get("independent_retest")
    print(
        json.dumps(
            {
                "rq3_analysis": str(analysis_path),
                "events": current["event_count"],
                "recipient_audit_units": current["recipient_event_count"],
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
    return analysis_path


if __name__ == "__main__":
    main()
