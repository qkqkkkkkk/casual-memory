#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import CachedChat
from .mas import run_episode
from .run_experiment import load, retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="G0 cache/determinism gate for the real-LLM MAS")
    p.add_argument("--test", type=Path, required=True); p.add_argument("--memory-bank", type=Path, required=True); p.add_argument("--endpoint", default="http://127.0.0.1:11434/v1"); p.add_argument("--api-key", default=None); p.add_argument("--model", default="qwen2.5:7b"); p.add_argument("--top-k", type=int, default=6); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "g0_report.json"
    if report_path.exists(): raise SystemExit(f"Refusing to overwrite existing report: {report_path}")
    claim = load(args.test)[0]; bank = load(args.memory_bank, binary=False)
    candidates = {aid: retrieve(claim, bank, aid, args.top_k) for aid in ("A1", "A2", "A3")}
    client = CachedChat(args.endpoint, args.model, args.output_dir / "llm_cache.sqlite", api_key=args.api_key)
    first = run_episode(client, claim, candidates, 0, "memory_all")
    second = run_episode(client, claim, candidates, 0, "memory_all")
    same = first.round1 == second.round1 and first.round2 == second.round2 and first.team_verdict == second.team_verdict
    report = {"pass": bool(same and second.llm_calls == 0 and second.cache_hits == 6), "semantic_outputs_identical": same, "second_run_llm_calls": second.llm_calls, "second_run_cache_hits": second.cache_hits, "expected_second_run_cache_hits": 6}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))
    if not report["pass"]: raise SystemExit(2)


if __name__ == "__main__": main()
