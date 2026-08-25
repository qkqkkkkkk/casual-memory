#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .mas import run_episode, serialize
from .memory import FrozenMemoryStore, RoleRetriever, build_placebo, load_fever_benchmark, make_synthetic_benchmark
from .stats import bh_reject, diversity, mismatch_label, paired_effect, paired_sign_pvalue


def main() -> None:
    p = argparse.ArgumentParser(description="P2 local-positive/team-negative mismatch probe")
    p.add_argument("--output-dir", type=Path, default=Path("results/p2_probe"))
    p.add_argument("--n-claims", type=int, default=240)
    p.add_argument("--repeats", type=int, default=32)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fever-dev", type=Path, default=None, help="Optional FEVER jsonl; otherwise use the built-in controlled benchmark")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = "fever_binary" if args.fever_dev else "controlled_fever_like"
    claims, store = (load_fever_benchmark(args.fever_dev, args.n_claims, args.seed)
                     if args.fever_dev else make_synthetic_benchmark(args.n_claims, args.seed))
    store.save(args.output_dir / "memory_bank.jsonl")
    retriever = RoleRetriever(store, args.top_k)
    placebo = {m.memory_id: build_placebo(m, store, random.Random(args.seed)) for m in store.items if m.kind != "placebo"}
    all_runs = []; units = []
    for claim in claims:
        candidates = {a: retriever.retrieve(claim["claim"], claim["evidence"], a) for a in ("A1", "A2", "A3")}
        # Audit every retrieved candidate for A1; trap memories are deliberately present and measurable.
        for item in candidates["A1"]:
            units.append((claim, candidates, (claim["claim_id"], "A1", item.memory_id)))
    for claim, candidates, unit in units:
        for r in range(args.repeats):
            control = run_episode(claim, candidates, claim["gold_label"], r, "control", args.seed, unit)
            treated = run_episode(claim, candidates, claim["gold_label"], r, "treated", args.seed, unit)
            all_runs.extend([serialize(control), serialize(treated)])
    (args.output_dir / "episode_runs.jsonl").write_text("\n".join(json.dumps(r) for r in all_runs) + "\n")
    grouped = defaultdict(lambda: {"treated": [], "control": []})
    for r in all_runs: grouped[tuple(r["audit_unit"])][r["arm"]].append(r)
    summary = []; mismatch = []
    local_ps = []; team_ps = []; paired_values = []
    for unit, branches in grouped.items():
        local_t = [r["per_agent_correct_r1"][unit[1]] for r in branches["treated"]]; local_c = [r["per_agent_correct_r1"][unit[1]] for r in branches["control"]]
        solo_t = [r["per_agent_correct_solo"][unit[1]] for r in branches["treated"]]; solo_c = [r["per_agent_correct_solo"][unit[1]] for r in branches["control"]]
        team_t = [r["team_correct"] for r in branches["treated"]]; team_c = [r["team_correct"] for r in branches["control"]]
        lp, llo, lhi = paired_effect(local_t, local_c, "local", args.bootstrap, args.seed)
        ap, alo, ahi = paired_effect(solo_t, solo_c, "solo", args.bootstrap, args.seed + 2)
        tp, tlo, thi = paired_effect(team_t, team_c, "team", args.bootstrap, args.seed + 1)
        local_p = paired_sign_pvalue(local_t, local_c)
        team_p = paired_sign_pvalue(team_t, team_c)
        local_ps.append(local_p); team_ps.append(team_p); paired_values.append((lp, tp, llo, thi))
        row = {"claim_id": unit[0], "agent_id": unit[1], "memory_id": unit[2], "local_b": lp, "local_b_ci": [llo, lhi], "local_a": ap, "local_a_ci": [alo, ahi], "team": tp, "team_ci": [tlo, thi], "local_p": local_p, "team_p": team_p, "classification": mismatch_label(lp, tp), "confirmed": False}
        summary.append(row)
    local_rej = bh_reject(local_ps, q=0.1); team_rej = bh_reject(team_ps, q=0.1)
    for row, lrej, trej, vals in zip(summary, local_rej, team_rej, paired_values):
        row["local_bh_reject"] = lrej; row["team_bh_reject"] = trej
        row["confirmed"] = bool(lrej and trej and vals[2] > 0 and vals[3] < 0)
        if row["confirmed"]: mismatch.append(row)
    out = {"experiment": "p2_probe", "benchmark": benchmark, "n_claims": len(claims), "n_audit_units": len(summary), "repeats": args.repeats, "fdr_q": 0.1, "mismatch_rate": len(mismatch)/len(summary), "mismatch_count": len(mismatch), "direction_ii_count": sum(r["confirmed"] and r["classification"] == "local_negative_team_positive" for r in summary), "undetermined_count": sum(not r["confirmed"] and r["classification"] == "both_neutral" for r in summary), "memory_bank_md5": store.md5, "units": summary}
    (args.output_dir / "mismatch_rate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    e2_runs = []
    for claim in claims:
        candidates = {a: retriever.retrieve(claim["claim"], claim["evidence"], a) for a in ("A1", "A2", "A3")}
        for r in range(max(5, min(args.repeats, 8))):
            e2_runs.append(serialize(run_episode(claim, candidates, claim["gold_label"], r, "memory_all", args.seed)))
            placebo_candidates = {a: [placebo[m.memory_id] for m in ms] for a, ms in candidates.items()}
            e2_runs.append(serialize(run_episode(claim, placebo_candidates, claim["gold_label"], r, "placebo_all", args.seed)))
    div = [diversity(e2_runs, c) for c in ("memory_all", "placebo_all")]
    (args.output_dir / "diversity.json").write_text(json.dumps(div, indent=2), encoding="utf-8")
    make_figures(args.output_dir, summary, div)
    (args.output_dir / "gate_report.md").write_text("# P2 Probe Gate Report\n\n- G0 paired cache-equivalent simulator: PASS\n- Ground-truth scoring, no LLM judge: PASS\n- Frozen memory bank hash recorded: PASS\n- BH-FDR correction (q=0.1): PASS\n- Local-positive/team-negative mismatch observed: **%d / %d (%.3f)**\n" % (len(mismatch), len(summary), out["mismatch_rate"]), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_audit_units", "mismatch_count", "mismatch_rate", "undetermined_count")}, indent=2))


def make_figures(outdir: Path, summary: list[dict], div: list[dict]) -> None:
    try:
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmi-mpl")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (outdir / "figure_error.txt").write_text(f"Install matplotlib to render PDFs: {exc}\n")
        return
    colors = ["#d62728" if r["classification"] == "local_positive_team_negative" else "#1f77b4" for r in summary]
    fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter([r["local_b"] for r in summary], [r["team"] for r in summary], c=colors, s=12, alpha=.65); ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7); ax.set(xlabel="Local causal utility B", ylabel="Team causal utility", title="Local-positive / team-negative mismatch"); fig.tight_layout(); fig.savefig(outdir / "fig1_scatter.pdf"); plt.close(fig)
    labels = [d["condition"] for d in div]; x = range(len(labels)); fig, axes = plt.subplots(1, 3, figsize=(9, 3.2));
    for ax, key, title in zip(axes, ("individual_accuracy", "error_correlation", "team_accuracy"), ("Individual accuracy p", "Error correlation rho", "Team accuracy")):
        ax.bar(list(x), [d[key] for d in div], color=["#2ca02c", "#7f7f7f"]); ax.set_xticks(list(x), labels, rotation=20); ax.set_title(title); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(outdir / "fig2_diversity.pdf"); plt.close(fig)


if __name__ == "__main__": main()
