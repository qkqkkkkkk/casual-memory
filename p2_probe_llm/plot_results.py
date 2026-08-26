#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Render real-LLM P2 E1/E2 figures")
    p.add_argument("--e1-dir", type=Path, required=True); p.add_argument("--e2-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmi-mpl")
    import matplotlib.pyplot as plt

    e1 = json.loads((args.e1_dir / "mismatch_rate.json").read_text(encoding="utf-8")); rows = e1["units"]
    colors = ["#d62728" if row.get("confirmed") else "#9aa0a6" if row.get("classification") != "other" else "#1f77b4" for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 5.2)); ax.scatter([r["local_b"] for r in rows], [r["team"] for r in rows], c=colors, s=18, alpha=.7)
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8); ax.set(xlabel="Local causal utility B", ylabel="Team causal utility", title=f"Real-LLM FEVER mismatch ({e1['model']})")
    fig.tight_layout(); fig.savefig(args.output_dir / "fig1_llm_scatter.pdf"); plt.close(fig)

    e2 = json.loads((args.e2_dir / "diversity.json").read_text(encoding="utf-8")); conditions = e2["conditions"]; labels = [x["condition"] for x in conditions]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    for ax, key, title in zip(axes, ("individual_accuracy", "error_correlation", "team_accuracy_round1"), ("Individual accuracy p", "Error correlation rho", "Round-1 majority accuracy")):
        ax.bar(range(len(labels)), [x[key] for x in conditions], color=["#2ca02c", "#7f7f7f"]); ax.set_xticks(range(len(labels)), labels, rotation=20); ax.set_title(title); ax.set_ylim(-.1 if key == "error_correlation" else 0, 1)
    fig.tight_layout(); fig.savefig(args.output_dir / "fig2_llm_diversity.pdf"); plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__": main()
