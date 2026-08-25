#!/usr/bin/env python3
"""Render P2 PDFs from an existing mismatch_rate.json and diversity.json."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmi-mpl")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from exc

    results_dir = args.results_dir
    summary = json.loads((results_dir / "mismatch_rate.json").read_text(encoding="utf-8"))["units"]
    diversity = json.loads((results_dir / "diversity.json").read_text(encoding="utf-8"))

    colors = ["#d62728" if row.get("classification") == "local_positive_team_negative" else "#1f77b4" for row in summary]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([row["local_b"] for row in summary], [row["team"] for row in summary], c=colors, s=12, alpha=.65)
    ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7)
    ax.set(xlabel="Local causal utility B", ylabel="Team causal utility", title="Local-positive / team-negative mismatch")
    fig.tight_layout(); fig.savefig(results_dir / "fig1_scatter.pdf"); plt.close(fig)

    labels = [row["condition"] for row in diversity]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for axis, key, title in zip(axes, ("individual_accuracy", "error_correlation", "team_accuracy"), ("Individual accuracy p", "Error correlation rho", "Team accuracy")):
        axis.bar(range(len(labels)), [row[key] for row in diversity], color=["#2ca02c", "#7f7f7f"])
        axis.set_xticks(range(len(labels)), labels, rotation=20); axis.set_title(title); axis.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(results_dir / "fig2_diversity.pdf"); plt.close(fig)
    print(f"Rendered PDFs in {results_dir}")


if __name__ == "__main__":
    main()
