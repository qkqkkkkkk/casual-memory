#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Render the real-LLM E1 figure")
    p.add_argument("--e1-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmi-mpl")
    import matplotlib.pyplot as plt

    e1 = json.loads((args.e1_dir / "mismatch_rate.json").read_text(encoding="utf-8")); rows = e1["units"]
    colors = ["#d62728" if row.get("confirmed") else "#9aa0a6" if row.get("classification") != "other" else "#1f77b4" for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 5.2)); ax.scatter([r["local_b"] for r in rows], [r["team"] for r in rows], c=colors, s=18, alpha=.7)
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8); ax.set(xlabel="Local causal utility B", ylabel="Team causal utility", title=f"Real-LLM FEVER mismatch ({e1['model']})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig1_llm_scatter.png", dpi=220, format="png")
    plt.close(fig)

    print(args.output_dir)


if __name__ == "__main__": main()
