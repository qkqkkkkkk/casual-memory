# Mem: P2 mismatch probe

This directory contains a standalone, ground-truth MAS experiment for detecting cases where a memory has positive local causal utility but negative team causal utility.

The checked-in FEVER file is `data/fever/fever_dev.jsonl`. The large ALFWorld
download is intentionally ignored by Git; it is not needed for this P2 probe.

Run from this directory:

```bash
python -m p2_probe.run_experiment --n-claims 240 --repeats 32 --output-dir results/p2_probe
```

To use the binary slice of the FEVER dev split shipped with gmemory:

```bash
python3 -m p2_probe.run_experiment --fever-dev /Users/xiaobaobei/gmemory/data/fever/fever_dev.jsonl --n-claims 240 --repeats 32 --output-dir results/fever_p2_probe
```

Without `--fever-dev`, the built-in controlled FEVER-like benchmark is used;
it is intended as a fast mechanism sanity check and has no LLM judge.

Outputs are `mismatch_rate.json`, `diversity.json`, `gate_report.md`, `fig1_scatter.pdf`, `fig2_diversity.pdf`, `episode_runs.jsonl`, and the frozen `memory_bank.jsonl`.
