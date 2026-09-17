# RQ3: Recipient Heterogeneity

This folder is a self-contained experiment wrapper for the question:

> Does the same retrieved memory have different team-level causal utility for
> different recipient agents?

It reuses the existing real FEVER/P2 GMemory retriever, prompts, three-agent
workflow, inference client, response cache, and receiver-level intervention.
It does not change memory construction, retrieval, or the MAS architecture.

## Estimand and design

For one frozen claim/state and one shared Top-1 memory, the runner first
evaluates the common `USE_ALL` branch and then removes that memory from one
recipient at a time:

```text
U_team(m, Ai) = Y(USE_ALL) - Y(DROP_FROM_Ai_ONLY)
```

Every claim-memory event must contain A1, A2, and A3. The analysis refuses to
continue if a recipient is missing or if the matched-seed `USE_ALL` outcomes
differ across recipients.

The primary descriptive statistic is the direct recipient sign-flip rate:

```text
P(event contains at least one positive and one negative recipient utility)
```

Because binary rewards and finite repeats produce many exact zero estimates,
the report separately provides:

- direct positive/negative sign flips;
- any sign heterogeneity, including nonzero/neutral differences;
- the within-event utility range;
- all three pairwise recipient comparisons;
- event-level bootstrap confidence intervals.

A second run with a disjoint inference-seed range is required. The strongest
noise-aware RQ3 statistic is the reproducible directional flip rate: the same
recipient is positive and another recipient is negative in both runs.

## Outputs

In addition to the files produced by `causal_memory_control.run_fever_audit`,
the wrapper writes:

- `rq3_analysis.json`: complete single-run and independent-retest statistics;
- `recipient_matrix.csv`: one row per claim-memory event with A1/A2/A3 utility;
- `rq3_report.md`: a compact results table and noise-aware interpretation.

## Tests

From the repository root:

```bash
python -m unittest discover -s recipient_heterogeneity/tests -v
```

## 1. Cheap smoke run

Use the endpoint and served model name configured on the experiment server:

```bash
python -m recipient_heterogeneity.run_fever_rq3 \
  --test results/fever_difficulty_qwen7b_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --claims 2 \
  --repeats 2 \
  --sample-seed-base 0 \
  --output-dir results/rq3_fever_qwen3b_smoke_seed0
```

Inspect:

```bash
cat results/rq3_fever_qwen3b_smoke_seed0/rq3_report.md
```

## 2. Full first run

The following uses the same 87 requested claims and four repeats as the RQ2
audit. The effective event count may be smaller when retrieval has no eligible
memory; the previous run produced 85 valid events.

```bash
python -m recipient_heterogeneity.run_fever_rq3 \
  --test results/fever_difficulty_qwen7b_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --claims 87 \
  --repeats 4 \
  --sample-seed-base 0 \
  --cache-seed-from results/cmc_fever_qwen3b_global_87_r4_seed0/llm_cache.sqlite \
  --output-dir results/rq3_fever_qwen3b_87_r4_seed0
```

`--cache-seed-from` is optional. It safely copies the previous seed-0 cache
into the new output directory and can reuse identical prompts without changing
the old result. Omit the option if that SQLite file is not present.

## 3. Independent retest

Keep every design argument unchanged. Only change the inference seed range and
output directory, and point `--retest-results` at the first RQ3 run:

```bash
python -m recipient_heterogeneity.run_fever_rq3 \
  --test results/fever_difficulty_qwen7b_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11436/v1 \
  --model qwen2.5:3b \
  --claims 87 \
  --repeats 4 \
  --sample-seed-base 1000 \
  --cache-seed-from results/cmc_fever_qwen3b_global_87_r4_seed1000_retest/llm_cache.sqlite \
  --retest-results results/rq3_fever_qwen3b_87_r4_seed0 \
  --output-dir results/rq3_fever_qwen3b_87_r4_seed1000_retest
```

The final report is:

```bash
cat results/rq3_fever_qwen3b_87_r4_seed1000_retest/rq3_report.md
```

## Reanalyze without LLM calls

Analysis can be regenerated from completed audit units:

```bash
python -m recipient_heterogeneity.analysis \
  --results results/rq3_fever_qwen3b_87_r4_seed1000_retest \
  --retest-results results/rq3_fever_qwen3b_87_r4_seed0 \
  --bootstrap-samples 10000
```

Do not interpret a one-run sign flip as conclusive recipient heterogeneity.
Report the one-run rate as descriptive and use the stable/reproducible rates
from the independent retest as the main evidence.
