# Causal Memory Control

This package implements the final event-level method alongside G-Memory.  It
does not change G-Memory storage, graph construction, retrieval, or write
policy.  The causal unit is one candidate memory used by one receiver at one
frozen state:

```text
x = (query, task state, receiver agent, memory, retrieved candidate set)
```

The implementation has four gated stages:

1. `CounterfactualAudit` replays matched-seed `USE` versus receiver-level
   `DROP`. `PLACEBO` and `GLOBAL_DROP` are optional robustness arms.
2. `OracleControllability` tests whether the oracle policy
   `1[Q_use - Q_drop > delta]` improves final team return before any predictor
   is trained. Similarity, reliability, judge, and local policies are compared
   at the oracle's acceptance budget.
3. `AmortizedUtilityEstimator` learns the two potential outcomes `Q_use` and
   `Q_drop` with a deterministic bootstrap ensemble. Team utility is their
   difference. Receiver behavior change stays a diagnostic signal.
4. `RelianceController` emits only `ACCEPT`, `REJECT`, or `VERIFY`:

```text
ACCEPT  when U_hat - kappa * sigma >  delta
REJECT  when U_hat + kappa * sigma < -delta
VERIFY  otherwise
```

Shapley/PID attribution, propagation actions, GNNs, RL, LoRA, and changes to
the memory write policy are intentionally outside V1.

## G-Memory insertion point

In the current MacNet scheduler, retrieval is performed in
`tasks/mas_workflow/macnet/graph_mas.py`, then `successful_shots` and
`raw_rules` are passed to `format_task_prompt_with_insights()`.  Capture the
checkpoint and run/gate this module between those two operations.

```python
from causal_memory_control import GMemoryRetrievalAdapter

adapter = GMemoryRetrievalAdapter()
adapted = adapter.adapt(meta_memory.retrieve_memory(...))

event = adapter.build_event(
    adapted,
    target_id=adapted.candidates[0].memory_id,
    query=task_main,
    task_state=meta_memory.summarize(upstream_agent_ids=None),
    receiver_agent_id=curr_node.id,
    receiver_role=curr_node._agent.profile,
    recipient_agent_ids=[node.id for node in self._agent_nodes.values()],
)
```

The current MacNet path gives every worker the same retrieved trajectories and
rules.  Passing all worker IDs above therefore records the real `|O(m)|`
instead of silently assuming one recipient.  If role projection makes the
sets differ, pass `candidate_ids_by_recipient` explicitly.

For an audit, the host supplies a `BranchRunner` that restores
`AuditCheckpoint.upstream_state`, fixes the request seed/sampling config,
renders only `request.recipient_contexts`, executes the receiver and downstream
suffix, and returns a `BranchOutcome`.  `CallableBranchRunner` adapts an
ordinary function. `GMemoryRetrievalAdapter.render_prompt_inputs()` turns a
branch request back into `memory_few_shots` and `insights`.

G-Memory does not currently expose a receiver-level suffix checkpoint API.
Until the host provides one, its existing frozen whole-run diagnostic can be
wrapped as a higher-cost fallback, but it must be labeled as whole-run replay;
it is not equivalent to the primary receiver-level intervention.

## Existing P2 data checks

The cheap pre-training checks are public functions:

- `stratify_pivotality`: mismatch rate by the other agents' vote margin.
- `observation_count_distribution`: empirical distribution of `|O(m)|`.
- `oracle_noise_floor`: per-event and aggregate sign consistency across exact
  repeated `(memory, receiver)` configurations.

If only final binary answers were logged, continuous team vote probabilities
cannot be reconstructed.  The FEVER runner used here does retain each agent's
verdict and self-reported confidence.  Vote distributions are therefore
recoverable, while normalized confidence remains an uncalibrated auxiliary
signal rather than a true answer probability.

`evaluate_pretraining_gate` combines the oracle-headroom check with repeated
sign consistency. `CausalMemoryControlMethod.fit()` requires a passing report
by default, so low oracle consistency cannot silently flow into predictor
training. The consistency threshold is configurable and reported explicitly;
it is not hidden inside the estimator.

## Tests

From the repository root:

```bash
python3 -m unittest discover -s causal_memory_control/tests -v
```

All unit tests are dependency-free and do not open or mutate a G-Memory
database.

## Real FEVER/P2 runner

`run_fever_audit` reuses the real `p2_probe_llm` semantic retriever, prompts,
two-round heterogeneous team, OpenAI-compatible client, and SQLite response
cache. Retrieval and current evidence are frozen before the receiver-level
branch. Identical unaffected round-1 prompts use the same inference seed and
must produce byte-identical cached outputs; the runner stops if this invariant
is violated.

The default primary arm is receiver-level `DROP`. `PLACEBO` and `GLOBAL_DROP`
are optional controls. Every invocation supports `--resume` and writes each
completed branch immediately to `audit_runs.jsonl`.

The analysis estimand can be selected independently of collection with
`--primary-arm drop|global_drop`.  This option is deliberately excluded from
the collection design/run hashes: if both arms were already collected, an
existing output can be reanalysed with `--resume --primary-arm global_drop`
without issuing any new LLM requests.  `arm_evaluations.json` always reports
oracle, repeat noise, independent-retest noise, and the gate for every
available arm.  `oracle_evaluation.json`, `pretraining_gate.json`, and the
predictor refer to the selected primary arm.

The oracle gate now has two parts.  The original matched-budget point estimate
must exceed `--minimum-oracle-gain`, and the event-bootstrap confidence
interval must also have a lower bound above that threshold.  Configure the
interval with `--oracle-bootstrap-samples` and
`--oracle-confidence-level`.  This prevents a single minimum reward step in a
small pilot from being reported as robust headroom.

When both receiver `DROP` and `GLOBAL_DROP` were collected, each unit records:

- `receiver_marginal = Y(use_all) - Y(drop_receiver_only)`;
- `exposure_set_total = Y(use_all) - Y(drop_from_all_observers)`;
- `spillover_redundancy = exposure_set_total - receiver_marginal`.

Independent diagnostics separately report strict sign agreement, direct
positive/negative reversals, nonzero/neutral transitions, and stable nonzero
effects.  Raw confidence values are normalized from either `[0,1]` or
`(1,100]` into `[0,1]` and summarized only as an uncalibrated auxiliary
diagnostic; they are not silently substituted for the binary team reward.

Before launching, locate the selected difficulty split produced by the
existing calibration:

```bash
find results -name fever_dev_selected_difficulty.jsonl -print
```

Cheap two-claim smoke run:

```bash
python -m causal_memory_control.run_fever_audit \
  --test results/fever_difficulty_qwen3b_gmemory_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:3b \
  --claims 2 --repeats 2 \
  --receivers A1 --arms drop,placebo \
  --sample-seed-base 1000 \
  --output-dir results/cmc_fever_qwen3b_smoke_r1
```

The first scientifically usable audit run can then use the same 87 selected
claims as the existing E1 result:

```bash
python -m causal_memory_control.run_fever_audit \
  --test results/fever_difficulty_qwen3b_gmemory_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:3b \
  --claims 87 --repeats 4 \
  --receivers A1 --arms drop,placebo \
  --sample-seed-base 0 \
  --cache-seed-from results/fever_p2_llm_e1_qwen3b_gmemory_v2_87_r4/llm_cache.sqlite \
  --output-dir results/cmc_fever_qwen3b_87_r4_seed0
```

This first run intentionally leaves the estimator blocked because within-run
repeats do not replace an independent retest. Run the exact same design with a
disjoint inference-seed range and point it to the first output:

```bash
python -m causal_memory_control.run_fever_audit \
  --test results/fever_difficulty_qwen3b_gmemory_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:3b \
  --claims 87 --repeats 4 \
  --receivers A1 --arms drop,placebo \
  --sample-seed-base 1000 \
  --retest-results results/cmc_fever_qwen3b_87_r4_seed0 \
  --output-dir results/cmc_fever_qwen3b_87_r4_seed1000_retest
```

Keep `--test`, bank files, model, retrieval parameters, claims, repeats,
receivers, arms, and selection seed unchanged between the two runs. The runner
checks a design hash and refuses an invalid comparison. Only
`--sample-seed-base`, `--retest-results`, and `--output-dir` should differ.

`--cache-seed-from` is optional. With the existing `87_r4` cache and seed range
`0..3`, matching USE/PLACEBO requests are reused while new DROP responses are
added only to the copied cache in the new output directory. The old result and
its SQLite file remain unchanged. Omit this option if that server-side cache no
longer exists. Never use the old cache for the independent retest with a new
seed range.

Use all receivers only after the A1 pilot is healthy because it multiplies the
number of intervention events:

```bash
--receivers A1,A2,A3 --arms drop,placebo,global_drop
```

Output files:

- `audit_runs.jsonl`: append-only raw USE/control episode branches.
- `audit_units.json`: per `(claim, receiver, memory)` potential outcomes.
- `arm_evaluations.json`: per-arm oracle, noise, sign-transition, and gate
  results.
- `diagnostics.json`: pivotality, `|O(m)|`, and within/independent sign noise.
- `oracle_evaluation.json`: oracle and matched-budget baseline values.
- `pretraining_gate.json`: the hard decision on whether training is allowed.
- `estimator_evaluation.json`: held-out potential-outcome/controller metrics,
  or the explicit reason training was blocked.
- `run_report.md`: short human-readable summary.

To resume an interrupted invocation, repeat the identical command and append
`--resume`. Completed branches are loaded from `audit_runs.jsonl` and are not
called again.

To reanalyse a completed run whose `--arms` already included `global_drop`,
repeat its exact collection command, add `--primary-arm global_drop`, and add
`--resume`.  Keep `--retest-results` on the independent run.  A successful
analysis-only resume reports `"llm_calls": 0`; the raw collection totals are
retained separately in `diagnostics.json`.
