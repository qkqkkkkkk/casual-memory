# Real-LLM FEVER mismatch probe

This package is intentionally separate from `p2_probe/`. It calls a real
OpenAI-compatible chat model, scores exact FEVER labels, and never uses an
LLM judge. The current study focuses on unilateral E1 interventions. Its
memory layer follows G-Memory's frozen precedent pattern: each historical
`task_main` is embedded with `sentence-transformers/all-MiniLM-L6-v2`, candidates
are ranked by cosine similarity, and the same claim-level candidates are
available to all roles. FEVER labels remain metadata rather than a retrieval
filter. E1 changes only A1's top-1 memory and measures local versus team
effects. E2 remains in the repository but is intentionally not part of this
run.

This is a controlled G-Memory adaptation, not a claim that the full G-Memory
graph has been copied. It reuses successful reference-case records,
`task_main` cosine retrieval, the `0.3` similarity threshold, and the
reference-case-first prompt. It intentionally omits query/insight graphs,
continual writes, and success/failure retrieval buckets. Those components
would change memory during the experiment, while FEVER's `SUPPORTS` and
`REFUTES` labels are answers rather than G-Memory success/failure outcomes.

## 1. Obtain official FEVER resources

The existing `data/fever/fever_dev.jsonl` contains Wikipedia page and sentence
IDs, not evidence text. Download the official FEVER train split and
`wiki-pages` dump on a machine with enough disk space. The canonical FEVER
download page is <https://fever.ai/dataset/fever.html>. Common direct URLs are:

```bash
cd /mnt/mydata/hjq/casual-memory/data/fever
wget -c https://fever.ai/download/fever/train.jsonl -O fever_train.jsonl
wget -c https://fever.ai/download/fever/wiki-pages.zip -O wiki-pages.zip
unzip wiki-pages.zip
```

After extraction, locate the directory containing files such as
`wiki-001.jsonl`:

```bash
find . -name 'wiki-*.jsonl' | head
```

The wiki dump is large and is ignored by Git. Do not push it to GitHub.

## 2. Resolve evidence IDs

The resolver accepts one JSONL file, a directory of JSONL shards, or gzipped
JSONL shards. It scans the dump but retains only pages referenced by the split.

```bash
python -m p2_probe_llm.prepare_evidence \
  --input data/fever/fever_dev.jsonl \
  --wiki data/fever/wiki-pages \
  --output data/fever/fever_dev_enriched.jsonl

python -m p2_probe_llm.prepare_evidence \
  --input data/fever/fever_train.jsonl \
  --wiki data/fever/wiki-pages \
  --output data/fever/fever_train_enriched.jsonl
```

The command reports `binary_with_evidence`. Do not continue unless its ratio
to `binary` is high; a low ratio usually means `--wiki` points to the wrong
directory or the dump format does not match.

## 3. Build disjoint experience and distractor pools

```bash
python -m p2_probe_llm.build_pools \
  --input data/fever/fever_train_enriched.jsonl \
  --exclude-claims-from data/fever/fever_dev_enriched.jsonl \
  --experience-output data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-output data/fever/distractor_pool_gmemory_v2.jsonl \
  --max-experience-items 2000 --max-distractor-items 4000 \
  --seed 42
```

The command writes schema `gmemory-fever-v2`. Older E1 pools must not be
reused. The experience bank is used only for retrieval. The disjoint
distractor pool is used only to construct the selected evidence bundles. Both hashes are
recorded in every E1 result.

Create an isolated environment and install the plotting and semantic-retrieval
dependencies before running validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r p2_probe_llm/requirements.txt
```

Validate all inputs before starting the model:

```bash
python -m p2_probe_llm.validate_inputs \
  --test data/fever/fever_dev_enriched.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --sample-claims 100 --top-k 1 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-threshold 0.3 \
  --output results/e1_input_validation_gmemory_v2.json
```

Do not spend LLM calls unless this command reports `"pass": true`. The gate
allows up to 5% of sampled claims to have no eligible memory above the frozen
`0.3` threshold. Such claims are excluded and reported rather than assigned a
lower-quality memory.

## 4. Start a model server

The semantic retriever downloads the same embedding model used by G-Memory on
its first run. On a cluster without direct Hugging Face access, set
`HF_ENDPOINT` to the institution's mirror before validation.

Ollama example:

```bash
ollama serve
ollama pull qwen2.5:7b
curl http://127.0.0.1:11434/v1/models
```

For vLLM, start its OpenAI-compatible server and use its `/v1` endpoint. If
the server requires authentication, pass `--api-key` or set
`OPENAI_API_KEY`. The primary paper configuration should use Qwen2.5-14B;
7B is suitable for the pipeline pilot.

Example vLLM command:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5:14b \
  --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --enable-prefix-caching
```

Run the G0 cache gate before calibration or experiments:

```bash
python -m p2_probe_llm.gate_determinism \
  --test data/fever/fever_dev_enriched.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b --top-k 1 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-threshold 0.3 \
  --output-dir results/fever_p2_llm_g0_gmemory_v2
```

Do not continue unless it reports `"pass": true`.

## 5. Calibrate evidence difficulty

Use the same model that will run E1. This tests
`gold_recall = {0, 0.3, 0.5, 0.7, 1.0}` and freezes the condition whose
placebo-team accuracy lies in `[0.62, 0.80]` and is closest to `0.70`:

```bash
python -m p2_probe_llm.calibrate_difficulty \
  --test data/fever/fever_dev_enriched.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 100 --repeats 1 --top-k 1 --seed 42 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-threshold 0.3 \
  --output-dir results/fever_difficulty_qwen7b_gmemory_v2
```

Do not continue unless the report contains `"pass": true`. E1 must use the
generated `fever_dev_selected_difficulty.jsonl`.

## 6. Run a real-LLM pilot

E1 local/team mismatch:

```bash
python -m p2_probe_llm.run_experiment \
  --test results/fever_difficulty_qwen7b_gmemory_v2/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --distractor-bank data/fever/distractor_pool_gmemory_v2.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 20 --repeats 8 --top-k 1 --audit-top-n 1 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-threshold 0.3 \
  --bootstrap 2000 --seed 42 \
  --output-dir results/fever_p2_llm_e1_gmemory_v2_pilot
```

Each output directory must be new. The runner refuses to overwrite existing
LLM logs. The SQLite cache is a correctness component: identical prompts in
paired arms are served identically even when the model server itself is not
bitwise deterministic.

Render the E1 scatter as a PNG after the pilot finishes:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cmi-mpl \
python -m p2_probe_llm.plot_results \
  --e1-dir results/fever_p2_llm_e1_gmemory_v2_pilot \
  --output-dir results/fever_p2_llm_e1_gmemory_v2_figures
```

The output file is `fig1_llm_scatter.png`.

To diagnose a completed E1 run without making new model calls:

```bash
python -m p2_probe_llm.diagnose_e1 \
  --results-dir results/fever_p2_llm_e1_gmemory_v2_pilot \
  --experience-bank data/fever/experience_bank_gmemory_v2.jsonl \
  --test results/fever_difficulty_qwen7b_gmemory_v2/fever_dev_selected_difficulty.jsonl \
  --limit 20
```

The important fields are `a1_memory_id_changed`, `a1_round1_verdict_changed`,
`a1_solo_verdict_changed`, and the `influence_*` counts. If the first is zero,
the intervention is not being applied. If only the first is nonzero, the model
received different memories but did not change its binary answer. Also inspect
`retrieval_score_summary` and `a1_round1_memory_use` in `mismatch_rate.json`;
they distinguish weak retrieval from a model that explicitly rejects memory.

Only after the pilot passes `gate_report.md` should E1 be expanded toward
`--claims 240 --repeats 32`. Increasing `--audit-top-n` multiplies cost and
should be done only after screening or budgeting.

The E1 pilot is roughly 1,900 uncached calls at 20 claims x 8 repeats x one
audited memory. First verify that the top-1 memory changes A1's round-1 or
solo answer. Then expand toward 40 claims x 16 repeats. Do not launch the
full 87-claim run before checking the pilot gates and throughput.
