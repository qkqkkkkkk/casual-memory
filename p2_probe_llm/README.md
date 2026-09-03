# Real-LLM FEVER mismatch probe

This package is intentionally separate from `p2_probe/`. It calls a real
OpenAI-compatible chat model, scores exact FEVER labels, and never uses an
LLM judge. `run_experiment.py` implements unilateral E1 interventions;
`run_diversity.py` implements memory-all versus placebo-all E2. Retrieval is
now a deterministic, dependency-free BM25 index with fixed role queries;
placebo diagnostics still use lexical overlap only to verify dissimilarity.

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

## 3. Build the frozen memory bank

```bash
python -m p2_probe_llm.build_memory_bank \
  --input data/fever/fever_train_enriched.jsonl \
  --exclude-claims-from data/fever/fever_dev_enriched.jsonl \
  --output data/fever/memory_bank.jsonl \
  --max-items 2000 --seed 42
```

This produces a balanced SUPPORTS/REFUTES bank and prints its MD5. The runner
records the same MD5 in every result.

Validate all inputs before starting the model:

```bash
python -m p2_probe_llm.validate_inputs \
  --test data/fever/fever_dev_enriched.jsonl \
  --memory-bank data/fever/memory_bank.jsonl \
  --sample-claims 100 --top-k 1
```

Do not spend LLM calls unless this command reports `"pass": true`.

## 4. Start a model server

The probe itself uses only the Python standard library. Create an isolated
environment and install matplotlib only when figures are needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r p2_probe_llm/requirements.txt
```

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
  --memory-bank data/fever/memory_bank.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b --top-k 1 \
  --output-dir results/fever_p2_llm_g0
```

Do not continue unless it reports `"pass": true`.

## 5. Calibrate evidence difficulty

Use the same model that will run E1/E2. This tests
`gold_recall = {0, 0.3, 0.5, 0.7, 1.0}` and freezes the condition whose
placebo-team accuracy lies in `[0.62, 0.80]` and is closest to `0.70`:

```bash
python -m p2_probe_llm.calibrate_difficulty \
  --test data/fever/fever_dev_enriched.jsonl \
  --memory-bank data/fever/memory_bank.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 100 --repeats 1 --top-k 1 --seed 42 \
  --output-dir results/fever_difficulty_qwen7b
```

Do not continue unless the report contains `"pass": true`. E1 and E2 must
both use the generated `fever_dev_selected_difficulty.jsonl`; changing the
evidence policy between them invalidates the mechanism comparison.

## 6. Run a real-LLM pilot

E1 local/team mismatch:

```bash
python -m p2_probe_llm.run_experiment \
  --test results/fever_difficulty_qwen7b/fever_dev_selected_difficulty.jsonl \
  --memory-bank data/fever/memory_bank.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 20 --repeats 5 --top-k 1 --audit-top-n 1 \
  --bootstrap 2000 --seed 42 \
  --output-dir results/fever_p2_llm_e1_pilot
```

E2 diversity mechanism:

```bash
python -m p2_probe_llm.run_diversity \
  --test results/fever_difficulty_qwen7b/fever_dev_selected_difficulty.jsonl \
  --memory-bank data/fever/memory_bank.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 20 --repeats 5 --top-k 1 --bootstrap 2000 --seed 42 \
  --output-dir results/fever_p2_llm_e2_pilot
```

Each output directory must be new. The runner refuses to overwrite existing
LLM logs. The SQLite cache is a correctness component: identical prompts in
paired arms are served identically even when the model server itself is not
bitwise deterministic.

Render the two paper-style figures after both experiments finish:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cmi-mpl \
python -m p2_probe_llm.plot_results \
  --e1-dir results/fever_p2_llm_e1_pilot \
  --e2-dir results/fever_p2_llm_e2_pilot \
  --output-dir results/fever_p2_llm_figures
```

Only after the pilot passes `gate_report.md` should E1 be expanded toward
`--claims 240 --repeats 32`. Increasing `--audit-top-n` multiplies cost and
should be done only after screening or budgeting.

The E1 pilot is roughly 1,200 uncached calls at 20 claims x 5 repeats x one
audited memory. A direct 240 x 32 run can approach 90,000 calls, so do not
launch it before checking the pilot gates and throughput.
