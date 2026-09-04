# Real-LLM FEVER mismatch probe

This package is intentionally separate from `p2_probe/`. It calls a real
OpenAI-compatible chat model, scores exact FEVER labels, and never uses an
LLM judge. The current study focuses on unilateral E1 interventions:
`run_experiment.py` changes only A1's top-1 memory and measures local versus
team effects. E2 remains in the repository but is intentionally not part of
this run. Retrieval is a deterministic, dependency-free BM25 index with fixed
role queries; placebo diagnostics use lexical overlap only to verify
dissimilarity.

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
  --experience-output data/fever/experience_bank.jsonl \
  --distractor-output data/fever/distractor_pool.jsonl \
  --max-experience-items 2000 --max-distractor-items 4000 \
  --seed 42
```

The experience bank is used only for retrieval. The disjoint distractor pool
is used only to construct the selected evidence bundles. Both hashes are
recorded in every E1 result.

Validate all inputs before starting the model:

```bash
python -m p2_probe_llm.validate_inputs \
  --test data/fever/fever_dev_enriched.jsonl \
  --experience-bank data/fever/experience_bank.jsonl \
  --distractor-bank data/fever/distractor_pool.jsonl \
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
  --experience-bank data/fever/experience_bank.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b --top-k 1 \
  --output-dir results/fever_p2_llm_g0
```

Do not continue unless it reports `"pass": true`.

## 5. Calibrate evidence difficulty

Use the same model that will run E1. This tests
`gold_recall = {0, 0.3, 0.5, 0.7, 1.0}` and freezes the condition whose
placebo-team accuracy lies in `[0.62, 0.80]` and is closest to `0.70`:

```bash
python -m p2_probe_llm.calibrate_difficulty \
  --test data/fever/fever_dev_enriched.jsonl \
  --experience-bank data/fever/experience_bank.jsonl \
  --distractor-bank data/fever/distractor_pool.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 100 --repeats 1 --top-k 1 --seed 42 \
  --output-dir results/fever_difficulty_qwen7b
```

Do not continue unless the report contains `"pass": true`. E1 must use the
generated `fever_dev_selected_difficulty.jsonl`.

## 6. Run a real-LLM pilot

E1 local/team mismatch:

```bash
python -m p2_probe_llm.run_experiment \
  --test results/fever_difficulty_qwen7b/fever_dev_selected_difficulty.jsonl \
  --experience-bank data/fever/experience_bank.jsonl \
  --distractor-bank data/fever/distractor_pool.jsonl \
  --endpoint http://127.0.0.1:11434/v1 \
  --model qwen2.5:7b \
  --claims 20 --repeats 8 --top-k 1 --audit-top-n 1 \
  --bootstrap 2000 --seed 42 \
  --output-dir results/fever_p2_llm_e1_pilot
```

Each output directory must be new. The runner refuses to overwrite existing
LLM logs. The SQLite cache is a correctness component: identical prompts in
paired arms are served identically even when the model server itself is not
bitwise deterministic.

Render the E1 scatter after the pilot finishes:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cmi-mpl \
python -m p2_probe_llm.plot_results \
  --e1-dir results/fever_p2_llm_e1_pilot \
  --output-dir results/fever_p2_llm_figures
```

Only after the pilot passes `gate_report.md` should E1 be expanded toward
`--claims 240 --repeats 32`. Increasing `--audit-top-n` multiplies cost and
should be done only after screening or budgeting.

The E1 pilot is roughly 1,900 uncached calls at 20 claims x 8 repeats x one
audited memory. First verify that the top-1 memory changes A1's round-1 or
solo answer. Then expand toward 40 claims x 16 repeats. Do not launch the
full 87-claim run before checking the pilot gates and throughput.
