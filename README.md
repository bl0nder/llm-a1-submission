# Assignment 1: PII Sanitization and Training Data Extraction

This assignment has three parts:

- **Part A:** a LoRA/QLoRA sweep that trains a PII sanitizer on Qwen2.5-1.5B-Instruct.
- **Part B:** evaluates the best sanitizer against regex, few-shot and zero-shot baselines.
- **Part C:** trains Qwen2.5-0.5B victims on a raw corpus and a sanitized corpus, then runs training data extraction attacks against them.

## Setup

Python 3 with a CUDA GPU. Install the dependencies:

```bash
pip install torch transformers peft bitsandbytes pandas numpy tqdm matplotlib
```

`bitsandbytes` is only needed for the 4-bit QLoRA run (A6).

Put the dataset bundle in `data/`. To check that the files are intact, see `data/README.md`.

## Running everything

```bash
bash run_all.sh
```

This runs every system one after another and stops at the first failure. Every script runs on GPU 0 by default. To use another GPU, add `--gpu_id <id>` to the commands.

The CSV files in `results/` get new rows added on every run. Delete them before a clean full run so that no row appears twice.

## Layout

```
src/
  peft_sweep.py      Part A: trains one LoRA/QLoRA adapter, or plots the sweep with --make_fig
  sanitize_eval.py   Part B: runs one sanitizer system on a test file and scores it
  train_victim.py    Part C.1: trains a victim model, building the sanitized corpus if needed
  attack_extract.py  Part C.2: runs the extraction attacks on a victim or the base model
  metrics.py         Shared span recovery, F1, leak rate, perplexity and exposure metrics
data/                Supplied dataset bundle (do not resplit or subsample)
adapters/            Saved LoRA adapters (A1 to A6, victim_raw, victim_sanitized)
results/             Metrics, predictions, CSV tables and figures
run_all.sh           Reproduces every result in order
report.pdf           Report
```

## Part A: Adapter placement sweep

`src/peft_sweep.py` trains on `data/partA_train.jsonl` and validates on `data/partA_validation.jsonl`.

These settings are the same for every run:

| Setting | Value |
|---|---|
| Epochs | 1 |
| Optimizer | AdamW |
| Learning rate | 2e-4 (10% linear warmup, then linear decay) |
| Batch size | 4, with 8 gradient accumulation steps (effective 32) |
| Max sequence length | 1024 |
| LoRA dropout | 0.05 |
| Seed | 42 |

The runs differ in these settings:

| Run | Target modules | r | alpha | Quant |
|---|---|---|---|---|
| A1 | q, v | 8 | 16 | none |
| A2 | q, k, v, o | 8 | 16 | none |
| A3 | q, k, v, o, gate, up, down | 8 | 16 | none |
| A4 | q, k, v, o, gate, up, down | 32 | 64 | none |
| A5 | q, k, v, o, gate, up, down | 32 | 16 | none |
| A6 | q, k, v, o, gate, up, down | 32 | 64 | 4bit (NF4, double quant) |

Each run adds a row to `results/partA_sweep.csv` and saves its adapter to `adapters/<run_id>`. Passing `--make_fig` skips training. It only reads the CSV and writes `results/partA_fig.png`.

A4 is the chosen checkpoint and is used in Parts B and C.

## Part B: Sanitizer evaluation

`src/sanitize_eval.py --system {finetuned, regex, fewshot, zeroshot}` runs one system on a test file. It writes:

- predictions to `results/partB_predictions_<name>.jsonl`
- metrics to `results/partB_metrics_<name>.json`
- one summary row to `results/partB_metrics.csv`

`run_all.sh` scores all four systems on `data/partB_test.jsonl`. It also scores the fine-tuned model on `data/partB_ood_test.jsonl` twice: once as is, and once with `--convert_pii_labels`, which maps the OOD label set onto the training labels when computing metrics. The predictions themselves are unchanged.

## Part C.1: Victim training

`src/train_victim.py` puts LoRA on the attention projections of Qwen2.5-0.5B (r=32, alpha=64, 1 epoch, seed 42).

The corpus filename decides which victim is trained:

- `data/partC_corpus_raw.jsonl` trains `adapters/victim_raw`.
- `data/partC_corpus_sanitized.jsonl` trains `adapters/victim_sanitized`.

If the sanitized corpus doesn't exist, the script builds it first by passing every raw row through the A4 sanitizer (`--sanitizer_adapter`, default `adapters/A4`). A4 must therefore exist before this step runs.

Victim statistics go to `results/partC_victims.csv`.

## Part C.2: Extraction attacks

`src/attack_extract.py --attack all` runs four attacks: discoverable, untargeted, canary and targeted. It uses 5000 samples, seed 42, and the base model as the reference. `run_all.sh` attacks three models:

| Target | Adapter | Report |
|---|---|---|
| Raw victim | `adapters/victim_raw` | `results/partC_attacks_raw.json` |
| Sanitized victim | `adapters/victim_sanitized` | `results/partC_attacks_sanitized.json` |
| Base model | none | `results/partC_attacks_base.json` |

Summary rows go to `results/partC_attacks.csv`, and the canary exposure figure goes to `results/partC_exposure.png`.

## Reproducibility

Every run uses seed 42 and reads the supplied files in their given order. Evaluation decoding is greedy. The scripts log trainable parameters, peak GPU memory (`torch.cuda.max_memory_allocated()`), wall-clock time and throughput.
