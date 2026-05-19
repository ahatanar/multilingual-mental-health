# Multilingual Mental Health — LLM Evaluation Framework

Evaluates LLM depression detection across Arabic, Urdu, and Chinese social media posts. Four experiments: monolingual classification, keyword attribution, cross-lingual consistency, and fresh classification with justification. Supports both cloud APIs and local models via LM Studio.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
#ignore unless specified tor un
python scripts/prepare_data.py   # creates the 5 000-post eval files
```

Open `.env` and fill in the keys for the models you want to use:

```
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
CLAUDE_API_KEY=your-claude-key-here
```

Keys for models you don't use can be left blank. Local models (Llama, Gemma, Qwen) run via LM Studio and need no API key — see the Local Models section below.

---

## Running Experiments

### Interactive runner (recommended)

```bash
python scripts/runner.py
```

Presents a menu: pick an experiment, pick models, pick languages. Handles all four experiments. Every run is **resumable** — restart the same command after a crash and it picks up from the last checkpoint.

```
Flags:
  --fresh       Discard partial checkpoints, start from scratch
  --delay N     Seconds between API calls (default: 1.0)
  --workers N   Parallel requests per model (default: 1)
  --limit N     Only process first N samples (sanity check)
  --prompt KEY  Override the default prompt (see Prompt Versions below)
```

### Experiment 4 — dedicated CLI

Experiment 4 (fresh classification + justification) has its own scriptable CLI:

```bash
# Few-shot, specific models and languages
python scripts/run_exp4.py --models claude,openai --languages arabic,urdu

# Full 5k dataset (checkpoints every 50 rows, auto-resumes on restart)
python scripts/run_exp4.py --model claude --language arabic --full

# Zero-shot mode
python scripts/run_exp4.py --model claude --language arabic --full --zeroshot

# Smoke test one row
python scripts/run_exp4.py --model claude --language arabic --debug 42

# All models, all languages
python scripts/run_exp4.py --models all --languages all --full
```

```
Flags:
  --model / --models      Single key or comma-separated; also: online, local, all
  --language / --languages
  --full                  Use 5k dataset instead of 15-row error-analysis CSVs
  --zeroshot              Use universal zero-shot prompt
  --fresh                 Discard existing results and start over
  --limit N               Process first N rows only
  --debug INDEX           Run one row, print full response, write nothing
```

---

## Utility Scripts

```bash
# Rerun entries that came back as error/unclear (Exp 1)
python scripts/rerun_failed.py
python scripts/rerun_failed.py --file results/phase2/experiment1/claude_arabic_<ts>.json

# Rerun transient LM Studio failures from an Exp 3 file
python scripts/rerun_failed_exp3.py --file results/phase2/experiment3/gemma/gemma_chinese_<ts>.json

# Merge Exp 4 classifications into the error-analysis CSVs
python scripts/merge_exp4.py             # few-shot → {lang}_all_wrong.csv
python scripts/merge_exp4.py --zeroshot  # zero-shot → {lang}_zeroshot.csv

# Export Exp 1 metrics to a summary CSV
python scripts/export_metrics.py
python scripts/export_metrics.py --out results/summary.csv --models gemini,claude,openai
```

---

## Models

| Key | Model | Type |
|-----|-------|------|
| `gemini` | Gemini 2.0 Flash | API |
| `openai` | GPT-4o-mini | API |
| `deepseek` | DeepSeek Chat | API |
| `claude` | Claude Haiku 4.5 | API |
| `llama` | Llama 3.3 8B | Local (LM Studio) |
| `gemma` | Gemma 4 E2B | Local (LM Studio) |
| `qwen` | Qwen 3.5 9B | Local (LM Studio) |

**Adding a model:** Create a provider class in `src/models/`, add the API key mapping to `src/config.py` and `.env`, then add an entry to the `MODELS` dict in `scripts/runner.py` and `scripts/run_exp4.py`.

**Local models:** Any OpenAI-compatible local inference server works — LM Studio, Ollama, vLLM, llama.cpp, etc. Start the server, then update `"default_model"` for the relevant key in the `MODELS` dict. If the server runs on a different port, add a `"base_url"` field to the entry:

```python
# in scripts/runner.py and scripts/run_exp4.py MODELS dict:
"llama": {"class": LMStudioProvider, "name": "Llama 3.3 8B (Local)",
          "default_model": "llama3.3:8b",          # model ID as the server reports it
          "base_url": "http://localhost:11434/v1",  # Ollama default; omit for LM Studio
          "max_workers": 1, "delay": 0},
```

Default ports: LM Studio → `1234`, Ollama → `11434`, vLLM → `8000`, llama.cpp → `8080`.

---

## Prompt Versions

| Key | Used for |
|-----|----------|
| `v3_arabic` | Exp 1 — Arabic (default) |
| `v3` | Exp 1 — Urdu (default) |
| `v3_chinese` | Exp 1 — Chinese (default) |
| `v3_arabic_exp2` | Exp 2 — Arabic attribution (default) |
| `v3_exp2` | Exp 2 — Urdu attribution (default) |
| `v3_chinese_exp2` | Exp 2 — Chinese attribution (default) |
| `v3_exp3` | Exp 3 — cross-lingual consistency, all languages (default) |
| `v1`, `v2` | Older prompt versions, available for comparison |

All prompts are defined in `evaluation/prompts.py`.

---

## Repository Structure

```
├── scripts/
│   ├── runner.py               # Main interactive runner — all 4 experiments
│   ├── run_exp4.py             # Experiment 4 standalone CLI
│   ├── prepare_data.py         # Build 5k eval files from raw datasets
│   ├── merge_exp4.py           # Merge Exp 4 results into error-analysis CSVs
│   ├── export_metrics.py       # Export Exp 1 metrics to CSV
│   ├── rerun_failed.py         # Retry error/unclear entries (Exp 1)
│   └── rerun_failed_exp3.py    # Retry transient failures (Exp 3)
│
├── src/                         # Library code (imported by scripts)
│   ├── evaluation/
│   │   ├── prompts.py          # All prompt definitions
│   │   ├── metrics.py          # Accuracy, precision, recall, F1
│   │   └── parsers.py          # Dataset parsers per language
│   ├── models/                 # One provider class per LLM
│   └── config.py               # API key loader (reads from .env)
│
├── data/phase2/                # 5k-post eval files (created by prepare_data.py)
│
├── results/
│   ├── all_models_wrong/       # 15-row error-analysis CSVs (Exp 4 input)
│   └── phase2/
│       ├── experiment1/
│       ├── experiment2/
│       ├── experiment3/
│       ├── experiment4/                  # 15-row few-shot results
│       ├── experiment4_zeroshot/
│       ├── experiment4_full/             # 5k few-shot results
│       └── experiment4_full_zeroshot/
│
├── visualization/              # Plot generation scripts
├── archive/                    # Completed one-time scripts (translation pipeline, etc.)
└── .env                        # API keys (not committed)
```

---

## Result File Format

Each experiment writes a timestamped JSON to its results directory:

```json
{
  "metadata": { "model": "claude", "language": "arabic", "experiment": 1, "timestamp": "..." },
  "metrics":  { "accuracy": 0.81, "precision": 0.80, "recall": 0.83, "f1_score": 0.81 },
  "results":  [ { "index": 0, "post_full": "...", "ground_truth": "depressed", "prediction": "depressed" }, ... ]
}
```

Experiment 4 adds `exp4_classification` and `exp4_justification` fields per entry.
