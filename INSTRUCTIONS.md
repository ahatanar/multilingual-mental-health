# How to Run This Project — Step by Step

A beginner-friendly walkthrough that takes you from a fresh clone all the way through Experiment 4. If you've never touched this repo before, follow the steps in order.

For reference-style docs (prompt list, file formats, how to add a new model), see [README.md](README.md). This file is the runbook.

---

## What You'll Be Doing

The project runs five stages, in order:

| Stage | What it does | Output |
|-------|--------------|--------|
| **Preprocessing** | Builds the 5 000-post evaluation files (2 500 depressed + 2 500 normal per language) | `data/phase2/{arabic,urdu,chinese}_5000samples_seed42.json` |
| **Experiment 1** | Each model classifies every post as Depressed / Not Depressed | `results/phase2/experiment1/` |
| **Experiment 2** | For each Exp 1 prediction, ask the same model which keywords drove its decision | `results/phase2/experiment2/` |
| **Experiment 3** | Re-evaluate Exp 1 labels using English translations — does the model still agree? | `results/phase2/experiment3/` |
| **Experiment 4** | Fresh classification with a written justification, either few-shot or zero-shot | `results/phase2/experiment4*/` |

Every stage is **resumable** — kill it mid-run, restart the same command, and it picks up from the last checkpoint.

---

## Step 0 — One-Time Setup

You only need to do this once per machine.

### 0.1 Install Python dependencies

From the project root:

```bash
pip install -r requirements.txt
```

This installs the SDKs for Gemini, OpenAI, DeepSeek, Claude, plus translation and tokenizer libraries.

### 0.2 Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and paste in API keys for the providers you plan to use. **Keys you won't use can be left blank** — the runner just skips that model.

```env
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
CLAUDE_API_KEY=your-key-here
```

Local models (Llama, Gemma, Qwen, DeepSeek-R1) don't need API keys — see "Running Local Models" below.

### 0.3 (Optional) Start LM Studio for local models

Skip this step if you only plan to use cloud APIs.

1. Install LM Studio.
2. Download the model you want (e.g. `meta-llama-3.1-8b-instruct`).
3. Open the **Local Server** tab and click **Start Server**.
4. Confirm the server is running at `http://localhost:1234`.

The model identifier shown in LM Studio's server tab must match the `default_model` field in [scripts/runner.py](scripts/runner.py#L68) and [scripts/run_exp4.py](scripts/run_exp4.py#L63). Update those if your IDs differ.

---

## Step 1 — Preprocessing (Build the 5 000-Sample Files)

```bash
python scripts/prepare_data.py
```

That's it. The script:

- Reads filtered Arabic + Chinese translation files from [data/phase2/translated/filtered/](data/phase2/translated/filtered/)
- Reads the raw Urdu CSV at [data/raw/urdu/Depression.csv](data/raw/urdu/Depression.csv)
- Stratified-samples **2 500 depressed + 2 500 normal** per language using `seed=42` (deterministic — same output every run)
- Writes [data/phase2/arabic_5000samples_seed42.json](data/phase2/), `urdu_5000samples_seed42.json`, `chinese_5000samples_seed42.json`

### Want just one language?

```bash
python scripts/prepare_data.py --lang arabic
python scripts/prepare_data.py --lang urdu
python scripts/prepare_data.py --lang chinese
```

### Want a smaller eval set for quick testing?

```bash
python scripts/prepare_data.py --n 200   # 100 + 100 per language
```

You should now see three files in `data/phase2/`. **Don't skip this step** — every experiment reads from these files.

---

## Step 2 — Experiment 1: Monolingual Classification

Each selected model reads each post (in its original language) and labels it as `depressed` or `not depressed`.

### Run it (interactive menu)

```bash
python scripts/runner.py
```

You'll be walked through three prompts:

1. **Select Experiment** → enter `1`
2. **Select Model(s)** → enter the numbers, comma-separated (e.g. `1,4` for Gemini + Claude). `8` runs all models.
3. **Select Language(s)** → same idea (`1,2,3` for all three, or `4` for "all available").

The runner then prints the plan, asks you to confirm, and starts.

### What you'll see while it's running

```
[Gemini 2.0 Flash][arabic] Classifying 1/5000...
[Gemini 2.0 Flash][arabic] Classifying 2/5000...
...
  Checkpoint: 10/5000
```

It checkpoints every 10 rows. If it crashes (network drop, killed shell, etc.), just rerun the same command and it resumes from the last checkpoint.

### Useful flags

```bash
python scripts/runner.py --limit 20      # only first 20 samples — quick sanity check
python scripts/runner.py --workers 5     # 5 parallel API requests per model
python scripts/runner.py --delay 2.0     # 2-second pause between requests (slower / safer)
python scripts/runner.py --fresh         # ignore any partial checkpoint and start over
```

### Output

One JSON per `(model, language)` pair, timestamped:

```
results/phase2/experiment1/
├── gemini_arabic_20260601_143022.json
├── gemini_urdu_20260601_143022.json
├── ...
```

Each file contains: `metadata`, `metrics` (accuracy/precision/recall/F1), and `results` (per-post predictions).

### Got `error` or `unclear` entries you want to retry?

```bash
python scripts/rerun_failed.py
# or target a specific file:
python scripts/rerun_failed.py --file results/phase2/experiment1/claude_arabic_20260601_143022.json
```

This re-asks the model only for entries that came back as `error` or `unclear`, and merges them back into a new timestamped file.

---

## Step 3 — Experiment 2: Keyword Attribution

For each Exp 1 prediction, the same model is asked **"which words in the post drove that prediction?"** — producing 1-3 keywords per post (in original language + English translation).

> **You must finish Experiment 1 first.** Exp 2 reads Exp 1's result files.

### Run it

```bash
python scripts/runner.py
```

1. **Select Experiment** → `2`
2. The runner scans `results/phase2/experiment1/` and lists every `(model, language)` pair it finds. Pick the ones you want.
3. Confirm the plan.

### What it does

For each entry in the Exp 1 result file, it sends:
- The original post
- The model's prior Exp 1 prediction

And gets back two lines: the keywords (original-language) and their English translations.

### Output

```
results/phase2/experiment2/
├── claude_arabic_20260602_091500.json
├── ...
```

Each entry now has new fields: `keywords`, `translations`, `raw_response_exp2`.

---

## Step 4 — Experiment 3: Cross-Lingual Consistency

The same posts are translated to English, and the model is asked whether — looking at the English version — it still agrees with the original-language prediction.

> **You must finish Experiment 1 first.** Exp 3 reads Exp 1 results + the English translations baked into the 5k sample files.

### Run it

```bash
python scripts/runner.py
```

1. **Select Experiment** → `3`
2. Pick the `(model, language)` pairs from your Exp 1 results.
3. Confirm.

### What it does

For each Exp 1 entry, the runner:
- Looks up the English translation (from the 5k sample file for Arabic/Chinese, or fallback files for Urdu)
- Sends the English text + the model's original-language prediction
- Asks the model to respond `yes` (agrees) or `no` (disagrees) with its prior label, plus 1-3 keywords

### Output

```
results/phase2/experiment3/
├── gemini_arabic_20260603_104522.json
├── ...
```

Each entry now has: `keywords_exp3`, `agreement` (`yes`/`no`/`no_translation`), `raw_response_exp3`. The metrics block adds `agreement_rate`.

### Retry transient local-model failures

If you ran Exp 3 with local models and some entries failed because LM Studio hiccuped:

```bash
python scripts/rerun_failed_exp3.py --file results/phase2/experiment3/gemma/gemma_chinese_20260603_104522.json
```

---

## Step 5 — Experiment 4: Fresh Classification + Justification

This is the most flexible experiment. The model classifies posts **from scratch** (ignoring any prior predictions) and writes a short English **justification** for each decision.

Two modes:
- **Few-shot** — the prompt includes language-specific examples of depressed/not-depressed posts
- **Zero-shot** — minimal universal prompt, no examples

Two datasets:
- **Full 5 000-sample dataset** (`--full`) — the main result
- **15-row error-analysis CSVs** (default) — the posts every model got wrong in Exp 1; useful for case studies

### Option A — Interactive (via `runner.py`)

```bash
python scripts/runner.py
```

1. **Select Experiment** → `4`
2. Pick models, languages, mode (few-shot / zero-shot), dataset (5k full / 15-row CSVs).
3. Confirm.

### Option B — Direct CLI (more control)

```bash
# Full 5k dataset, few-shot, Claude on Arabic
python scripts/run_exp4.py --model claude --language arabic --full

# Same, but zero-shot
python scripts/run_exp4.py --model claude --language arabic --full --zeroshot

# All models, all languages, full dataset
python scripts/run_exp4.py --models all --languages all --full

# Multiple models, specific languages
python scripts/run_exp4.py --models claude,openai --languages arabic,urdu --full

# Smoke test — run just row 42, print everything, write nothing
python scripts/run_exp4.py --model claude --language arabic --debug 42
```

### Useful flags for `run_exp4.py`

```
--model / --models       Single key or comma-separated. Also: 'online', 'local', 'all'
--language / --languages Same. Also: 'all'
--full                   Use the 5k dataset (otherwise: 15-row error-analysis CSVs)
--zeroshot               Use the universal zero-shot prompt (otherwise: few-shot)
--fresh                  Discard partial results and start over
--limit N                Process only the first N rows
--debug INDEX            Run a single row, print the full LLM response, write nothing
```

### Output

Goes to one of four directories depending on mode + dataset:

| Mode | Dataset | Output dir |
|------|---------|------------|
| Few-shot | 15-row CSVs | `results/phase2/experiment4/` |
| Zero-shot | 15-row CSVs | `results/phase2/experiment4_zeroshot/` |
| Few-shot | Full 5k | `results/phase2/experiment4_full/` |
| Zero-shot | Full 5k | `results/phase2/experiment4_full_zeroshot/` |

Each result entry adds two fields: `exp4_classification` and `exp4_justification`.

### Merge Exp 4 results back into the 15-row CSVs

After the 15-row runs finish, merge each model's predictions/justifications into the error-analysis spreadsheets:

```bash
python scripts/merge_exp4.py             # few-shot → {lang}_all_wrong.csv
python scripts/merge_exp4.py --zeroshot  # zero-shot → {lang}_zeroshot.csv
```

These live in [results/all_models_wrong/](results/all_models_wrong/).

---

## After Everything Runs — Export a Summary

Get a single CSV summarizing Experiment 1 accuracy/precision/recall/F1 across all models × languages:

```bash
python scripts/export_metrics.py
# or pick specific models:
python scripts/export_metrics.py --models gemini,claude,openai --out results/summary.csv
```

---

## Running Local Models

Local models (Llama, Gemma, Qwen, DeepSeek-R1) run through **LM Studio** (or any OpenAI-compatible local server: Ollama, vLLM, llama.cpp).

1. **Start the server** — load the model in LM Studio, click **Start Server** in the Local Server tab.
2. **Check the model ID** — copy the exact identifier shown in the server tab.
3. **Update [scripts/runner.py](scripts/runner.py#L68)** — make sure `default_model` for that key matches. Same for [scripts/run_exp4.py](scripts/run_exp4.py#L63).
4. **If you changed the port** (e.g. running Ollama on `11434`), add a `base_url` field to the model entry:

```python
"llama": {"class": LMStudioProvider, "name": "Llama 3.1 8B (Local)",
          "default_model": "llama3.1:8b",
          "base_url": "http://localhost:11434/v1",
          "max_workers": 1, "delay": 0},
```

5. **Run normally** — `python scripts/runner.py` and pick the local-model number.

Default ports: LM Studio `1234`, Ollama `11434`, vLLM `8000`, llama.cpp `8080`.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `No prepared data for 'arabic'` | You skipped Step 1 | Run `python scripts/prepare_data.py` |
| `No Experiment 1 results found` (when running Exp 2 or 3) | You skipped Exp 1, or it's in the wrong folder | Confirm files exist in `results/phase2/experiment1/` |
| API key error | Missing or wrong key in `.env` | Open `.env`, paste the correct key, save |
| Local model "connection refused" | LM Studio server isn't running | Start LM Studio's Local Server |
| Rate-limit errors mid-run | Calling too fast | Add `--delay 2.0` or reduce `--workers` |
| Run crashes — what now? | Anything | Just rerun the same command. Resume is automatic. Add `--fresh` only if you want to wipe and restart. |
| Want to see what a prompt looks like before running | — | All prompts are defined in [src/evaluation/prompts.py](src/evaluation/prompts.py) |

---

## Recommended Run Order — From a Fresh Clone

```bash
# 0. Setup (once)
pip install -r requirements.txt
cp .env.example .env
# edit .env, paste API keys

# 1. Preprocessing
python scripts/prepare_data.py

# 2. Experiment 1 — sanity check first
python scripts/runner.py --limit 5
# pick Exp 1, one model, one language → confirm it works
# then the real run:
python scripts/runner.py
# pick Exp 1, all models, all languages

# 3. Experiment 2
python scripts/runner.py
# pick Exp 2, then all (model, lang) pairs

# 4. Experiment 3
python scripts/runner.py
# pick Exp 3, then all (model, lang) pairs

# 5. Experiment 4 — full 5k, few-shot
python scripts/run_exp4.py --models all --languages all --full
# then zero-shot
python scripts/run_exp4.py --models all --languages all --full --zeroshot

# 6. Summary
python scripts/export_metrics.py
```
