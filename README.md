# Multilingual Mental Health — LLM Evaluation

Research project evaluating how well large language models detect depression in social media posts across multiple languages. The core question: can LLMs classify mental health signals in non-English, non-Western text as reliably as in English?

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API keys to .env (copy .env.example if present)
#    GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, CLAUDE_API_KEY

# 3. Prepare the 5 000-post evaluation files
python scripts/prepare_data.py

# 4. Run any experiment interactively
python scripts/runner.py
```

---

## How to Run Each Experiment

All experiments are launched through `scripts/runner.py` (interactive menu) or directly via their dedicated scripts. Every run is **resumable** — if a run is interrupted, restart the same command and it picks up where it left off.

### Experiment 1 — Monolingual Classification

```bash
# Interactive (recommended)
python scripts/runner.py
# -> Select [1] Experiment 1
# -> Select model(s) from the menu
# -> Select language(s): Arabic, Urdu, Chinese

# Quick smoke test (first 5 posts only)
python scripts/runner.py --limit 5

# Ignore a previous partial run, start fresh
python scripts/runner.py --fresh

# Slower rate for rate-limited APIs
python scripts/runner.py --delay 2.0
```

Results are saved to `results/phase2/experiment1/<model>_<language>_<timestamp>.json`.

### Experiment 2 — Keyword Attribution

```bash
# Must run Experiment 1 first for the model+language pair you want
python scripts/runner.py
# -> Select [2] Experiment 2
# -> Pick which Experiment 1 result file(s) to explain
```

Results are saved to `results/phase2/experiment2/`.

### Experiment 3 — Cross-Lingual Consistency

```bash
# Must run Experiment 1 first
python scripts/runner.py
# -> Select [3] Experiment 3
# -> Pick which Experiment 1 result file(s) to re-evaluate
```

Results are saved to `results/phase2/experiment3/`.

### Experiment 4 — Fresh Classification + Justification

**Via the interactive runner (recommended):**

```bash
python scripts/runner.py
# -> Select [4] Experiment 4
# -> Select model(s)
# -> Select language(s)
# -> Select mode: [1] Few-shot  or  [2] Zero-shot
# -> Select dataset: [1] Full 5k  or  [2] 15-row error-analysis CSVs
```

**Via the dedicated CLI (more control, scriptable):**

```bash
# Few-shot, all online models, all languages (15-row error-analysis mode)
python scripts/run_exp4.py --models claude,openai,gemini --languages all

# Full 5k dataset, one model at a time (checkpoints every 50 rows)
python scripts/run_exp4.py --model claude --language arabic --full
python scripts/run_exp4.py --model openai --language arabic --full

# Zero-shot, full dataset
python scripts/run_exp4.py --model claude --language arabic --full --zeroshot

# Smoke test: run first 3 rows only, write nothing to disk for 1 row
python scripts/run_exp4.py --model claude --language urdu --limit 3
python scripts/run_exp4.py --model claude --language arabic --debug 42

# Discard existing results and start over
python scripts/run_exp4.py --model claude --language arabic --full --fresh
```

**`run_exp4.py` flag reference:**

| Flag | Description |
|------|-------------|
| `--model <key>` | Single model key (e.g. `claude`, `openai`, `gemini`, `llama`, `gemma`, `qwen`) |
| `--models <keys>` | Comma-separated keys, or `online` / `local` / `all` |
| `--language <lang>` | Single language: `arabic`, `chinese`, `urdu` |
| `--languages <langs>` | Comma-separated, or `all` |
| `--full` | Use the 5 000-sample dataset instead of the 15-row error-analysis CSVs |
| `--zeroshot` | Use the universal zero-shot prompt (saves to `experiment4[_full]_zeroshot/`) |
| `--fresh` | Discard existing results / partial checkpoints and start from scratch |
| `--limit N` | Only process the first N rows (smoke-test) |
| `--debug INDEX` | Run one row by index, print full response, write nothing |

Results are saved to:

| Mode | Directory |
|------|-----------|
| Few-shot, 15-row CSVs | `results/phase2/experiment4/<model>/` |
| Zero-shot, 15-row CSVs | `results/phase2/experiment4_zeroshot/<model>/` |
| Few-shot, full 5k | `results/phase2/experiment4_full/<model>/` |
| Zero-shot, full 5k | `results/phase2/experiment4_full_zeroshot/<model>/` |

**Resumability:** Full 5k runs write a `.partial.json` checkpoint every 50 rows. If the run crashes, restarting the same command resumes automatically from the checkpoint. On successful completion the partial file is deleted and a clean timestamped JSON is written.

---

**`runner.py` flag reference (applies to all experiments):**

| Flag | Effect |
|------|--------|
| `--fresh` | Ignore partial checkpoints, start from scratch |
| `--delay N` | Seconds between API calls (default: 1.0) |
| `--workers N` | Parallel requests per model (default: 1) |
| `--prompt <key>` | Override the language-specific prompt (see Prompt Versions below) |
| `--limit N` | Only process the first N samples (quick sanity check) |

---

### Rerunning Failed Entries

If a run completes but some entries were classified as `error` or `unclear` (e.g. due to a transient API failure), you can rerun only those entries:

```bash
# Experiment 1 / 2 failures
python scripts/rerun_failed.py
# -> interactive file picker, or:
python scripts/rerun_failed.py --file results/phase2/experiment1/claude_arabic_20260501_120000.json
python scripts/rerun_failed.py --file <path> --errors    # errors only
python scripts/rerun_failed.py --file <path> --unclear   # unclear only

# Experiment 3 failures (LM Studio transient errors only — data gaps are left alone)
python scripts/rerun_failed_exp3.py --file results/phase2/experiment3/gemma/gemma_chinese_20260413_120000.json
python scripts/rerun_failed_exp3.py --file <path> --delay 0.5
```

---

### Merging Experiment 4 Results into Error-Analysis CSVs

After running Experiment 4 in few-shot or zero-shot mode, merge the results into the per-language error-analysis CSVs (`results/all_models_wrong/`):

```bash
# Few-shot results → {language}_all_wrong.csv (in place)
python scripts/merge_exp4.py

# Zero-shot results → {language}_zeroshot.csv
python scripts/merge_exp4.py --zeroshot
```

---

### Exporting Experiment 1 Metrics to CSV

```bash
# Writes results/phase2/experiment1/exp1_metrics_summary.csv
python scripts/export_metrics.py

# Custom output path
python scripts/export_metrics.py --out results/exp1_summary.csv

# Filter to specific models
python scripts/export_metrics.py --models gemini,claude,openai
```

---

## Research Experiments

### Experiment 1 — Monolingual Classification

Each LLM is evaluated on 5 000 posts per language, presented in the language's **native script** (no translation). The model must classify each post as **Depressed** or **Not Depressed**. Performance is measured with accuracy, precision, recall, and F1.

**Languages:**

| Language | Dataset | Script | Posts | Balance |
|----------|---------|--------|-------|---------|
| Arabic | CairoDep (Egyptian dialect) | Arabic script | 5 000 | 2 500 dep + 2 500 normal |
| Urdu | Urdu Depression Dataset | Roman Urdu (transliterated) | 5 000 | 2 500 dep + 2 500 normal |
| Chinese | Weibo Depression Dataset | Simplified Chinese | 5 000 | 2 500 dep + 2 500 normal |

**Why native script?** The research goal is to test true multilingual capability — not how well models handle translated content.

**Arabic pre-processing:** Raw CairoDep posts were translated to English (Cohere `command-r-08-2024`) for ethics screening before evaluation. 42 posts were removed (translation failures, explicit sexual content, graphic violence, PII). The final evaluation uses the **original Arabic text**, not the translations.

**Urdu labels:** Four severity levels collapsed to binary: `mild/moderate/severe → depressed`, `non-depression → not depressed`.

**Models evaluated:** Gemini 2.0 Flash, GPT-4o-mini, DeepSeek Chat, Claude Haiku 4.5, Llama 3.1 8B (local), Qwen 3.5 9B (local), Gemma 9B (local)

---

### Experiment 2 — Keyword Attribution

Builds on Experiment 1. Each model is given its **own Experiment 1 prediction** and asked: *"Given that you labelled this post as X, which specific words drove that decision?"*

**Design rationale:** Combining classification and keyword extraction in one prompt risks post-hoc rationalization. By separating the steps, Experiment 2 captures true attribution.

**Output format — 2 lines per post:**
```
الحزن, الوحدة, الخوف
sadness, loneliness, fear
```

Line 1: keywords in the original language. Line 2: one-word English translation of each, same order.

---

### Experiment 3 — Cross-Lingual Label Consistency

Builds on Experiment 1. Each model is given the **English translation** of a post it already classified and asked: *does the English translation still support your earlier classification?*

**Design rationale:** A classification driven by genuine clinical content should hold regardless of language. Label flips when shown the translation indicate the original decision was influenced by language-specific surface cues.

**Output format — 2 lines per post:**
```
hopelessness, alone
yes
```

Line 1: 1-2 key English words from the translation. Line 2: `yes` (model still agrees) or `no` (model disagrees).

---

### Experiment 4 — Fresh Classification + Justification

Each model reclassifies posts from scratch using a structured prompt, producing both a binary classification and a written justification. Two modes:

- **Few-shot:** language-specific prompts with in-context examples
- **Zero-shot:** universal minimal prompt with no examples

Two dataset sizes:
- **15-row error-analysis mode:** posts where all models were wrong in Experiments 1–3 (fast iteration / qualitative analysis)
- **Full 5k mode:** the complete 5 000-sample evaluation dataset (comparable to Experiment 1)

**Output per entry:**
```json
{
  "index": 42,
  "exp4_classification": "depressed",
  "exp4_justification": "The post describes persistent hopelessness and inability to function...",
  "ground_truth": "depressed"
}
```

---

## Dataset Sources

| Language | Dataset | Source |
|----------|---------|--------|
| Arabic | CairoDep | Egyptian Arabic social media (Twitter, Reddit, Facebook, crowdsourcing) |
| Urdu | Urdu Depression Dataset | Roman Urdu social media, academically curated |
| Chinese | Weibo Depression Dataset | Weibo (Chinese microblogging), Simplified Chinese |

---

## Prompt Versions

| Key | Description | Default for |
|-----|-------------|-------------|
| `v1` | Zero-shot, minimal instructions | — |
| `v2` | Enhanced clinical framework, handles sarcasm + edge cases | — |
| `v3` | Few-shot, 6 Roman Urdu examples, political/poetry false-positive guards | Urdu Exp 1 |
| `v3_arabic` | Few-shot, 6 Arabic-script examples, religious/hashtag false-positive guards | Arabic Exp 1 |
| `v3_chinese` | Few-shot, 6 Chinese Weibo examples, illness/fandom/lifestyle false-positive guards | Chinese Exp 1 |
| `v3_exp2` | Urdu attribution: given a label, identify the words that drove it | Urdu Exp 2 |
| `v3_arabic_exp2` | Arabic attribution | Arabic Exp 2 |
| `v3_chinese_exp2` | Chinese attribution | Chinese Exp 2 |
| `v3_exp3` | Cross-lingual consistency: English translation + Exp 1 label → keywords + yes/no | All languages Exp 3 |

---

## Repository Structure

```
multilingual-mental-health/
│
├── data/
│   ├── raw/                         # Original unmodified datasets
│   │   ├── arabic/CairoDep_Datasets.csv
│   │   └── urdu/Depression.csv
│   │
│   ├── cleaned/                     # Parsed + normalised full datasets (cached)
│   │   ├── arabic.json
│   │   └── urdu.json
│   │
│   ├── phase1/                      # Phase 1 (exploratory) data
│   │   └── sampled/                 # 500-post stratified samples
│   │
│   └── phase2/                      # Phase 2 evaluation data
│       ├── arabic_5000samples_seed42.json       # EXPERIMENT INPUT — 5k Arabic posts
│       ├── urdu_5000samples_seed42.json         # EXPERIMENT INPUT — 5k Urdu posts
│       ├── chinese_5000samples_seed42.json      # EXPERIMENT INPUT — 5k Chinese posts
│       ├── translated/              # Cohere translations (ethics review + Exp 3)
│       └── translation_progress/   # Translation checkpoints
│
├── evaluation/                      # Shared evaluation library
│   ├── prompts.py                   # All prompt versions (V1/V2/V3 + Exp 2/3/4 variants)
│   ├── parsers.py                   # Dataset parsers (one class per language)
│   ├── metrics.py                   # EvaluationMetrics: accuracy, precision, recall, F1
│   └── sampler.py                   # Stratified sampling utility
│
├── models/                          # LLM provider wrappers
│   ├── base.py                      # Abstract ModelProvider base class
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   ├── deepseek_provider.py
│   ├── claude_provider.py
│   └── lm_studio_provider.py        # Local models via LM Studio (Llama, Gemma, Qwen)
│
├── scripts/                         # All active research scripts
│   ├── runner.py                    # Main interactive runner — Experiments 1/2/3/4
│   ├── run_exp4.py                  # Experiment 4 standalone CLI (more control than runner.py)
│   ├── prepare_data.py              # Prepare 5 000-post eval files for Arabic, Urdu, Chinese
│   ├── merge_exp4.py                # Merge Exp 4 results into error-analysis CSVs
│   ├── export_metrics.py            # Export Exp 1 metrics to CSV
│   ├── rerun_failed.py              # Rerun error/unclear entries from an Exp 1 result file
│   └── rerun_failed_exp3.py         # Rerun transient-error entries from an Exp 3 result file
│
├── archive/                         # Completed one-time scripts (do not modify)
│   ├── phase1/                      # Phase 1 exploratory scripts
│   └── phase2/                      # Phase 2 data-prep scripts (Arabic translation pipeline, etc.)
│
├── results/
│   ├── all_models_wrong/            # 15-row error-analysis CSVs (Exp 4 input)
│   │   ├── arabic_all_wrong.csv
│   │   ├── chinese_all_wrong.csv
│   │   ├── urdu_all_wrong.csv
│   │   ├── arabic_zeroshot.csv      # After merge_exp4.py --zeroshot
│   │   ├── chinese_zeroshot.csv
│   │   └── urdu_zeroshot.csv
│   │
│   └── phase2/
│       ├── experiment1/             # <model>_<language>_<timestamp>.json
│       ├── experiment2/
│       ├── experiment3/
│       ├── experiment4/             # Few-shot 15-row results, per model subfolder
│       ├── experiment4_zeroshot/
│       ├── experiment4_full/        # Few-shot 5k results
│       └── experiment4_full_zeroshot/
│
├── visualization/                   # Analysis plots and word clouds
│   ├── main.py
│   ├── discovery.py
│   ├── generate_keyword_analysis.py
│   ├── generate_wordclouds.py
│   └── outputs/plots/
│
├── config.py                        # API key loader (reads from .env)
├── requirements.txt
└── .env                             # API keys (not committed)
```

---

## Local Models (LM Studio)

Llama, Gemma, and Qwen run locally via [LM Studio](https://lmstudio.ai/). To use them:

1. Download the model in LM Studio and start the local server (default: `localhost:1234`).
2. Copy the model identifier from LM Studio's **Local Server** tab.
3. Update the `"default_model"` value for the relevant key in the `MODELS` dict in `scripts/runner.py` or `scripts/run_exp4.py`.
4. Run as usual — no API key required.

---

## Arabic Translation Pipeline (Ethics Pre-processing)

The Arabic evaluation required a content screening step before the dataset could be used in research.

```
CairoDep_Datasets.csv (7 000 posts)
    ↓  archive/phase2/prepare_arabic.py
arabic_6000samples_seed42.json  (stratified 6 000, seed=42)
    ↓  archive/phase2/translate_arabic.py  (Cohere command-r-08-2024, ~1.7h)
arabic_6000samples_seed42_translated.json  (6 000 English translations)
    ↓  Manual content review (42 posts removed)
translated/filtered/arabic_6000samples_seed42_filtered.json  (5 958 posts)
    ↓  scripts/prepare_data.py
arabic_5000samples_seed42.json  (5 000 posts, original Arabic script, eval-ready)
```

Posts removed: 18 translation failures, 20 explicit sexual content, 2 graphic violence, 2 PII/spam.

---

## Phase 1 Archive

All Phase 1 exploratory work (500-sample pilots across Arabic, English, Spanish, Urdu) is archived in `archive/phase1/`. Phase 1 scripts are preserved for reproducibility and should not be modified.
