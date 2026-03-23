# Multilingual Mental Health — LLM Evaluation

Evaluates LLM classification of depression across Arabic, English, Spanish, and Urdu social media posts.

## Dataset Links

- Chinese: https://drive.google.com/file/d/1fNKtoo4SP98OAhalMjNRZfFqmQZsQ0fh/view
- Spanish: https://www.kaggle.com/datasets/francescoronzano/spanish-tweets-suggesting-depression

## Repository Structure

```
multilingual-mental-health/
├── data/
│   ├── raw/                    # Raw language datasets
│   │   ├── arabic/
│   │   ├── english/
│   │   ├── spanish/
│   │   └── urdu/
│   ├── cleaned/                # Parsed/cached datasets
│   ├── sampled/                # Stratified samples for evaluation
│   └── labeler_progress/       # Silver-labeling checkpoints
├── evaluation/                 # Metrics, parsers, samplers, prompts
├── models/                     # LLM provider wrappers
├── labeler/                    # Silver-label pipeline (Chinese/Spanish)
├── scripts/                    # Runnable pipeline scripts
│   ├── prepare_data.py         # Step 1: parse + sample datasets
│   ├── runner.py               # Step 2: run LLM evaluation
│   ├── smoke_test.py           # Quick API sanity check
│   ├── translate_urdu.py       # Translate Urdu samples to English
│   ├── analyze_urdu_errors.py  # Detailed Urdu error analysis
│   └── verify_labels.py        # Verify silver labels with HF model
├── experiments/
│   └── phase1/                 # Initial exploratory experiment results
│       ├── ANALYSIS_GUIDE.md   # Guide to interpreting phase 1 results
│       └── results/            # All phase 1 JSON result files
├── results/                    # Phase 2+ experiment results (new runs)
├── config.py                   # API key loading
└── requirements.txt
```

## Quickstart

```bash
# 1. Parse and sample datasets
py scripts/prepare_data.py

# 2. Run LLM evaluation (interactive)
py scripts/runner.py

# 3. Smoke test API connections
py scripts/smoke_test.py
```

## Phase 1 Experiments

The `experiments/phase1/` folder contains all initial exploratory work including:
- Full evaluation results across Gemini, DeepSeek, OpenAI, and Claude
- Cross-lingual comparisons across Arabic, English, Spanish, and Urdu
- Urdu native vs. English-translated comparison
- V1 vs. V2 prompt comparison
- Detailed analysis guide: `experiments/phase1/ANALYSIS_GUIDE.md`
