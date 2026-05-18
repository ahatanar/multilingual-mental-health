# Data Visualization Pipeline

Analysis and visualization pipeline for the multilingual mental health NLP project.
Crawls `results/phase2/`, normalizes all result JSON files, joins experiments,
and produces CSV tables + PNG plots organized by experiment and comparison type.

---

## Quick start

```bash
# From the project root
cd "c:/Users/abdul/NLP Project/multilingual-mental-health"
pip install pandas matplotlib seaborn numpy
python data_visualization/main.py
```

All outputs go to `data_visualization/outputs/`.

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--results PATH` | `results/` | Root of the results directory tree |
| `--output PATH` | `data_visualization/outputs/` | Where to write CSVs and plots |
| `--no-plots` | off | Skip plot generation (tables only) |
| `--no-tables` | off | Skip CSV saving (plots only) |

---

## Module overview

| File | Role |
|------|------|
| `discovery.py` | Crawls `results/phase2/`, infers model/language/experiment from path + filename |
| `loader.py` | Reads each JSON, flattens metrics, normalizes result rows into DataFrames |
| `merger.py` | Joins experiments by (model, language, index); builds keyword + agreement summaries |
| `plots.py` | All visualization functions; skips gracefully when data is missing |
| `main.py` | Orchestration: discover → load → merge → tables → plots → report |

---

## Output structure

```
data_visualization/outputs/
├── tables/
│   ├── master_summary.csv              # One row per (exp, model, language): all metrics
│   ├── samples_exp1.csv                # All Exp 1 result rows (flat)
│   ├── samples_exp2.csv                # All Exp 2 result rows (flat, includes keywords)
│   ├── samples_exp3.csv                # All Exp 3 result rows (flat, includes agreement)
│   ├── merged_exp1_exp2.csv            # Inner join Exp1 + Exp2 by index
│   ├── merged_exp2_exp3.csv            # Inner join Exp2 + Exp3 by index
│   ├── merged_exp1_exp3.csv            # Inner join Exp1 + Exp3 by index
│   ├── merged_all_three.csv            # Inner join all 3 experiments by index
│   ├── case_level_merged_all_three.csv # Same as above — useful for per-sample analysis
│   ├── keyword_summary.csv             # Exp2: per (model, lang) keyword stats
│   ├── top_keywords.csv                # Exp2: 50 most frequent translated keywords globally
│   ├── exp3_agreement_summary.csv      # Exp3: agreement/disagree/flip counts per (model, lang)
│   └── exp1_model_ranking_by_language.csv  # Models ranked by F1 per language
│
└── plots/
    ├── exp1/
    │   ├── metrics_by_model.png         # Avg metrics grouped by model (all langs avg)
    │   ├── metrics_by_language.png      # Avg metrics grouped by language (all models avg)
    │   ├── online_vs_local.png          # Online vs local side-by-side per metric
    │   ├── f1_heatmap.png               ★ F1 score heatmap: model × language
    │   ├── accuracy_heatmap.png         # Accuracy heatmap: model × language
    │   ├── model_ranking.png            ★ Models ranked by avg F1 (horizontal bar)
    │   ├── per_language_arabic.png      # All models on Arabic
    │   ├── per_language_urdu.png        # All models on Urdu
    │   ├── per_language_chinese.png     # All models on Chinese
    │   └── confusion_summary.png        # Stacked TP/FP/TN/FN proportions per model
    │
    ├── exp2/
    │   ├── keyword_count_distribution.png  # Box plots of keyword count per model+lang
    │   ├── keyword_stats_heatmap.png       # Avg keywords per entry: model × language
    │   ├── top_keywords_global.png         # Top 25 translated keywords globally
    │   ├── top_keywords_by_class.png       ★ Top keywords: depressed vs not-depressed
    │   └── keywords_correct_vs_incorrect.png  # Keyword count: correct vs wrong predictions
    │
    ├── exp3/
    │   ├── agreement_rates.png          ★ Agreement rate per model and language
    │   ├── agreement_breakdown.png      # Stacked agree/disagree/no_translation counts
    │   └── agreement_heatmap.png        ★ Agreement rate heatmap: model × language
    │
    └── cross/
        ├── f1_vs_agreement.png          ★ Scatter: Exp1 F1 vs Exp3 agreement rate
        ├── exp1_vs_exp2_accuracy.png    # Verify Exp1 base accuracy == Exp2 base accuracy
        ├── exp1_incorrect_flip_in_exp3.png  # Exp1 wrong predictions: flip rate in Exp3
        └── keyword_count_vs_agreement.png   # Keyword count (Exp2) vs agreement (Exp3)
```

★ = especially useful for the paper / report

---

## Data schema reference

### Experiment 1 result entries
| Field | Description |
|-------|-------------|
| `index` | Sample index (aligns across experiments within same model+language) |
| `ground_truth` | `"depressed"` or `"not depressed"` |
| `prediction` | Model prediction |
| `correct` | `True/False` (derived) |
| `word_count` | Post word count |
| `error` | Error message if classification failed |

### Experiment 2 adds
| Field | Description |
|-------|-------------|
| `keywords` | List of native-script keywords extracted |
| `translations` | English translations of those keywords |
| `keyword_count` | Length of keywords list |

### Experiment 3 adds
| Field | Description |
|-------|-------------|
| `translation` | Full English translation of the post |
| `original_language` | Language/dialect label |
| `keywords_exp3` | Keywords extracted from translated text |
| `agreement` | `"yes"`, `"no"`, or `"no_translation"` |

---

## What merges are possible

| Merge | Models/Languages | Notes |
|-------|-----------------|-------|
| Exp1 + Exp2 | All (6 models × 3 langs) | Full keyword-level analysis |
| Exp1 + Exp3 | Gemini + OpenAI × 3 langs | Agreement vs accuracy analysis |
| Exp2 + Exp3 | Gemini + OpenAI × 3 langs | Keyword behavior vs translation consistency |
| All three | Gemini + OpenAI × 3 langs | Full per-sample case analysis |

---

## Re-running after new results are added

Just re-run `python data_visualization/main.py`. The discovery step will pick up
any new files automatically. Outputs are overwritten each run.

If only one experiment has new files:
```bash
python data_visualization/main.py --no-plots   # fast: tables only
```
