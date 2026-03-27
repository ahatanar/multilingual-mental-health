# Multilingual Mental Health — LLM Evaluation

Research project evaluating how well large language models detect depression in social media posts across multiple languages. The core question: can LLMs classify mental health signals in non-English, non-Western text as reliably as in English?

---

## Research Experiments

### Experiment 1 — Monolingual Classification *(active)*

Each LLM is evaluated on 5 000 posts per language, presented in the language's **native script** (no translation). The model must classify each post as **Depressed** or **Not Depressed** without seeing the ground-truth label. Performance is measured with accuracy, precision, recall, and F1.

**Languages:**

| Language | Dataset | Script | Posts | Balance |
|----------|---------|--------|-------|---------|
| Arabic | CairoDep (Egyptian dialect) | Arabic script | 5 000 | 2 500 dep + 2 500 normal |
| Urdu | Urdu Depression Dataset | Roman Urdu (transliterated) | 5 000 | 2 500 dep + 2 500 normal |
| Chinese | TBD | Chinese script | 5 000 | 2 500 dep + 2 500 normal |

**Why native script?** The research goal is to test true multilingual capability — not how well models handle translated content. Arabic posts are sourced from Egyptian social media; Urdu from Pakistani Roman-script social media. Each has distinct cultural patterns that challenge generic models.

**Arabic pre-processing pipeline:** The Arabic dataset required an additional ethics step. Raw CairoDep posts were translated to English (Cohere `command-r-08-2024`) so the research team could screen for inappropriate content before evaluation. 42 posts were removed (translation failures, explicit sexual content, graphic violence, PII). The final evaluation uses the **original Arabic text**, not the translations.

**Urdu labels:** The raw Urdu dataset has four severity levels (`mild`, `moderate`, `severe`, `non-depression`). For binary classification these are collapsed: `mild/moderate/severe → depressed`, `non-depression → not depressed`. The original severity label is preserved in the data for analysis.

**Models evaluated:** Gemini 2.0 Flash, DeepSeek Chat, ChatGPT (GPT-4o-mini), Claude Haiku 4.5

**Prompts:** Each language uses a language-specific few-shot V3 prompt with culturally-relevant examples (e.g. Arabic V3 addresses religious phrases, Egyptian dialect slang, and hashtag patterns that commonly cause false positives). The Urdu V3 prompt addresses Roman Urdu political commentary and Urdu poetry.

**Results location:** `results/phase2/experiment1/`

---

### Experiment 2 — Keyword Attribution *(active)*

Builds directly on Experiment 1. Rather than asking the model to classify **and** explain simultaneously (which risks post-hoc rationalization), Experiment 2 feeds each model its **own Experiment 1 predictions** and asks: *"Given that you labelled this post as X, which specific words drove that decision?"*

**Design rationale:** Combining classification and keyword extraction in one prompt conflates the two tasks — the model may choose words to justify a label it has already settled on, rather than surfacing the evidence it actually used. By separating the steps, Experiment 2 captures true attribution: the model explains a prediction it has already committed to.

**Input per post (from Experiment 1 result files):**
- The original post text (Arabic script / Roman Urdu)
- The model's own predicted label (`Depressed` or `Not Depressed`)

**Output format — 2 lines per post (nothing else):**

```
الحزن, الوحدة, الخوف
sadness, loneliness, fear
```

| Line | Content |
|------|---------|
| 1 | Key word(s) from the post in the **original language** (Arabic script / Roman Urdu), comma-separated |
| 2 | One-word English translation of each keyword, in the same order, comma-separated |

**Why keyword attribution?**
Surfacing the words that drove each classification enables post-hoc analysis:
- Which Arabic/Urdu terms are most predictive of depression across models?
- Where do models diverge — and which words explain the disagreement?
- Do models latch onto genuine clinical signals or spurious surface features (e.g. religious phrases, common interjections)?

**Prompts:** Language-specific attribution V3 prompts (`v3_arabic_exp2`, `v3_exp2`) — 6 cultural few-shot examples each showing a post, its label, and the correct 2-line attribution response.

**Each result entry includes:**
```json
{
  "index": 42,
  "post_full": "...",
  "ground_truth": "depressed",
  "prediction": "depressed",
  "keywords": ["الحزن", "الوحدة", "الخوف"],
  "translations": ["sadness", "loneliness", "fear"],
  "raw_response_exp2": "الحزن, الوحدة, الخوف\nsadness, loneliness, fear"
}
```

**Results location:** `results/phase2/experiment2/`

---

### Experiment 3 — *Coming Soon*

Planned for a future phase. Will be implemented and documented when the design is finalised.

---

## Dataset Sources

| Language | Dataset | Source |
|----------|---------|--------|
| Arabic | [CairoDep](https://github.com/) | Egyptian Arabic social media (Twitter, Reddit, Facebook, crowdsourcing) |
| English | Sentiment Tweets | Kaggle |
| Spanish | [Spanish Depression Tweets](https://www.kaggle.com/datasets/francescoronzano/spanish-tweets-suggesting-depression) | Kaggle |
| Urdu | Urdu Depression Dataset | Roman Urdu social media, academically curated |
| Chinese | [Google Drive](https://drive.google.com/file/d/1fNKtoo4SP98OAhalMjNRZfFqmQZsQ0fh/view) | TBD |

---

## Repository Structure

```
multilingual-mental-health/
│
├── data/
│   ├── raw/                         # Original unmodified datasets
│   │   ├── arabic/
│   │   │   └── CairoDep_Datasets.csv        # 7 000 posts: post, label, dialect, source
│   │   ├── english/
│   │   │   └── sentiment_tweets3.csv
│   │   ├── spanish/
│   │   │   └── spanish_tweets_suggesting_signs_of_depression_v1.csv
│   │   └── urdu/
│   │       └── Depression.csv               # 25 004 posts: Text, Label (4-class severity)
│   │
│   ├── cleaned/                     # Parsed + normalised full datasets (cached)
│   │   ├── arabic.json              # 7 000 posts, binary labels only (dialect/source dropped)
│   │   └── urdu.json                # 25 002 posts, binary labels + severity preserved
│   │
│   ├── phase1/                      # Phase 1 (exploratory) data
│   │   ├── sampled/                 # 500-post stratified samples used in Phase 1
│   │   │   ├── arabic.json          #   500 posts (250+250), seed=42
│   │   │   ├── chinese.json         #   500 posts (silver-labelled via consensus)
│   │   │   ├── english.json
│   │   │   ├── spanish.json         #   500 posts (silver-labelled via consensus)
│   │   │   ├── urdu.json            #   500 posts (250+250), seed=42
│   │   │   └── urdu_english.json    #   Urdu posts machine-translated to English (analysis)
│   │   └── labeler_progress/        # Checkpoints from silver-label consensus pipeline
│   │
│   └── phase2/                      # Phase 2 (Experiment 1) data
│       ├── arabic_6000samples_seed42.json       # Intermediate: 6 000 raw Arabic posts for translation
│       ├── arabic_5000samples_seed42.json       # EXPERIMENT INPUT: 5 000 Arabic posts (original script)
│       ├── urdu_5000samples_seed42.json         # EXPERIMENT INPUT: 5 000 Urdu posts (Roman script)
│       ├── translated/
│       │   ├── arabic_6000samples_seed42_translated.json   # Cohere translations (ethics review)
│       │   └── filtered/
│       │       └── arabic_6000samples_seed42_filtered.json # 5 958 posts after removing 42 flagged
│       └── translation_progress/    # Checkpoints from Cohere translation pipeline
│
├── evaluation/                      # Shared evaluation library (used by all phases)
│   ├── prompts.py                   # Classification prompts (V1/V2/V3 + Arabic/Chinese V3)
│   ├── parsers.py                   # Language-specific dataset parsers (one class per language)
│   ├── metrics.py                   # EvaluationMetrics: accuracy, precision, recall, F1
│   ├── sampler.py                   # DatasetSampler: stratified sampling utility
│   └── cross_lingual.py             # Cross-lingual evaluation helpers (Phase 1)
│
├── models/                          # LLM provider wrappers (one file per provider)
│   ├── base.py                      # Abstract ModelProvider base class
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   ├── deepseek_provider.py
│   └── claude_provider.py
│
├── labeler/                         # Silver-label consensus pipeline (Chinese + Spanish)
│   ├── label_posts.py               # Main pipeline: XLM-RoBERTa + Grok consensus
│   ├── checkpoint.py                # Resume-safe progress tracking
│   ├── translator.py                # Cached Google Translate helper
│   └── classifiers/
│       ├── base.py                  # Abstract LabelerClassifier interface
│       ├── xlm_roberta.py           # HuggingFace multilingual depression classifier
│       └── grok.py                  # xAI Grok classifier
│
├── scripts/
│   ├── phase1/                      # Phase 1 exploratory scripts (archived, do not modify)
│   │   ├── prepare_data.py          # Parse raw data + create 500-post samples
│   │   ├── runner.py                # Phase 1 interactive evaluation runner
│   │   ├── smoke_test.py            # Quick API sanity check (10 posts)
│   │   ├── translate_urdu.py        # Machine-translate Urdu samples to English
│   │   ├── analyze_urdu_errors.py   # Deep-dive Urdu classification error analysis
│   │   └── verify_labels.py         # Verify silver-label quality
│   │
│   └── phase2/                      # Phase 2 scripts (active)
│       ├── prepare_arabic.py         # Step 1a: sample 6 000 Arabic posts from CairoDep
│       ├── translate_arabic.py       # Step 1b: translate with Cohere (ethics review)
│       ├── prepare_experiment1.py    # Step 2: create 5 000-post eval files for Arabic + Urdu
│       └── runner.py                 # Step 3: main Phase 2 evaluation runner (Experiments 1/2/3)
│
├── experiments/
│   └── phase1/                      # All Phase 1 results and analysis (read-only archive)
│       ├── ANALYSIS_GUIDE.md        # How to interpret Phase 1 results
│       └── results/                 # Phase 1 JSON result files
│           └── urdu_comprehensive_analysis.md
│
├── results/
│   └── phase2/
│       ├── experiment1/             # Experiment 1 output files
│       │   │                        #   <model>_<language>_<timestamp>.json
│       │   └── README.md            #   comparison_<timestamp>.json
│       └── experiment2/             # Experiment 2 output files (+ keywords/translations fields)
│
├── config.py                        # API key loader (reads from .env)
├── requirements.txt
└── .env                             # API keys (not committed)
```

---

## Quickstart — Phase 2 Experiment 1

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API keys to .env
#    GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, CLAUDE_API_KEY, COHERE_API_KEY

# 3. Prepare experiment data (Arabic + Urdu 5 000-post files)
python scripts/phase2/prepare_experiment1.py

# 4. Run evaluation (interactive menu)
python scripts/phase2/runner.py
#    -> Select [1] Experiment 1
#    -> Select model(s)
#    -> Select language(s): Arabic, Urdu (Chinese pending dataset)
```

**CLI flags for the runner:**

| Flag | Effect |
|------|--------|
| `--fresh` | Ignore partial results, start from scratch |
| `--delay N` | Seconds between API calls (default: 1.0) |
| `--workers N` | Parallel requests per model (default: 1) |
| `--prompt v2` | Override language-specific prompt (choices: v1, v2, v3, v3_arabic, v3_chinese) |

---

## Prompt Versions

| Key | Description | Best for |
|-----|-------------|----------|
| `v1` | Zero-shot, minimal instructions | Baseline comparison |
| `v2` | Enhanced clinical framework, handles sarcasm + edge cases | General use |
| `v3` | Few-shot with 6 Roman Urdu examples, political/poetry FP guards | Urdu (default) |
| `v3_arabic` | Few-shot with 6 Arabic-script examples, religious/hashtag FP guards | Arabic Exp 1 (default) |
| `v3_chinese` | Clinical framework only — examples pending dataset | Chinese (default) |
| `v3_exp2` | Urdu attribution: given a label, identify the words that drove it (2-line output) | Urdu Exp 2 (default) |
| `v3_arabic_exp2` | Arabic attribution: same design for Arabic/Egyptian dialect | Arabic Exp 2 (default) |

---

## Arabic Translation Pipeline (Ethics Pre-processing)

The Arabic evaluation required a content screening step before the dataset could be used in research. The pipeline is documented here for reproducibility.

```
CairoDep_Datasets.csv (7 000 posts)
    ↓  scripts/phase2/prepare_arabic.py
arabic_6000samples_seed42.json  (stratified 6 000, seed=42)
    ↓  scripts/phase2/translate_arabic.py  (Cohere command-r-08-2024, ~1.7h)
arabic_6000samples_seed42_translated.json  (6 000 English translations)
    ↓  Manual content review (42 posts removed)
translated/filtered/arabic_6000samples_seed42_filtered.json  (5 958 posts)
    ↓  scripts/phase2/prepare_experiment1.py
arabic_5000samples_seed42.json  (5 000 posts, original Arabic script, eval-ready)
```

Posts removed: 18 translation failures, 20 explicit sexual content, 2 graphic violence, 2 PII/spam.

---

## Phase 1 Archive

All exploratory Phase 1 work (500-sample pilots across Arabic, English, Spanish, Urdu) is archived in `experiments/phase1/`. See `experiments/phase1/ANALYSIS_GUIDE.md` for a guide to interpreting those results. Phase 1 scripts are preserved in `scripts/phase1/` and should not be modified.
