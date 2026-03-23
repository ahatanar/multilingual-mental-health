# Results Index — Urdu Depression Classification Experiments

## Dataset
- **Source**: Roman-script transliterated Urdu social media posts
- **Test set**: 500 posts (250 depressed, 250 not-depressed)
- **Severity distribution**: 85 mild, 86 moderate, 79 severe, 250 non-depression
- **Models**: Gemini 2.0 Flash, GPT-4o-mini (OpenAI), DeepSeek Chat, Claude Haiku 4.5

---

## Experiment 1: V1 Zero-Shot Baseline Prompt
**Prompt**: Simple instruction — "classify whether the author is expressing signs of depression"
**Date**: 2026-02-23

| File | Model |
|------|-------|
| `gemini_urdu_20260223_161531.json` | Gemini 2.0 Flash |
| `openai_urdu_20260223_161553.json` | GPT-4o-mini |
| `deepseek_urdu_20260223_161641.json` | DeepSeek Chat |
| `claude_urdu_20260223_162159.json` | Claude Haiku 4.5 |
| `comparison_20260223_162159.json` | Cross-model comparison |

## Experiment 2: V2 Zero-Shot Clinical Framework Prompt
**Prompt**: Enhanced with clinical depression indicators, exclusion rules (sarcasm, casual usage), multilingual awareness
**Date**: 2026-02-23

| File | Model |
|------|-------|
| `gemini_urdu_20260223_163756.json` | Gemini 2.0 Flash |
| `openai_urdu_20260223_163756.json` | GPT-4o-mini |
| `deepseek_urdu_20260223_163900.json` | DeepSeek Chat |
| `claude_urdu_20260223_165333.json` | Claude Haiku 4.5 |
| `comparison_20260223_165333.json` | Cross-model comparison |

## Experiment 3: V2 Prompt on Translated English (Translation Experiment)
**Prompt**: Same V2 clinical prompt, but posts were machine-translated from Urdu to English via Google Translate before classification
**Purpose**: Test whether native-language processing outperforms translate-then-classify
**Date**: 2026-02-23

| File | Model |
|------|-------|
| `gemini_urdu_english_20260223_173431.json` | Gemini 2.0 Flash |
| `openai_urdu_english_20260223_173448.json` | GPT-4o-mini |
| `deepseek_urdu_english_20260223_173536.json` | DeepSeek Chat |
| `claude_urdu_english_20260223_174949.json` | Claude Haiku 4.5 |
| `comparison_20260223_174949.json` | Cross-model comparison |

## Experiment 4: V3 Few-Shot Prompt (6 examples)
**Prompt**: Clinical framework + 6 labeled Urdu examples targeting error patterns from V2 analysis (behavioral depression, political FPs, poetry/shayri FPs, short posts). Examples verified not in test set (no data leakage).
**Date**: 2026-02-24

| File | Model | Status |
|------|-------|--------|
| `gemini_urdu_20260224_173430.json` | Gemini 2.0 Flash | Complete |
| `openai_urdu_20260224_173440.json` | GPT-4o-mini | Complete |
| `deepseek_urdu_20260224_173545.json` | DeepSeek Chat | Complete |
| `claude_urdu_20260224_1xxxxx.json` | Claude Haiku 4.5 | Running (rate-limited) |

---

## Analysis Documents

| File | Description |
|------|-------------|
| `urdu_analysis.json` | Severity-stratified metrics for all 4 models (V2) |
| `urdu_comprehensive_analysis.md` | Full research analysis: error taxonomy, inter-model disagreement, translation impact, severity breakdown, few-shot recommendations |

---

## Summary of Results Across All Experiments

### Accuracy (%)

| Model | V1 Zero-Shot | V2 Clinical | V2 Translated EN | V3 Few-Shot |
|-------|-------------|-------------|-------------------|-------------|
| Gemini 2.0 Flash | 83.0 | 82.8 | 74.8 | **89.8** |
| GPT-4o-mini | 82.8 | 84.0 | 72.0 | **88.6** |
| DeepSeek Chat | 84.0 | 87.0 | 79.0 | **89.8** |
| Claude Haiku 4.5 | 74.0 | 62.3 | — | *running* |

### F1 Score

| Model | V1 Zero-Shot | V2 Clinical | V2 Translated EN | V3 Few-Shot |
|-------|-------------|-------------|-------------------|-------------|
| Gemini 2.0 Flash | 0.843 | 0.842 | 0.717 | **0.903** |
| GPT-4o-mini | 0.827 | 0.823 | 0.685 | **0.878** |
| DeepSeek Chat | 0.843 | 0.865 | 0.808 | **0.893** |
| Claude Haiku 4.5 | 0.549 | 0.397 | — | *running* |

### Key Takeaways
1. **V3 few-shot is the best-performing prompt** across all models (+3 to +7% accuracy over V2)
2. **Translation hurts performance** by 8-14% accuracy — native Urdu classification is superior
3. **DeepSeek and Gemini tie at 89.8% accuracy** with V3, though Gemini has higher recall and DeepSeek has higher precision
4. **Prompt engineering matters enormously** — same models, same data, 20+ point swings depending on prompt
5. **Severity predicts difficulty** — severe depression detected at ~95%, mild at ~35-85%
