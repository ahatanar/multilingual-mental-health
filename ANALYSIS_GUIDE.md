# Analysis Guide — Multilingual Mental Health Classification

This guide explains how to interpret the results and write the analysis section for the report.

---

## How to Read the Result Files

Each result JSON file has this structure:

```json
{
  "metadata": { "model": "...", "language": "...", "sample_size": 500 },
  "metrics": {
    "confusion_matrix": { "true_positives": ..., "false_positives": ..., "true_negatives": ..., "false_negatives": ... },
    "precision": ...,
    "recall": ...,
    "f1_score": ...,
    "accuracy": ...
  },
  "results": [
    {
      "index": 1,
      "post_full": "...",
      "ground_truth": "depressed" or "not depressed",
      "prediction": "depressed" or "not depressed",
      "raw_response": "...",
      "word_count": ...
    }
  ]
}
```

- **`metrics`** = overall performance numbers (what goes in tables)
- **`results`** = per-post predictions (what you dig into for error analysis)
- **`comparison_*.json`** = side-by-side metrics for multiple models from the same run

See `results/RESULTS_INDEX.md` for which file corresponds to which experiment.

---

## What to Analyze (Milestone 3 Checklist)

### 1. Cross-Prompt Comparison (V1 vs V2 vs V3)

Compare the **same model on the same data** with different prompts.

**What to report:**
- Accuracy, Precision, Recall, F1 for each prompt version
- Which prompt version works best overall? (V3 few-shot)
- Which models benefited most from few-shot examples?
- Did any model get *worse* with a specific prompt? (Claude on V2)

**Where to find data:**
- V1 results: `*_urdu_20260223_161*.json`
- V2 results: `*_urdu_20260223_163*.json` / `*_165*.json`
- V3 results: `*_urdu_20260224_*.json`

**Key talking points:**
- V3 (few-shot) improved all models by +3 to +7% accuracy
- Few-shot examples let models learn Urdu-specific patterns (poetry vs real depression, political rants vs distress)
- V2 made Claude overly conservative (recall dropped from 36% to 25%)

### 2. Native vs Translated Performance

Compare **same model, same prompt (V2), different language input** (native Urdu vs English translation).

**What to report:**
- Side-by-side accuracy/F1 for native Urdu vs translated English
- Which direction did errors go? (more false negatives? more false positives?)
- Conclude: is translate-then-classify a viable strategy?

**Where to find data:**
- Native Urdu V2: `*_urdu_20260223_163*.json`
- Translated English V2: `*_urdu_english_20260223_17*.json`

**Key talking points:**
- Translation drops accuracy 8-14% across all models
- Translation especially hurts recall (models miss more depressed posts in English)
- Cultural nuance and transliteration patterns are lost in translation
- Conclusion: native-language processing is essential for mental health NLP

### 3. Cross-Model Comparison

Compare **different models on the same data with the same prompt**.

**What to report:**
- Table of all 4 models' metrics on V3 (best prompt)
- Precision vs recall tradeoff: Gemini = high recall, OpenAI/DeepSeek = balanced, Claude = ultra-high precision but misses most cases
- Which model would you recommend for a screening tool? (consider: is it worse to miss a depressed person or to flag a non-depressed one?)

**Key talking points:**
- DeepSeek and Gemini tied at 89.8% accuracy on V3
- Gemini has highest recall (94.4%) — catches the most depressed posts
- OpenAI has highest precision (94.5%) — fewest false alarms
- For mental health screening, high recall matters more (missing depression is more dangerous than a false alarm)

### 4. Severity Analysis

**Where to find data:** `results/urdu_analysis.json` has severity breakdown.

**What to report:**
- Detection rate by severity level: mild, moderate, severe
- Which models struggle most with mild depression?
- Is there a "severity threshold" below which models fail?

**Key talking points:**
- Severe depression detected at 75-99% across models
- Moderate: 50-95%
- Mild: 34-85% (the critical failure zone)
- Claude missed 66% of mild cases even with V2
- Clinical implication: LLMs are better at detecting obvious cases but miss subtle early-stage depression

### 5. Error Analysis (Deep Dive)

**Where to find data:** `results/urdu_comprehensive_analysis.md` has the full error taxonomy.

**What to report:**
- Categories of errors (false positives and false negatives)
- Example posts that all models got wrong, and why
- Inter-model agreement rates

**Error categories to discuss:**
1. Short posts with insufficient context (~58 instances)
2. Cultural/poetic expressions misread as depression (~35 instances)
3. Political commentary flagged as depression (~22 FP, mostly Gemini)
4. Behavioral depression symptoms missed (~30 FN, no sad words but depressed behavior)
5. Non-Urdu foreign language posts in the dataset (~12 instances)

---

## How to Write the Analysis Section

### Structure Template

```
1. Introduction
   - Task: binary depression classification on Roman Urdu social media posts
   - 4 LLMs evaluated, 3 prompt strategies, 1 translation experiment

2. Experimental Setup
   - Dataset: 500 posts, balanced classes, severity-stratified
   - Models: Gemini 2.0 Flash, GPT-4o-mini, DeepSeek Chat, Claude Haiku 4.5
   - Prompts: V1 (zero-shot baseline), V2 (clinical framework), V3 (6-example few-shot)

3. Results
   - Table 1: Cross-prompt comparison (V1/V2/V3 x 4 models)
   - Table 2: Native vs translated performance
   - Table 3: Severity breakdown
   - Figure: Confusion matrices or bar charts

4. Discussion
   - Few-shot prompting significantly improves performance (+3 to +7%)
   - Native language processing outperforms translation pipeline
   - Mild depression remains the critical blind spot
   - Cultural/linguistic factors (poetry, political discourse) cause systematic errors
   - Model selection depends on use case (recall vs precision tradeoff)

5. Limitations
   - Dataset contains ~12 non-Urdu posts (noise)
   - Single dataset, single language for few-shot experiment
   - Few-shot examples are Urdu-specific (may not generalize)
   - API-based evaluation (no fine-tuning explored)
```

### Tips for Marks
- **Always include the numbers.** Don't just say "V3 was better" — say "V3 improved Gemini's F1 from 0.842 to 0.903 (+7.2%)"
- **Explain WHY, not just WHAT.** The few-shot examples taught models to distinguish poetry from real depression — that's the insight, not just the accuracy number
- **Include example posts.** Show 2-3 actual posts that were misclassified and explain why the model failed. The comprehensive analysis doc has these ready
- **Discuss clinical implications.** Missing a depressed person (false negative) is more dangerous than a false alarm (false positive). Which model/prompt combination minimizes this risk?
- **Acknowledge limitations.** The severity labels come from the original dataset authors. The few-shot examples are Urdu-specific. State these clearly.

---

## Quick Reference: File Locations

| What | Where |
|------|-------|
| All result JSONs | `results/` |
| Results index (which file = which experiment) | `results/RESULTS_INDEX.md` |
| Full error analysis | `results/urdu_comprehensive_analysis.md` |
| Severity breakdown data | `results/urdu_analysis.json` |
| Prompt definitions (V1/V2/V3) | `evaluation/prompts.py` |
| Sampled test data | `data/sampled/urdu.json` |
| Translated test data | `data/sampled/urdu_english.json` |
| Translation script | `translate_urdu.py` |
| Main evaluation runner | `runner.py` |
