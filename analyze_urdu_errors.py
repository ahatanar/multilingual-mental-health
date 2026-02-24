#!/usr/bin/env python3
"""
Comprehensive Urdu mental health classification error analysis.
Reads all result files and produces detailed analysis data.
"""

import json
import os
import sys

BASE = r"D:\Downloads\multilingual-mental-health"

def load_json(path):
    with open(os.path.join(BASE, path), 'r', encoding='utf-8') as f:
        return json.load(f)

# Load all V2 result files
gemini = load_json("results/gemini_urdu_20260223_163756.json")
openai_r = load_json("results/openai_urdu_20260223_163756.json")
deepseek = load_json("results/deepseek_urdu_20260223_163900.json")
claude = load_json("results/claude_urdu_20260223_165333.json")

# Load sampled data for severity
sampled = load_json("data/sampled/urdu.json")

# Load V1 and V2 comparison
v1_comp = load_json("results/comparison_20260223_162159.json")
v2_comp = load_json("results/comparison_20260223_165333.json")

# Load translation results
gemini_eng = load_json("results/gemini_urdu_english_20260223_173431.json")
openai_eng = load_json("results/openai_urdu_english_20260223_173448.json")
deepseek_eng = load_json("results/deepseek_urdu_english_20260223_173536.json")
claude_eng = load_json("results/claude_urdu_english_20260223_174949.json")

# Load analysis
analysis = load_json("results/urdu_analysis.json")

models = {
    'gemini': gemini,
    'openai': openai_r,
    'deepseek': deepseek,
    'claude': claude
}

# Build severity map from sampled data
severity_map = {}
for i, sample in enumerate(sampled['samples']):
    severity_map[i+1] = sample['severity']

print("=" * 80)
print("SECTION 1: Overall V2 Metrics")
print("=" * 80)
for name, data in models.items():
    m = data['metrics']
    cm = m['confusion_matrix']
    print(f"\n{name.upper()}:")
    print(f"  TP={cm['true_positives']}, FP={cm['false_positives']}, TN={cm['true_negatives']}, FN={cm['false_negatives']}")
    print(f"  Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, Accuracy={m['accuracy']:.4f}")
    # Compute specificity
    spec = cm['true_negatives'] / (cm['true_negatives'] + cm['false_positives']) if (cm['true_negatives'] + cm['false_positives']) > 0 else 0
    print(f"  Specificity={spec:.4f}")

print("\n" + "=" * 80)
print("SECTION 2: Error Extraction - All Errors Per Model")
print("=" * 80)

all_errors = {}  # model -> list of error dicts
all_results_by_index = {}  # index -> {model: prediction}

for name, data in models.items():
    errors = []
    for r in data['results']:
        idx = r['index']
        if idx not in all_results_by_index:
            all_results_by_index[idx] = {
                'post': r['post_full'],
                'ground_truth': r['ground_truth'],
                'word_count': r['word_count'],
                'severity': severity_map.get(idx, 'unknown')
            }
        all_results_by_index[idx][name] = r['prediction']

        if r['prediction'] != r['ground_truth']:
            errors.append({
                'index': idx,
                'post': r['post_full'],
                'ground_truth': r['ground_truth'],
                'prediction': r['prediction'],
                'word_count': r['word_count'],
                'severity': severity_map.get(idx, 'unknown'),
                'error_type': 'FP' if r['ground_truth'] == 'not depressed' else 'FN'
            })
    all_errors[name] = errors
    print(f"\n{name.upper()}: {len(errors)} total errors (FP={sum(1 for e in errors if e['error_type']=='FP')}, FN={sum(1 for e in errors if e['error_type']=='FN')})")

print("\n" + "=" * 80)
print("SECTION 3: Severity Breakdown for Errors")
print("=" * 80)
for name, errors in all_errors.items():
    fn_errors = [e for e in errors if e['error_type'] == 'FN']
    sev_counts = {}
    for e in fn_errors:
        sev = e['severity']
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    print(f"\n{name.upper()} False Negatives by severity:")
    for sev in ['mild', 'moderate', 'severe']:
        count = sev_counts.get(sev, 0)
        total = analysis['severity_breakdown'][name][sev]['total'] if sev in analysis['severity_breakdown'].get(name, {}) else '?'
        print(f"  {sev}: {count} missed out of {total}")

print("\n" + "=" * 80)
print("SECTION 4: Inter-Model Disagreement")
print("=" * 80)

# Find posts where models disagree
disagreement_count = 0
all_agree_correct = 0
all_agree_wrong = 0
disagreement_posts = []

for idx in sorted(all_results_by_index.keys()):
    info = all_results_by_index[idx]
    preds = [info.get(m) for m in ['gemini', 'openai', 'deepseek', 'claude']]
    gt = info['ground_truth']

    if len(set(preds)) > 1:
        disagreement_count += 1
        correct_models = [m for m in ['gemini', 'openai', 'deepseek', 'claude'] if info.get(m) == gt]
        wrong_models = [m for m in ['gemini', 'openai', 'deepseek', 'claude'] if info.get(m) != gt]
        disagreement_posts.append({
            'index': idx,
            'post': info['post'][:80],
            'ground_truth': gt,
            'severity': info['severity'],
            'correct': correct_models,
            'wrong': wrong_models,
            'word_count': info['word_count']
        })
    else:
        if preds[0] == gt:
            all_agree_correct += 1
        else:
            all_agree_wrong += 1

print(f"Total disagreements: {disagreement_count}")
print(f"All agree correct: {all_agree_correct}")
print(f"All agree wrong: {all_agree_wrong}")

# Pairwise agreement
print("\nPairwise agreement rates:")
model_names = ['gemini', 'openai', 'deepseek', 'claude']
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        agree = sum(1 for idx in all_results_by_index if all_results_by_index[idx].get(m1) == all_results_by_index[idx].get(m2))
        print(f"  {m1}-{m2}: {agree}/500 ({agree/5:.1f}%)")

# Which model is most often the sole wrong one?
print("\nSole wrong model counts:")
sole_wrong = {m: 0 for m in model_names}
for dp in disagreement_posts:
    if len(dp['wrong']) == 1:
        sole_wrong[dp['wrong'][0]] += 1
print(f"  {sole_wrong}")

# Which model is most often the sole correct one?
sole_correct = {m: 0 for m in model_names}
for dp in disagreement_posts:
    if len(dp['correct']) == 1:
        sole_correct[dp['correct'][0]] += 1
print(f"Sole correct: {sole_correct}")

print("\n" + "=" * 80)
print("SECTION 5: Word Count / Post Length Analysis")
print("=" * 80)

# Bucket posts by word count
buckets = [(1, 3, "1-3 words"), (4, 7, "4-7 words"), (8, 15, "8-15 words"), (16, 30, "16-30 words"), (31, 999, "31+ words")]

for name in model_names:
    print(f"\n{name.upper()} accuracy by post length:")
    for lo, hi, label in buckets:
        total = 0
        correct = 0
        for idx, info in all_results_by_index.items():
            wc = info['word_count']
            if lo <= wc <= hi:
                total += 1
                if info.get(name) == info['ground_truth']:
                    correct += 1
        if total > 0:
            print(f"  {label}: {correct}/{total} = {correct/total:.3f}")

print("\n" + "=" * 80)
print("SECTION 6: False Positive Posts (All Models)")
print("=" * 80)

# Collect all unique FP posts
fp_posts = {}
for name, errors in all_errors.items():
    for e in errors:
        if e['error_type'] == 'FP':
            idx = e['index']
            if idx not in fp_posts:
                fp_posts[idx] = {'post': e['post'], 'word_count': e['word_count'], 'models_wrong': []}
            fp_posts[idx]['models_wrong'].append(name)

print(f"\nTotal unique FP posts: {len(fp_posts)}")
for idx in sorted(fp_posts.keys()):
    fp = fp_posts[idx]
    print(f"\n  Index {idx} (wc={fp['word_count']}, models={fp['models_wrong']}):")
    print(f"    \"{fp['post'][:120]}\"")

print("\n" + "=" * 80)
print("SECTION 7: False Negative Posts (All Models)")
print("=" * 80)

fn_posts = {}
for name, errors in all_errors.items():
    for e in errors:
        if e['error_type'] == 'FN':
            idx = e['index']
            if idx not in fn_posts:
                fn_posts[idx] = {'post': e['post'], 'word_count': e['word_count'], 'severity': e['severity'], 'models_wrong': []}
            fn_posts[idx]['models_wrong'].append(name)

print(f"\nTotal unique FN posts: {len(fn_posts)}")
for idx in sorted(fn_posts.keys()):
    fn = fn_posts[idx]
    print(f"\n  Index {idx} (wc={fn['word_count']}, sev={fn['severity']}, models={fn['models_wrong']}):")
    print(f"    \"{fn['post'][:120]}\"")

print("\n" + "=" * 80)
print("SECTION 8: Translation Comparison Metrics")
print("=" * 80)

trans_models = {
    'gemini': gemini_eng,
    'openai': openai_eng,
    'deepseek': deepseek_eng,
    'claude': claude_eng
}

for name in model_names:
    native_m = models[name]['metrics']
    trans_m = trans_models[name]['metrics']
    print(f"\n{name.upper()} Native Urdu vs English Translation:")
    print(f"  Native:  Acc={native_m['accuracy']:.4f}, P={native_m['precision']:.4f}, R={native_m['recall']:.4f}, F1={native_m['f1_score']:.4f}")
    print(f"  English: Acc={trans_m['accuracy']:.4f}, P={trans_m['precision']:.4f}, R={trans_m['recall']:.4f}, F1={trans_m['f1_score']:.4f}")
    print(f"  Delta:   Acc={trans_m['accuracy']-native_m['accuracy']:+.4f}, F1={trans_m['f1_score']-native_m['f1_score']:+.4f}")

# Also track which posts changed prediction after translation
print("\n\nTranslation flips per model:")
for name in model_names:
    native_results = {r['index']: r['prediction'] for r in models[name]['results']}
    trans_results = {r['index']: r['prediction'] for r in trans_models[name]['results']}

    improved = 0
    degraded = 0
    for idx in native_results:
        gt = all_results_by_index[idx]['ground_truth']
        n_pred = native_results[idx]
        t_pred = trans_results.get(idx)
        if t_pred is None:
            continue
        n_correct = (n_pred == gt)
        t_correct = (t_pred == gt)
        if not n_correct and t_correct:
            improved += 1
        elif n_correct and not t_correct:
            degraded += 1
    print(f"  {name}: improved={improved}, degraded={degraded}")

print("\n" + "=" * 80)
print("SECTION 9: V1 vs V2 Comparison")
print("=" * 80)

v1_models = v1_comp['models']
v2_models = v2_comp['models']

model_map = {
    'Gemini 2.0 Flash': 'gemini',
    'ChatGPT (GPT-4o-mini)': 'openai',
    'DeepSeek Chat': 'deepseek',
    'Claude Haiku 4.5': 'claude'
}

for full_name, short_name in model_map.items():
    v1 = v1_models[full_name]['urdu']
    v2 = v2_models[full_name]['urdu']
    print(f"\n{full_name}:")
    print(f"  V1: Acc={v1['accuracy']:.4f}, P={v1['precision']:.4f}, R={v1['recall']:.4f}, F1={v1['f1_score']:.4f}")
    print(f"  V2: Acc={v2['accuracy']:.4f}, P={v2['precision']:.4f}, R={v2['recall']:.4f}, F1={v2['f1_score']:.4f}")
    v1_cm = v1['confusion_matrix']
    v2_cm = v2['confusion_matrix']
    print(f"  V1 FP={v1_cm['false_positives']}, V2 FP={v2_cm['false_positives']} (delta={v2_cm['false_positives']-v1_cm['false_positives']})")
    print(f"  V1 FN={v1_cm['false_negatives']}, V2 FN={v2_cm['false_negatives']} (delta={v2_cm['false_negatives']-v1_cm['false_negatives']})")

print("\n" + "=" * 80)
print("SECTION 10: All FP posts with full text for categorization")
print("=" * 80)
for idx in sorted(fp_posts.keys()):
    fp = fp_posts[idx]
    print(f"IDX={idx}|WC={fp['word_count']}|MODELS={','.join(fp['models_wrong'])}|POST={fp['post']}")

print("\n" + "=" * 80)
print("SECTION 11: All FN posts with full text for categorization")
print("=" * 80)
for idx in sorted(fn_posts.keys()):
    fn = fn_posts[idx]
    print(f"IDX={idx}|WC={fn['word_count']}|SEV={fn['severity']}|MODELS={','.join(fn['models_wrong'])}|POST={fn['post']}")

print("\n" + "=" * 80)
print("SECTION 12: Posts where ALL 4 models got it wrong")
print("=" * 80)
all_wrong_posts = []
for idx in sorted(all_results_by_index.keys()):
    info = all_results_by_index[idx]
    gt = info['ground_truth']
    if all(info.get(m) != gt for m in model_names):
        all_wrong_posts.append({
            'index': idx,
            'post': info['post'],
            'ground_truth': gt,
            'severity': info['severity'],
            'word_count': info['word_count']
        })
        print(f"  IDX={idx}|GT={gt}|SEV={info['severity']}|WC={info['word_count']}|POST={info['post'][:100]}")

print(f"\nTotal all-4-wrong: {len(all_wrong_posts)}")

# Non-Urdu or foreign language posts
print("\n" + "=" * 80)
print("SECTION 13: Potential non-Urdu posts")
print("=" * 80)
# Check for posts that seem to be in other languages
for idx in sorted(all_results_by_index.keys()):
    info = all_results_by_index[idx]
    post = info['post'].lower()
    # Check for Turkish, Malay, Hindi-only patterns
    # Simple heuristic: high ratio of non-common-Urdu patterns
    # Just print posts that have common Turkish/Malay words
    turkish_words = ['bir', 'olmuyorsun', 'yeter', 'dil', 'ama', 'olan', 'icin', 'benim']
    malay_words = ['perlu', 'sepatutnya', 'kenapa', 'orang', 'sangat', 'bodoh', 'macam']

    has_turkish = sum(1 for w in turkish_words if w in post.split()) >= 2
    has_malay = sum(1 for w in malay_words if w in post.split()) >= 2

    if has_turkish or has_malay:
        print(f"  IDX={idx}|GT={info['ground_truth']}|POST={info['post'][:100]}")

# Agreement matrix
print("\n" + "=" * 80)
print("SECTION 14: Model agreement on error cases")
print("=" * 80)
# For each pair of models, count how many errors they share
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        e1_indices = set(e['index'] for e in all_errors[m1])
        e2_indices = set(e['index'] for e in all_errors[m2])
        shared = e1_indices & e2_indices
        only_m1 = e1_indices - e2_indices
        only_m2 = e2_indices - e1_indices
        print(f"  {m1}-{m2}: shared_errors={len(shared)}, only_{m1}={len(only_m1)}, only_{m2}={len(only_m2)}")

print("\n\nDONE.")
