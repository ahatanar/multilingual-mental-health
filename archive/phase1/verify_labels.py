"""
Verify silver labels on Chinese & Spanish sampled data using a second HF model.

Uses SajjadIslam/multiMentalRoBERTA-5-class (multiclass: Anxiety, Depression,
PTSD, Suicidal, None) to cross-check the gold labels from our labeling pipeline.

Flags critical disagreements: gold_label=depressed BUT model says not depressed.

Usage:
    py verify_labels.py                         # verify both languages
    py verify_labels.py --language chinese       # one language only
    py verify_labels.py --batch-size 32          # control batch size
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLED_DIR = os.path.join(PROJECT_ROOT, "data", "phase1", "sampled")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

VERIFIER_MODEL = "SajjadIslam/multiMentalRoBERTA-5-class"
DEPRESSION_THRESHOLD = 0.50

LANGUAGES = ["chinese", "spanish"]


# ── Translation ──────────────────────────────────────────────────────────────

def batch_translate(posts: List[str], language: str) -> List[str]:
    """Translate posts to English. Chinese uses googletrans, Spanish passed through."""
    if language == "spanish":
        # The verifier model handles English best, translate spanish too
        pass  # fall through to googletrans

    try:
        from googletrans import Translator
        translator = Translator()
        translated = []
        src = "zh-cn" if language == "chinese" else "es"

        # Translate in small batches to avoid API limits
        CHUNK = 20
        for i in range(0, len(posts), CHUNK):
            chunk = posts[i:i + CHUNK]
            for post in chunk:
                try:
                    result = translator.translate(post, src=src, dest="en")
                    translated.append(result.text)
                except Exception as e:
                    logger.warning(f"Translation failed for post {i}: {e}")
                    translated.append(post)  # fallback to original

            if i + CHUNK < len(posts):
                logger.info(f"  Translated {min(i + CHUNK, len(posts))}/{len(posts)}...")

        return translated
    except ImportError:
        logger.warning("googletrans not installed, using original text")
        return posts


# ── Model loading ────────────────────────────────────────────────────────────

def load_verifier():
    """Load the multiclass mental health classifier."""
    logger.info(f"Loading verifier model: {VERIFIER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(VERIFIER_MODEL)
    model.eval()

    # Get label mapping from config
    label_map = model.config.id2label
    logger.info(f"  Label mapping: {label_map}")

    # Find the depression class index
    dep_idx = None
    for idx, label in label_map.items():
        if "depress" in label.lower():
            dep_idx = int(idx)
            break

    if dep_idx is None:
        raise ValueError(f"Could not find 'Depression' class in model labels: {label_map}")

    logger.info(f"  Depression class index: {dep_idx} ('{label_map[dep_idx]}')")
    return tokenizer, model, label_map, dep_idx


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(texts: List[str], tokenizer, model, dep_idx: int,
                  label_map: dict, batch_size: int = 16) -> List[Dict]:
    """Run batched inference and return per-sample results."""
    results = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        for j in range(len(batch_texts)):
            p_dep = probs[j][dep_idx].item()
            top_idx = probs[j].argmax().item()
            top_label = label_map[top_idx]
            model_label = 1 if p_dep >= DEPRESSION_THRESHOLD else 0

            results.append({
                "model_label": model_label,
                "model_prob": round(p_dep, 4),
                "model_top_label": top_label,
            })

        if i + batch_size < total:
            logger.info(f"  Inference: {min(i + batch_size, total)}/{total}...")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def verify_language(language: str, tokenizer, model, label_map: dict,
                    dep_idx: int, batch_size: int = 16) -> Dict:
    """Run verification on a single language's sampled data."""
    print(f"\n{'='*60}")
    print(f"  🔍 Verifying: {language.upper()}")
    print(f"{'='*60}")

    # 1. Load sampled data
    sampled_path = os.path.join(SAMPLED_DIR, f"{language}.json")
    if not os.path.isfile(sampled_path):
        raise FileNotFoundError(f"No sampled data for '{language}'. Run labeler first.")

    with open(sampled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"  Loaded {len(samples)} samples")

    # 2. Translate to English
    print(f"  Translating to English...")
    posts = [s["post"] for s in samples]
    translated = batch_translate(posts, language)
    print(f"  ✓ Translated {len(translated)} posts")

    # 3. Run inference
    print(f"  Running verifier model...")
    model_results = run_inference(translated, tokenizer, model, dep_idx,
                                  label_map, batch_size=batch_size)
    print(f"  ✓ Inference complete")

    # 4. Build results with disagreement flags
    gold_to_int = {"depressed": 1, "not depressed": 0}
    results = []
    n_flagged = 0
    tp = fp = tn = fn = 0

    for i, (sample, mr) in enumerate(zip(samples, model_results)):
        gold = gold_to_int.get(sample["ground_truth"], 0)
        pred = mr["model_label"]

        # Confusion matrix
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        else:  # gold == 1 and pred == 0
            fn += 1

        flag = (gold == 1 and pred == 0)
        if flag:
            n_flagged += 1

        results.append({
            "id": str(i),
            "text": sample["post"][:200],  # truncate for readability
            "translated": translated[i][:200],
            "gold_label": gold,
            "model_label": pred,
            "model_prob": mr["model_prob"],
            "model_top_label": mr["model_top_label"],
            "disagreement_flag": flag,
        })

    # 5. Compute metrics
    total_n = len(results)
    n_gold_pos = sum(1 for r in results if r["gold_label"] == 1)
    n_gold_neg = total_n - n_gold_pos

    accuracy = (tp + tn) / total_n if total_n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Cohen's Kappa
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total_n ** 2) if total_n > 0 else 0
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # Top disagreement examples (lowest P_dep among flagged)
    flagged = [r for r in results if r["disagreement_flag"]]
    flagged_sorted = sorted(flagged, key=lambda r: r["model_prob"])
    top_examples = flagged_sorted[:10]

    summary = {
        "language": language,
        "inference_mode": "python_transformers",
        "verifier_model": VERIFIER_MODEL,
        "n_total": total_n,
        "n_gold_pos": n_gold_pos,
        "n_gold_neg": n_gold_neg,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_pos": round(precision, 4),
            "recall_pos": round(recall, 4),
            "f1_pos": round(f1, 4),
        },
        "cohen_kappa": round(kappa, 4),
        "disagreements": {
            "n_disagreement_flagged": n_flagged,
            "n_flagged_pct": round(n_flagged / n_gold_pos * 100, 1) if n_gold_pos > 0 else 0,
            "top_examples": [
                {
                    "id": ex["id"],
                    "model_prob": ex["model_prob"],
                    "model_top_label": ex["model_top_label"],
                    "text_preview": ex["text"][:100],
                }
                for ex in top_examples
            ],
        },
    }

    # Print summary
    print(f"\n  📊 Verification Results — {language.upper()}")
    print(f"  {'─'*40}")
    print(f"  Samples:     {total_n} ({n_gold_pos} depressed / {n_gold_neg} not depressed)")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Kappa:       {kappa:.4f}")
    print(f"  Confusion:   TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  ⚠ Flagged:   {n_flagged} posts (gold=depressed, model=not depressed)")
    if n_gold_pos > 0:
        print(f"               ({n_flagged/n_gold_pos*100:.1f}% of depressed posts)")

    return {"results": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Verify silver labels with second HF model")
    parser.add_argument("--language", choices=LANGUAGES, default=None,
                        help="Verify a specific language (default: both)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    languages = [args.language] if args.language else LANGUAGES

    # Load model once (shared across languages)
    tokenizer, model, label_map, dep_idx = load_verifier()

    all_summaries = {}
    for lang in languages:
        try:
            output = verify_language(lang, tokenizer, model, label_map,
                                      dep_idx, batch_size=args.batch_size)

            # Save results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(RESULTS_DIR, f"verification_{lang}_{timestamp}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved: {out_path}")

            all_summaries[lang] = output["summary"]

        except Exception as e:
            logger.error(f"[{lang}] Verification failed: {e}", exc_info=True)
            print(f"\n  ❌ {lang}: {e}")

    # Combined summary
    if all_summaries:
        print(f"\n{'='*60}")
        print(f"  ✅ Verification Complete")
        print(f"{'='*60}")
        for lang, s in all_summaries.items():
            m = s["metrics"]
            d = s["disagreements"]
            print(f"  {lang.upper():<10}  acc={m['accuracy']:.3f}  f1={m['f1_pos']:.3f}  "
                  f"kappa={s['cohen_kappa']:.3f}  flagged={d['n_disagreement_flagged']}/{s['n_gold_pos']}")
        print()


if __name__ == "__main__":
    main()
