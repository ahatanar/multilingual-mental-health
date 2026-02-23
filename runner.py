"""
Part 2: Run LLM evaluation on pre-sampled datasets.

Prerequisites:
    Run `py prepare_data.py` first to parse and sample datasets.

Run:
    py runner.py                        # interactive mode
    py runner.py --reparse              # re-parse + re-sample data first
    py runner.py --fresh                # ignore partial results, start fresh
    py runner.py --delay 21             # 21s between API calls (OpenAI free tier)

Loads sampled data from data/sampled/<language>.json and runs
classification with selected LLM provider(s).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict

from config import get_api_key, DEFAULT_MODELS
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider, ClaudeProvider
from evaluation.metrics import EvaluationMetrics
from evaluation.prompts import CLASSIFICATION_PROMPT
from evaluation.parsers import PARSERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Registry ────────────────────────────────────────────────────────────────

MODELS = {
    "gemini": {"class": GeminiProvider, "name": "Gemini 2.0 Flash", "default_model": "gemini-2.0-flash"},
    "deepseek": {"class": DeepSeekProvider, "name": "DeepSeek Chat", "default_model": "deepseek-chat"},
    "openai": {"class": OpenAIProvider, "name": "ChatGPT (GPT-4o-mini)", "default_model": "gpt-4o-mini"},
    "claude": {"class": ClaudeProvider, "name": "Claude Haiku 4.5", "default_model": "claude-haiku-4-5"},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLED_DIR = os.path.join(DATA_DIR, "sampled")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Data loading ─────────────────────────────────────────────────────────────

def discover_sampled_languages() -> Dict[str, str]:
    """Scan data/sampled/ for prepared language datasets."""
    available = {}
    if not os.path.isdir(SAMPLED_DIR):
        return available
    for f in sorted(os.listdir(SAMPLED_DIR)):
        if f.endswith(".json"):
            lang = f.replace(".json", "")
            available[lang] = os.path.join(SAMPLED_DIR, f)
    return available


def load_sampled_data(language: str) -> List[Dict]:
    """Load pre-sampled data from data/sampled/<language>.json."""
    filepath = os.path.join(SAMPLED_DIR, f"{language}.json")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"No sampled data for '{language}'. Run `py prepare_data.py` first.\n"
            f"  Expected: {filepath}"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    metadata = data["metadata"]
    logger.info(
        f"[{language}] Loaded {len(samples)} pre-sampled posts "
        f"(seed={metadata.get('seed')}, created={metadata.get('created')})"
    )
    return samples


# ── Interactive menus ────────────────────────────────────────────────────────

def print_header():
    print("\n" + "=" * 60)
    print("  🧠 Multilingual Mental Health — LLM Evaluation Pipeline")
    print("=" * 60)


def select_models() -> List[str]:
    """Interactive model selection menu."""
    print("\n📋 Available Models:")
    print("-" * 40)
    model_keys = list(MODELS.keys())
    for i, key in enumerate(model_keys, 1):
        info = MODELS[key]
        print(f"  {i}. {info['name']}  ({info['default_model']})")
    print(f"  {len(model_keys) + 1}. All models")
    print()

    while True:
        choice = input("Select model(s) (comma-separated numbers, e.g. 1,2): ").strip()
        if not choice:
            continue

        if choice == str(len(model_keys) + 1):
            selected = model_keys
            break

        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(model_keys):
                    selected.append(model_keys[idx - 1])
                else:
                    print(f"  ⚠ Invalid number: {idx}")
                    selected = []
                    break
            if selected:
                break
        except ValueError:
            print("  ⚠ Please enter numbers separated by commas.")

    print(f"  ✓ Selected: {', '.join(MODELS[k]['name'] for k in selected)}")
    return selected


def select_languages(available: Dict[str, str]) -> List[str]:
    """Interactive language selection menu."""
    lang_keys = list(available.keys())

    print("\n🌍 Available Languages (pre-sampled):")
    print("-" * 40)
    for i, lang in enumerate(lang_keys, 1):
        # Show sample count
        try:
            with open(available[lang], "r", encoding="utf-8") as f:
                data = json.load(f)
            count = data["metadata"]["sample_size"]
            seed = data["metadata"]["seed"]
            print(f"  {i}. {lang.capitalize()}  ({count} samples, seed={seed})")
        except Exception:
            print(f"  {i}. {lang.capitalize()}")
    print(f"  {len(lang_keys) + 1}. All languages (cross-lingual)")
    print()

    while True:
        choice = input("Select language(s) (comma-separated numbers, e.g. 1,3): ").strip()
        if not choice:
            continue

        if choice == str(len(lang_keys) + 1):
            selected = list(lang_keys)
            break

        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(lang_keys):
                    selected.append(lang_keys[idx - 1])
                else:
                    print(f"  ⚠ Invalid number: {idx}")
                    selected = []
                    break
            if selected:
                break
        except ValueError:
            print("  ⚠ Please enter numbers separated by commas.")

    print(f"  ✓ Selected: {', '.join(l.capitalize() for l in selected)}")
    return selected


# ── Evaluation logic ─────────────────────────────────────────────────────────

SAVE_EVERY = 10  # Save partial results every N classifications


def _partial_path(model_name: str, language: str) -> str:
    """Return the path for a partial (in-progress) results file."""
    return os.path.join(RESULTS_DIR, f"{model_name}_{language}.partial.json")


def _save_partial(results: List[Dict], model_name: str, language: str,
                  total_samples: int) -> str:
    """Save partial results to disk so we can resume after a crash."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = _partial_path(model_name, language)
    output = {
        "metadata": {
            "model": model_name,
            "language": language,
            "status": "partial",
            "completed": len(results),
            "total": total_samples,
            "last_saved": datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return filepath


def _load_partial(model_name: str, language: str) -> List[Dict]:
    """Load partial results if they exist. Returns empty list if none."""
    filepath = _partial_path(model_name, language)
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        completed = data["metadata"]["completed"]
        total = data["metadata"]["total"]
        logger.info(f"Found partial results: {completed}/{total} completed")
        return results
    except Exception as e:
        logger.warning(f"Could not load partial results: {e}")
        return []


def _remove_partial(model_name: str, language: str):
    """Remove the partial results file after successful completion."""
    filepath = _partial_path(model_name, language)
    if os.path.isfile(filepath):
        os.remove(filepath)


def run_evaluation(provider, samples: List[Dict], language: str,
                   delay: float = 1.0, fresh: bool = False) -> List[Dict]:
    """
    Run classification on all samples, with crash-resume support.

    Saves partial results every SAVE_EVERY classifications.
    If partial results exist from a previous run, resumes from there.
    """
    model_name = provider.name if hasattr(provider, 'name') else str(provider)
    total = len(samples)

    # Check for partial results to resume from
    if fresh:
        _remove_partial(model_name, language)
        print(f"  🔄 Fresh run — ignoring any previous partial results")
        results = []
        start_idx = 0
    else:
        existing = _load_partial(model_name, language)
        start_idx = len(existing)

        if start_idx > 0 and start_idx < total:
            print(f"  🔄 Resuming from {start_idx}/{total} (found partial results)")
            results = existing
        elif start_idx >= total:
            print(f"  ✓ Already completed {total}/{total} — skipping")
            return existing
        else:
            results = []
            start_idx = 0

    for i in range(start_idx, total):
        sample = samples[i]
        idx = i + 1
        logger.info(f"[{model_name}][{language}] Classifying {idx}/{total}...")

        result = provider.classify(
            post=sample["post"],
            prompt_template=CLASSIFICATION_PROMPT,
        )

        results.append({
            "index": idx,
            "post_preview": sample["post"][:100] + ("..." if len(sample["post"]) > 100 else ""),
            "post_full": sample["post"],
            "ground_truth": sample["ground_truth"],
            "prediction": result["prediction"],
            "raw_response": result["raw_response"],
            "error": result["error"],
            "word_count": sample.get("word_count", 0),
        })

        # Save partial results periodically
        if idx % SAVE_EVERY == 0:
            _save_partial(results, model_name, language, total)
            logger.info(f"  💾 Checkpoint: {idx}/{total} saved")

        if idx < total:
            time.sleep(delay)

    # Final save — remove partial file
    _remove_partial(model_name, language)
    return results


def save_results(results: List[Dict], metrics: Dict, model_name: str,
                 language: str, output_dir: str = None) -> str:
    """Save final evaluation results and metrics to JSON."""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{language}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)


    output = {
        "metadata": {
            "model": model_name,
            "language": language,
            "timestamp": timestamp,
            "sample_size": len(results),
        },
        "metrics": metrics,
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Run LLM evaluation on pre-sampled datasets"
    )
    parser.add_argument("--reparse", action="store_true",
                        help="Re-run prepare_data.py before evaluation")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore partial results and start fresh (overwrite)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls (default 1.0, use 21 for OpenAI free tier)")
    args = parser.parse_args()

    # If --reparse, run prepare_data first
    if args.reparse:
        print("\n🔄 Re-parsing data first...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "prepare_data.py", "--reparse"],
            cwd=os.path.dirname(__file__) or ".",
        )
        if result.returncode != 0:
            print("  ❌ Data preparation failed. Aborting.")
            return

    print_header()

    # Check for sampled data
    available = discover_sampled_languages()
    if not available:
        print("\n⚠ No sampled data found in data/sampled/")
        print("  Run `py prepare_data.py` first to parse and sample datasets.")
        return

    # 1. Select model(s)
    selected_models = select_models()

    # 2. Select language(s)
    selected_languages = select_languages(available)

    # 3. Confirm
    print(f"\n{'='*60}")
    print(f"  📋 Evaluation Plan:")
    print(f"     Models:     {', '.join(MODELS[k]['name'] for k in selected_models)}")
    print(f"     Languages:  {', '.join(l.capitalize() for l in selected_languages)}")
    print(f"{'='*60}")

    confirm = input("\nProceed? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        return

    # 4. Run evaluations
    all_output_files = []
    all_metrics = {}  # (model_name, language) -> metrics

    for model_key in selected_models:
        info = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"  🚀 Evaluating: {info['name']}")
        print(f"{'='*60}")

        try:
            api_key = get_api_key(model_key)
        except EnvironmentError as e:
            print(f"  ❌ {e}")
            print(f"  Skipping {info['name']}.\n")
            continue

        provider = info["class"](api_key=api_key, model_name=info["default_model"])

        for lang in selected_languages:
            print(f"\n  📂 Language: {lang.capitalize()}")
            try:
                samples = load_sampled_data(lang)
            except FileNotFoundError as e:
                print(f"  ❌ {e}")
                continue

            results = run_evaluation(provider, samples, lang,
                                      delay=args.delay, fresh=args.fresh)
            metrics = EvaluationMetrics.compute(results)

            # Print report
            report = EvaluationMetrics.format_report(metrics, f"{info['name']} — {lang.capitalize()}")
            print(report)

            # Save
            filepath = save_results(results, metrics, model_key, lang)
            all_output_files.append(filepath)
            all_metrics[(info["name"], lang)] = metrics
            print(f"  💾 Saved: {filepath}")

    # 5. Cross-lingual comparison (per model, if multiple languages)
    if len(selected_languages) > 1:
        for model_key in selected_models:
            model_name = MODELS[model_key]["name"]
            lang_metrics = {
                lang: m for (mn, lang), m in all_metrics.items() if mn == model_name
            }
            if len(lang_metrics) > 1:
                print(f"\n{'='*70}")
                print(f"  CROSS-LINGUAL COMPARISON: {model_name}")
                print(f"{'='*70}")
                print(f"  {'Language':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
                print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                for lang, m in lang_metrics.items():
                    print(
                        f"  {lang.capitalize():<12} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                        f"{m['recall']:>8.4f} {m['f1_score']:>8.4f}"
                    )
                print(f"{'='*70}")

    # 6. Multi-model comparison (per language, if multiple models)
    if len(selected_models) > 1:
        for lang in selected_languages:
            model_metrics = {
                mn: m for (mn, l), m in all_metrics.items() if l == lang
            }
            if len(model_metrics) > 1:
                comparison = EvaluationMetrics.comparison_table(model_metrics)
                print(f"\n  📊 Model Comparison — {lang.upper()}:")
                print(comparison)

        # Save combined comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.json")
        comparison_data = {}
        for (mn, lang), m in all_metrics.items():
            comparison_data.setdefault(mn, {})[lang] = m
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump({"models": comparison_data}, f, indent=2)
        all_output_files.append(comp_path)
        print(f"  💾 Comparison saved: {comp_path}")

    # 7. Summary
    print(f"\n{'='*60}")
    print(f"  ✅ Evaluation complete!")
    print(f"{'='*60}")
    print(f"  Output files:")
    for f in all_output_files:
        print(f"    → {f}")
    print()


if __name__ == "__main__":
    main()
