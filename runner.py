"""
Interactive evaluation runner for multilingual mental health LLM evaluation.

Run:
    python3 runner.py

Presents interactive menus to select model(s) and dataset, then runs
the evaluation and saves results to results/.
"""

import json
import logging
import os
import sys
import time
import glob
from datetime import datetime
from typing import List, Dict

from config import get_api_key, DEFAULT_MODELS
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider
from evaluation.sampler import DatasetSampler
from evaluation.metrics import EvaluationMetrics
from evaluation.prompts import CLASSIFICATION_PROMPT

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
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Interactive menus ────────────────────────────────────────────────────────

def discover_datasets() -> Dict[str, str]:
    """Scan data/ directory for CSV files, organized by language folder."""
    datasets = {}
    if not os.path.isdir(DATA_DIR):
        return datasets

    for lang_dir in sorted(os.listdir(DATA_DIR)):
        lang_path = os.path.join(DATA_DIR, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        for csv_file in sorted(glob.glob(os.path.join(lang_path, "*.csv"))):
            label = f"{lang_dir}/{os.path.basename(csv_file)}"
            datasets[label] = csv_file

    return datasets


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


def select_dataset() -> str:
    """Interactive dataset selection menu."""
    datasets = discover_datasets()

    if not datasets:
        print("\n⚠ No datasets found in data/ directory!")
        print("  Place CSV files in data/<language>/ folders.")
        sys.exit(1)

    print("\n📂 Available Datasets:")
    print("-" * 40)
    dataset_keys = list(datasets.keys())
    for i, label in enumerate(dataset_keys, 1):
        # Show row count
        try:
            with open(datasets[label], "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1  # minus header
        except Exception:
            row_count = "?"
        print(f"  {i}. {label}  ({row_count} rows)")
    print()

    while True:
        choice = input("Select dataset (number): ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(dataset_keys):
                selected = dataset_keys[idx - 1]
                break
            else:
                print(f"  ⚠ Enter a number between 1 and {len(dataset_keys)}")
        except ValueError:
            print("  ⚠ Please enter a number.")

    print(f"  ✓ Selected: {selected}")
    return datasets[selected], selected


def select_sample_size() -> int:
    """Ask for sample size."""
    print()
    while True:
        choice = input("Sample size (default 50): ").strip()
        if not choice:
            print("  ✓ Using default: 50")
            return 50
        try:
            n = int(choice)
            if n > 0:
                print(f"  ✓ Sample size: {n}")
                return n
            print("  ⚠ Must be a positive number.")
        except ValueError:
            print("  ⚠ Please enter a number.")


# ── Evaluation logic ─────────────────────────────────────────────────────────

def run_evaluation(provider, samples: List[Dict], delay: float = 1.0) -> List[Dict]:
    """Run classification on all samples using the given provider."""
    results = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        logger.info(f"[{provider.name}] Classifying {i}/{total}...")

        result = provider.classify(
            post=sample["post"],
            prompt_template=CLASSIFICATION_PROMPT,
        )

        results.append({
            "index": i,
            "post_preview": sample["post"][:100] + ("..." if len(sample["post"]) > 100 else ""),
            "post_full": sample["post"],
            "ground_truth": sample["ground_truth"],
            "prediction": result["prediction"],
            "raw_response": result["raw_response"],
            "error": result["error"],
            "word_count": sample.get("word_count", 0),
        })

        if i < total:
            time.sleep(delay)

    return results


def save_results(results: List[Dict], metrics: Dict, model_name: str,
                 dataset_label: str, output_dir: str = "results") -> str:
    """Save evaluation results and metrics to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = dataset_label.replace("/", "_").replace(".csv", "")
    filename = f"{model_name}_{safe_dataset}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "metadata": {
            "model": model_name,
            "dataset": dataset_label,
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
    print_header()

    # 1. Select model(s)
    selected_models = select_models()

    # 2. Select dataset
    dataset_path, dataset_label = select_dataset()

    # 3. Select sample size
    sample_size = select_sample_size()

    # 4. Load and sample
    print(f"\n🔄 Loading dataset...")
    sampler = DatasetSampler(dataset_path)
    stats = sampler.get_stats()
    print(f"  Total eligible posts: {stats['total_posts']}")
    print(f"  Label distribution:   {stats['label_distribution']}")
    print(f"  Avg word count:       {stats['avg_word_count']}")

    samples = sampler.sample(n=sample_size, seed=42)
    gt_dist = {}
    for s in samples:
        gt_dist[s["ground_truth"]] = gt_dist.get(s["ground_truth"], 0) + 1
    print(f"  Sampled: {len(samples)} posts — {gt_dist}")

    # 5. Run evaluations
    all_metrics = {}
    all_files = []

    for model_key in selected_models:
        info = MODELS[model_key]
        print(f"\n{'='*50}")
        print(f"  🚀 Evaluating: {info['name']}")
        print(f"{'='*50}")

        try:
            api_key = get_api_key(model_key)
        except EnvironmentError as e:
            print(f"  ❌ {e}")
            print(f"  Skipping {info['name']}.\n")
            continue

        provider = info["class"](api_key=api_key, model_name=info["default_model"])
        results = run_evaluation(provider, samples)
        metrics = EvaluationMetrics.compute(results)

        # Print report
        report = EvaluationMetrics.format_report(metrics, info["name"])
        print(report)

        # Save
        filepath = save_results(results, metrics, model_key, dataset_label)
        all_files.append(filepath)
        all_metrics[info["name"]] = metrics
        print(f"  💾 Saved: {filepath}")

    # 6. Comparison (if multiple models)
    if len(all_metrics) > 1:
        comparison = EvaluationMetrics.comparison_table(all_metrics)
        print(comparison)

        comp_path = os.path.join("results", f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(comp_path, "w") as f:
            json.dump({"models": {k: v for k, v in all_metrics.items()}}, f, indent=2)
        all_files.append(comp_path)
        print(f"  💾 Comparison saved: {comp_path}")

    # 7. Summary
    print(f"\n{'='*50}")
    print(f"  ✅ Evaluation complete!")
    print(f"{'='*50}")
    print(f"  Output files:")
    for f in all_files:
        print(f"    → {f}")
    print()


if __name__ == "__main__":
    main()
