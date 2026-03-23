"""
Cross-lingual evaluation orchestrator.

Runs evaluation across multiple languages for a given model, collects
per-language metrics, and produces a combined comparison table.
Results are saved both individually (per-language) and combined.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

from .parsers import get_parser, PARSERS
from .sampler import DatasetSampler
from .metrics import EvaluationMetrics
from .prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SAMPLED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "phase1", "sampled")


def _save_sampled_data(samples: List[Dict], language: str, sample_size: int, seed: int) -> str:
    """Save the sampled dataset to data/sampled/ for reproducibility."""
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    filename = f"{language}_{sample_size}_seed{seed}.json"
    filepath = os.path.join(SAMPLED_DIR, filename)

    export = {
        "metadata": {
            "language": language,
            "sample_size": len(samples),
            "seed": seed,
            "created": datetime.now().isoformat(),
        },
        "samples": [
            {"post": s["post"], "label": s["label"], "ground_truth": s["ground_truth"],
             "word_count": s.get("word_count", 0)}
            for s in samples
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    logger.info(f"[{language}] Saved {len(samples)} sampled posts to {filepath}")
    return filepath


def run_single_language(
    provider,
    language: str,
    sample_size: int = 500,
    seed: int = 42,
    reparse: bool = False,
    delay: float = 1.0,
) -> Dict:
    """
    Run evaluation for a single language with a given provider.

    Returns dict with keys: language, results, metrics, stats, output_file.
    """
    logger.info(f"--- [{language.upper()}] Starting evaluation with {provider.name} ---")

    # 1. Parse data
    parser = get_parser(language, DATA_DIR)
    data = parser.get_data(reparse=reparse)
    logger.info(f"[{language}] Parsed {len(data)} total posts")

    # 2. Sample
    sampler = DatasetSampler(data, language=language)
    stats = sampler.get_stats()
    logger.info(f"[{language}] Stats: {stats}")

    samples = sampler.sample(n=sample_size, seed=seed)
    gt_dist = {}
    for s in samples:
        gt_dist[s["ground_truth"]] = gt_dist.get(s["ground_truth"], 0) + 1
    logger.info(f"[{language}] Sampled {len(samples)} posts — {gt_dist}")

    # 2b. Save sampled data for reproducibility
    sampled_file = _save_sampled_data(samples, language, sample_size, seed)
    logger.info(f"[{language}] Sampled dataset saved to {sampled_file}")

    # 3. Classify
    results = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        logger.info(f"[{provider.name}][{language}] Classifying {i}/{total}...")
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

    # 4. Compute metrics
    metrics = EvaluationMetrics.compute(results)

    # 5. Save individual results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{provider.name}_{language}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    output = {
        "metadata": {
            "model": provider.name,
            "language": language,
            "timestamp": timestamp,
            "sample_size": len(results),
        },
        "stats": stats,
        "metrics": metrics,
        "results": results,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"[{language}] Saved results to {filepath}")

    return {
        "language": language,
        "results": results,
        "metrics": metrics,
        "stats": stats,
        "output_file": filepath,
    }


def run_cross_lingual(
    provider,
    languages: Optional[List[str]] = None,
    sample_size: int = 500,
    seed: int = 42,
    reparse: bool = False,
    delay: float = 1.0,
) -> Dict:
    """
    Run evaluation across multiple languages and save combined results.

    Args:
        provider: An LLM provider instance.
        languages: List of language keys to evaluate. Defaults to all.
        sample_size: Samples per language.
        seed: Random seed.
        reparse: Force re-parsing of cached data.
        delay: Seconds between API calls.

    Returns:
        Dict with per-language results and combined comparison.
    """
    if languages is None:
        languages = list(PARSERS.keys())

    all_results = {}
    all_files = []

    for lang in languages:
        try:
            result = run_single_language(
                provider=provider,
                language=lang,
                sample_size=sample_size,
                seed=seed,
                reparse=reparse,
                delay=delay,
            )
            all_results[lang] = result
            all_files.append(result["output_file"])
        except Exception as e:
            logger.error(f"[{lang}] Evaluation failed: {e}")
            all_results[lang] = {"language": lang, "error": str(e)}

    # Save combined cross-lingual comparison
    if len(all_results) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"{provider.name}_cross_lingual_{timestamp}.json"
        combined_path = os.path.join(RESULTS_DIR, combined_filename)

        combined = {
            "metadata": {
                "model": provider.name,
                "type": "cross_lingual",
                "languages": languages,
                "timestamp": timestamp,
                "sample_size_per_language": sample_size,
            },
            "per_language": {},
        }

        for lang, result in all_results.items():
            if "error" in result and "metrics" not in result:
                combined["per_language"][lang] = {"error": result["error"]}
            else:
                combined["per_language"][lang] = {
                    "metrics": result["metrics"],
                    "stats": result["stats"],
                    "individual_file": result["output_file"],
                }

        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved cross-lingual comparison to {combined_path}")
        all_files.append(combined_path)

    return {
        "per_language": all_results,
        "output_files": all_files,
    }


def format_cross_lingual_report(cross_results: Dict, model_name: str) -> str:
    """Format cross-lingual results into a readable comparison table."""
    per_lang = cross_results["per_language"]

    lines = [
        f"\n{'='*70}",
        f"  CROSS-LINGUAL REPORT: {model_name}",
        f"{'='*70}",
        f"  {'Language':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Samples':>8} {'Errors':>8}",
        f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ]

    for lang, result in per_lang.items():
        if "error" in result and "metrics" not in result:
            lines.append(f"  {lang:<12} {'FAILED':>8} — {result['error']}")
            continue
        m = result["metrics"]
        lines.append(
            f"  {lang:<12} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} "
            f"{m['total_samples']:>8d} {m['error_responses']:>8d}"
        )

    lines.append(f"{'='*70}\n")
    return "\n".join(lines)
