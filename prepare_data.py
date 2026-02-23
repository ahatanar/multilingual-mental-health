"""
Part 1: Parse raw datasets and create sampled subsets.

Run:
    py prepare_data.py                  # parse + sample all languages
    py prepare_data.py --reparse        # force re-parse from raw files
    py prepare_data.py --languages english arabic   # specific languages only
    py prepare_data.py --sample-size 100            # custom sample size

Outputs:
    data/cleaned/<language>.json   — full parsed dataset (cached)
    data/sampled/<language>.json   — stratified sampled subset for evaluation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict

from evaluation.parsers import get_parser, PARSERS
from evaluation.sampler import DatasetSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLED_DIR = os.path.join(DATA_DIR, "sampled")


def prepare_language(language: str, sample_size: int = 500, seed: int = 42,
                     reparse: bool = False) -> Dict:
    """Parse and sample a single language dataset."""
    print(f"\n{'─'*50}")
    print(f"  📦 Preparing: {language.upper()}")
    print(f"{'─'*50}")

    # 1. Parse
    parser = get_parser(language, DATA_DIR)
    data = parser.get_data(reparse=reparse)
    print(f"  ✓ Parsed {len(data)} total posts → {parser.cache_path()}")

    # 2. Sample
    sampler = DatasetSampler(data, language=language)
    stats = sampler.get_stats()
    print(f"  ℹ Stats: {stats}")

    samples = sampler.sample(n=sample_size, seed=seed)

    gt_dist = {}
    for s in samples:
        gt_dist[s["ground_truth"]] = gt_dist.get(s["ground_truth"], 0) + 1
    print(f"  ✓ Sampled {len(samples)} posts — {gt_dist}")

    # 3. Save sampled subset
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    sampled_path = os.path.join(SAMPLED_DIR, f"{language}.json")

    export = {
        "metadata": {
            "language": language,
            "sample_size": len(samples),
            "requested_size": sample_size,
            "seed": seed,
            "created": datetime.now().isoformat(),
            "source_total": len(data),
            "label_distribution": gt_dist,
        },
        "samples": [
            {
                "post": s["post"],
                "label": s["label"],
                "ground_truth": s["ground_truth"],
                "word_count": s.get("word_count", 0),
            }
            for s in samples
        ],
    }

    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved sampled data → {sampled_path}")

    return {
        "language": language,
        "total_parsed": len(data),
        "sampled": len(samples),
        "distribution": gt_dist,
        "sampled_path": sampled_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Part 1: Parse and sample multilingual mental health datasets"
    )
    parser.add_argument("--reparse", action="store_true",
                        help="Force re-parsing of raw data (ignore cache)")
    parser.add_argument("--languages", nargs="+", default=None,
                        choices=list(PARSERS.keys()),
                        help="Languages to prepare (default: all)")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of samples per language (default: 500, split 250/250)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    languages = args.languages or list(PARSERS.keys())

    print("\n" + "=" * 50)
    print("  📋 Data Preparation Pipeline")
    print("=" * 50)
    print(f"  Languages:    {', '.join(l.capitalize() for l in languages)}")
    print(f"  Sample size:  {args.sample_size} per language")
    print(f"  Seed:         {args.seed}")
    print(f"  Re-parse:     {'Yes' if args.reparse else 'No (use cache)'}")

    results = []
    for lang in languages:
        try:
            result = prepare_language(
                language=lang,
                sample_size=args.sample_size,
                seed=args.seed,
                reparse=args.reparse,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"[{lang}] Failed: {e}")
            print(f"  ❌ {lang}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  ✅ Data preparation complete!")
    print(f"{'='*50}")
    print(f"  Cached parsed data:  data/cleaned/")
    print(f"  Sampled subsets:     data/sampled/")
    print()
    for r in results:
        print(f"    {r['language']:<10}  {r['total_parsed']:>6} parsed → {r['sampled']:>4} sampled  {r['distribution']}")
    print()
    print(f"  Next step: run `py runner.py` to evaluate with LLMs")
    print()


if __name__ == "__main__":
    main()
