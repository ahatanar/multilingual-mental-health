"""
Prepare the Phase 2 Chinese dataset for translation and ethics review.

Reads from the cleaned Weibo cache (data/cleaned/chinese.json) via ChineseParser.
Creates a stratified 6 000-post sample (3 000 depressed + 3 000 normal) that will
be fed into translate_chinese.py for ethics review before final evaluation.

Usage:
    python scripts/phase2/prepare_chinese.py
    python scripts/phase2/prepare_chinese.py --samples 100   # smaller test set
    python scripts/phase2/prepare_chinese.py --seed 99       # different seed
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.parsers import ChineseParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "data" / "phase2"

DEFAULT_SAMPLES = 6000
DEFAULT_SEED    = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cleaned() -> List[Dict]:
    """
    Load the cleaned Chinese dataset via ChineseParser.

    Uses data/cleaned/chinese.json cache if present (recommended — avoids
    re-parsing the 5 GB raw JSONL files).  If the cache is missing the parser
    will fall back to data/raw/chinese/depressed.jsonl + control.jsonl.
    """
    parser = ChineseParser(data_dir=str(PROJECT_ROOT / "data"))
    cache  = PROJECT_ROOT / "data" / "cleaned" / "chinese.json"
    raw_dep = PROJECT_ROOT / "data" / "raw" / "chinese" / "depressed.jsonl"

    if not cache.exists() and not raw_dep.exists():
        logger.error(
            "Neither cleaned cache nor raw files found.\n"
            f"  Cache expected at : {cache}\n"
            f"  Raw files expected: {raw_dep.parent}"
        )
        sys.exit(1)

    data = parser.get_data()   # loads cache when available
    logger.info(f"Loaded {len(data)} cleaned posts")
    return data


def stratified_sample(data: List[Dict], n_total: int, seed: int) -> List[Dict]:
    """Return n_total posts split evenly between depressed and normal."""
    rng = random.Random(seed)

    depressed = [r for r in data if r["label"] == "depression"]
    normal    = [r for r in data if r["label"] == "normal"]

    n_dep  = n_total // 2
    n_norm = n_total - n_dep

    if len(depressed) < n_dep:
        logger.warning(f"Only {len(depressed)} depressed posts (wanted {n_dep})")
        n_dep = len(depressed)
    if len(normal) < n_norm:
        logger.warning(f"Only {len(normal)} normal posts (wanted {n_norm})")
        n_norm = len(normal)

    sampled = rng.sample(depressed, n_dep) + rng.sample(normal, n_norm)
    rng.shuffle(sampled)

    # Attach a stable index for reference during translation
    for i, entry in enumerate(sampled):
        entry["index"] = i

    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 Chinese sample set for translation."
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Number of posts to sample (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    data    = load_cleaned()
    samples = stratified_sample(data, args.samples, args.seed)
    logger.info(f"Sampled {len(samples)} posts (seed={args.seed})")

    dep_count  = sum(1 for s in samples if s["label"] == "depression")
    norm_count = sum(1 for s in samples if s["label"] == "normal")

    out_path = OUT_DIR / f"chinese_{len(samples)}samples_seed{args.seed}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "language":        "chinese",
            "dataset":         "Weibo Depression Dataset",
            "seed":            args.seed,
            "total_available": len(data),
            "total_sampled":   len(samples),
            "depressed_count": dep_count,
            "normal_count":    norm_count,
            "columns":         ["index", "post", "label"],
            "note":            "is_origin=True posts only, HTML tags stripped",
            "created":         datetime.now().isoformat(),
        },
        "samples": samples,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  Phase 2 Chinese dataset ready")
    print(f"  Output : {out_path}")
    print(f"\n  Labels : depressed={dep_count}, normal={norm_count}")
    print(f"\n  Next   : python scripts/phase2/translate_chinese.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
