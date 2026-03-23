"""
Prepare the Phase 2 Arabic dataset for translation and review.

Reads the raw CairoDep CSV directly (bypassing ArabicParser) so we
can preserve the two columns that the shared parser drops:
  - dialect  ("slang" | "standard arabic")
  - source   ("crowdsourcing" | "facebook" | "Nafsany" | "reddit" |
               "semEval" | "Twitter_API" | "Data_World")

Creates a stratified 6 000-post sample (3 000 depressed + 3 000 normal)
that will be fed into translate_arabic.py.

Usage:
    python scripts/phase2/prepare_arabic.py
    python scripts/phase2/prepare_arabic.py --samples 100   # smaller test set
    python scripts/phase2/prepare_arabic.py --seed 99       # different seed
"""

import argparse
import csv
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV   = PROJECT_ROOT / "data" / "raw" / "arabic" / "CairoDep_Datasets.csv"
OUT_DIR   = PROJECT_ROOT / "data" / "phase2"
OUT_FILE  = OUT_DIR / "arabic_{n}samples_seed{seed}.json"

DEFAULT_SAMPLES = 6000
DEFAULT_SEED    = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> List[Dict]:
    """Read CairoDep CSV preserving all four columns."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            post    = row.get("post", "").strip()
            label   = row.get("label", "").strip().lower()
            dialect = row.get("dialect", "").strip()
            source  = row.get("source", "").strip()
            if not post or label not in ("depression", "normal"):
                continue
            rows.append({
                "post":    post,
                "label":   label,
                "dialect": dialect,
                "source":  source,
            })
    return rows


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


def print_breakdown(samples: List[Dict]) -> None:
    from collections import Counter

    labels   = Counter(s["label"]   for s in samples)
    dialects = Counter(s["dialect"] for s in samples)
    sources  = Counter(s["source"]  for s in samples)

    print(f"\n  Labels   : {dict(labels)}")
    print(f"  Dialects : {dict(dialects)}")
    print(f"  Sources  : {dict(sources)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 Arabic sample set for translation."
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

    # Load
    logger.info(f"Reading {RAW_CSV} …")
    data = load_raw(RAW_CSV)
    logger.info(f"Loaded {len(data)} valid rows from CairoDep CSV")

    # Sample
    samples = stratified_sample(data, args.samples, args.seed)
    logger.info(f"Sampled {len(samples)} posts (seed={args.seed})")

    # Output path
    out_path = OUT_DIR / f"arabic_{len(samples)}samples_seed{args.seed}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "language":        "arabic",
            "dataset":         "CairoDep",
            "seed":            args.seed,
            "total_available": len(data),
            "total_sampled":   len(samples),
            "depressed_count": sum(1 for s in samples if s["label"] == "depression"),
            "normal_count":    sum(1 for s in samples if s["label"] == "normal"),
            "columns":         ["index", "post", "label", "dialect", "source"],
            "created":         datetime.now().isoformat(),
        },
        "samples": samples,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  Phase 2 Arabic dataset ready")
    print(f"  Output : {out_path}")
    print_breakdown(samples)
    print(f"  Next   : python scripts/phase2/translate_arabic.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
