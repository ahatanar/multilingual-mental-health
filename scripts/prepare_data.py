"""
Prepare Phase 2 Experiment 1 sample files.

Creates 5 000-post (2 500 depressed + 2 500 normal) evaluation-ready datasets
for Arabic, Urdu, and Chinese.

Outputs:
    data/phase2/arabic_5000samples_seed42.json
    data/phase2/urdu_5000samples_seed42.json
    data/phase2/chinese_5000samples_seed42.json

Usage:
    python scripts/prepare_data.py              # all languages
    python scripts/prepare_data.py --lang arabic
    python scripts/prepare_data.py --lang urdu
    python scripts/prepare_data.py --lang chinese
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.parsers import UrduParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_TOTAL  = 5000
N_EACH   = N_TOTAL // 2
SEED     = 42

PHASE2_DIR     = PROJECT_ROOT / "data" / "phase2"
ARABIC_SOURCE  = PHASE2_DIR / "translated" / "filtered" / "arabic_6000samples_seed42_filtered.json"
CHINESE_SOURCE = PHASE2_DIR / "translated" / "filtered" / "chinese_6000samples_seed42_filtered.json"
URDU_RAW_DIR   = str(PROJECT_ROOT / "data")


def _ground_truth(label: str) -> str:
    return "depressed" if label == "depression" else "not depressed"


def _stratified_sample(items: List[Dict], n_each: int, seed: int) -> List[Dict]:
    rng       = random.Random(seed)
    depressed = [x for x in items if x["label"] == "depression"]
    normal    = [x for x in items if x["label"] == "normal"]
    if len(depressed) < n_each:
        logger.warning(f"Only {len(depressed)} depressed available (wanted {n_each})")
        n_each = min(n_each, len(depressed))
    if len(normal) < n_each:
        logger.warning(f"Only {len(normal)} normal available (wanted {n_each})")
        n_each = min(n_each, len(normal))
    selected = rng.sample(depressed, n_each) + rng.sample(normal, n_each)
    rng.shuffle(selected)
    return selected


def _save(samples: List[Dict], metadata: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "samples": samples}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(samples)} samples to {out_path}")


def prepare_arabic(n_each: int = N_EACH, seed: int = SEED) -> Optional[Path]:
    """Sample from the filtered Arabic translation file. Maps 'original' → 'post'."""
    if not ARABIC_SOURCE.exists():
        logger.error(f"Filtered Arabic file not found: {ARABIC_SOURCE}")
        return None
    with open(ARABIC_SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    raw = [
        {
            "index":        s["index"],
            "post":         s["original"],
            "ground_truth": _ground_truth(s["label"]),
            "label":        s["label"],
            "translation":  s.get("translation", ""),
            "dialect":      s.get("dialect", ""),
            "source":       s.get("source", ""),
        }
        for s in data["samples"]
    ]
    selected   = _stratified_sample(raw, n_each, seed)
    dep_count  = sum(1 for s in selected if s["label"] == "depression")
    norm_count = sum(1 for s in selected if s["label"] == "normal")
    logger.info(f"[Arabic] Sampled {len(selected)}: {dep_count} depressed, {norm_count} normal")
    out_path = PHASE2_DIR / f"arabic_{len(selected)}samples_seed{seed}.json"
    _save(selected, {
        "language": "arabic", "experiment": 1,
        "dataset": "CairoDep (translated + filtered)", "seed": seed,
        "total_available": len(raw), "total_sampled": len(selected),
        "depressed_count": dep_count, "normal_count": norm_count,
        "created": datetime.now().isoformat(),
    }, out_path)
    return out_path


def prepare_urdu(n_each: int = N_EACH, seed: int = SEED) -> Optional[Path]:
    """Sample from the raw Urdu dataset via UrduParser."""
    parser = UrduParser(data_dir=URDU_RAW_DIR)
    try:
        raw = parser.parse()
    except FileNotFoundError as e:
        logger.error(f"Urdu dataset not found: {e}")
        return None
    for entry in raw:
        entry["word_count"]   = len(entry["post"].split())
        entry["ground_truth"] = _ground_truth(entry["label"])
    selected   = _stratified_sample(raw, n_each, seed)
    dep_count  = sum(1 for s in selected if s["label"] == "depression")
    norm_count = sum(1 for s in selected if s["label"] == "normal")
    logger.info(f"[Urdu] Sampled {len(selected)}: {dep_count} depressed, {norm_count} normal")
    out_path = PHASE2_DIR / f"urdu_{len(selected)}samples_seed{seed}.json"
    _save(selected, {
        "language": "urdu", "experiment": 1,
        "dataset": "Urdu Depression Dataset (Roman Urdu)", "seed": seed,
        "total_available": len(raw), "total_sampled": len(selected),
        "depressed_count": dep_count, "normal_count": norm_count,
        "label_map": "mild/moderate/severe → depressed | non-depression → not depressed",
        "created": datetime.now().isoformat(),
    }, out_path)
    return out_path


def prepare_chinese(n_each: int = N_EACH, seed: int = SEED) -> Optional[Path]:
    """Sample from the ethics-reviewed Chinese translation file."""
    if not CHINESE_SOURCE.exists():
        logger.error(f"Filtered Chinese file not found: {CHINESE_SOURCE}")
        return None
    with open(CHINESE_SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    raw = [
        {
            "index":        s["index"],
            "post":         s["original"],
            "ground_truth": _ground_truth(s["label"]),
            "label":        s["label"],
            "translation":  s.get("translation", ""),
            "word_count":   len(s["original"]),
        }
        for s in data["samples"]
    ]
    selected   = _stratified_sample(raw, n_each, seed)
    dep_count  = sum(1 for s in selected if s["label"] == "depression")
    norm_count = sum(1 for s in selected if s["label"] == "normal")
    logger.info(f"[Chinese] Sampled {len(selected)}: {dep_count} depressed, {norm_count} normal")
    out_path = PHASE2_DIR / f"chinese_{len(selected)}samples_seed{seed}.json"
    _save(selected, {
        "language": "chinese", "experiment": 1,
        "dataset": "Weibo Depression Dataset (translated + filtered)", "seed": seed,
        "total_available": len(raw), "total_sampled": len(selected),
        "depressed_count": dep_count, "normal_count": norm_count,
        "word_count_note": "Character count (CJK text has no word spaces)",
        "created": datetime.now().isoformat(),
    }, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 2 Experiment 1 sample files (5 000 posts per language)."
    )
    parser.add_argument("--lang", choices=["arabic", "urdu", "chinese", "all"], default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n",   type=int, default=N_TOTAL, help=f"Total samples per language (default: {N_TOTAL})")
    args   = parser.parse_args()
    n_each = args.n // 2
    langs  = ["arabic", "urdu", "chinese"] if args.lang == "all" else [args.lang]

    print(f"\n{'='*58}")
    print(f"  Phase 2 — Experiment 1 Data Preparation")
    print(f"  Target: {args.n} samples ({n_each}+{n_each}) per language, seed={args.seed}")
    print(f"{'='*58}\n")

    results = {}
    if "arabic"  in langs: results["arabic"]  = prepare_arabic(n_each, args.seed)
    if "urdu"    in langs: results["urdu"]    = prepare_urdu(n_each, args.seed)
    if "chinese" in langs: results["chinese"] = prepare_chinese(n_each, args.seed)

    print(f"\n{'='*58}")
    print("  Ready files:")
    for lang, path in results.items():
        status = path.name if path else "FAILED (check logs above)"
        print(f"    {lang.capitalize():<10}  {status}")
    print(f"\n  Next: python scripts/runner.py")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
