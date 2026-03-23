"""
Translate the Phase 2 Arabic sample set to English using Cohere.

Reads the prepared dataset produced by prepare_arabic.py
(data/phase2/arabic_6000samples_seed42.json), translates each post,
and saves the result with all original metadata preserved (label,
dialect, source) so downstream analysis can filter by any of those.

Usage:
    # Step 1 — prepare the dataset first (run once)
    python scripts/phase2/prepare_arabic.py

    # Step 2 — quick smoke test (10 posts)
    python scripts/phase2/translate_arabic.py --samples 10

    # Step 3 — full run
    python scripts/phase2/translate_arabic.py --samples 6000

    # Resume after interruption (auto-detects checkpoint)
    python scripts/phase2/translate_arabic.py --samples 6000

    # Start over from scratch
    python scripts/phase2/translate_arabic.py --samples 6000 --fresh

Model: command-r-08-2024
  • Equivalent translation quality to command-a-translate for this task
  • Higher rate limit → 1.0 s delay vs 3.5 s → ~1.7 h for 6 000 posts
  • Billed at ~$0.15 / 1M input + $0.60 / 1M output (negligible for this volume)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup (scripts/phase2/ → scripts/ → project root) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHASE2_DIR    = PROJECT_ROOT / "data" / "phase2"
TRANSLATED_DIR = PHASE2_DIR / "translated"
PROGRESS_DIR   = PHASE2_DIR / "translation_progress"

# ── Cohere settings ───────────────────────────────────────────────────────────
#
# command-r-08-2024:
#   • Natively multilingual; Arabic → English well-supported
#   • Manual review confirmed quality on par with command-a-translate-08-2025
#   • Higher rate limit allows 1.0 s delay (vs 3.5 s for command-a-translate)
#     → 6 000-post run takes ~1.7 h instead of ~6 h
#
MODEL          = "command-r-08-2024"
SAVE_EVERY     = 50    # checkpoint every N translations
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5.0   # seconds, multiplied by attempt number on retry
REQUEST_DELAY  = 1.0   # seconds between requests

DEFAULT_SAMPLES = 6000
DEFAULT_SEED    = 42


# ─────────────────────────────────────────────────────────────────────────────
# Input loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prepared(n_samples: int, seed: int) -> List[Dict]:
    """
    Load the dataset produced by prepare_arabic.py.
    Exits with a clear message if the file doesn't exist yet.
    """
    path = PHASE2_DIR / f"arabic_{n_samples}samples_seed{seed}.json"
    if not path.exists():
        logger.error(
            f"Prepared dataset not found: {path}\n"
            f"  Run first: python scripts/phase2/prepare_arabic.py --samples {n_samples} --seed {seed}"
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    logger.info(
        f"Loaded {len(samples)} posts from prepared dataset  "
        f"(seed={seed}, source: {path.name})"
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint & output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(n_samples: int, seed: int) -> Path:
    return PROGRESS_DIR / f"arabic_translation_{n_samples}s_seed{seed}.json"


def _output_path(n_samples: int, seed: int) -> Path:
    return TRANSLATED_DIR / f"arabic_{n_samples}samples_seed{seed}_translated.json"


def load_checkpoint(n_samples: int, seed: int) -> List[Dict]:
    path = _checkpoint_path(n_samples, seed)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        logger.info(f"Resumed checkpoint: {len(items)}/{data.get('total', n_samples)} translated")
        return items
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return []


def save_checkpoint(translated: List[Dict], total: int, n_samples: int, seed: int) -> None:
    path = _checkpoint_path(n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_dep  = sum(1 for t in translated if t["label"] == "depression")
    payload = {
        "total":                   total,
        "translated_count":        len(translated),
        "depressed_translated":    n_dep,
        "normal_translated":       len(translated) - n_dep,
        "failed_count":            sum(1 for t in translated if t["translation_failed"]),
        "seed":                    seed,
        "last_updated":            datetime.now().isoformat(),
        "items":                   translated,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_output(translated: List[Dict], n_samples: int, seed: int, status: str) -> Path:
    path = _output_path(n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_dep  = sum(1 for t in translated if t["label"] == "depression")
    output = {
        "metadata": {
            "language":           "arabic",
            "dataset":            "CairoDep",
            "model":              MODEL,
            "seed":               seed,
            "total_requested":    n_samples,
            "total_translated":   len(translated),
            "depressed_count":    n_dep,
            "normal_count":       len(translated) - n_dep,
            "failed_count":       sum(1 for t in translated if t["translation_failed"]),
            "status":             status,
            "timestamp":          datetime.now().isoformat(),
        },
        # Each entry: index, original, translation, label, dialect, source, translation_failed
        "samples": translated,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_post(co, text: str) -> Optional[str]:
    """Translate one Arabic post to English. Returns None on total failure."""
    prompt = (
        "Translate the following Arabic text to English.\n"
        "Output ONLY the English translation — no explanation, no preamble.\n\n"
        f"{text}"
    )
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = co.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Phase 2 Arabic posts to English via Cohere Command R."
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Number of posts in the prepared dataset (default: {DEFAULT_SAMPLES}). "
             "Must match the value used with prepare_arabic.py.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Seed used when running prepare_arabic.py (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any saved checkpoint and start from scratch.",
    )
    args = parser.parse_args()

    # ── Cohere client ─────────────────────────────────────────────────────────
    try:
        import cohere
    except ImportError:
        logger.error("cohere package not installed.  Run: pip install cohere")
        sys.exit(1)

    co = cohere.ClientV2(api_key=get_api_key("cohere"), timeout=30.0)
    logger.info(f"Cohere client ready — model: {MODEL}")

    # ── Load prepared dataset ─────────────────────────────────────────────────
    sampled = load_prepared(args.samples, args.seed)
    total   = len(sampled)

    # ── Resume or start fresh ─────────────────────────────────────────────────
    translated: List[Dict] = [] if args.fresh else load_checkpoint(args.samples, args.seed)
    start_index = len(translated)
    remaining   = total - start_index

    if remaining == 0:
        logger.info("Already complete. Use --fresh to re-run from scratch.")
        return

    logger.info(
        f"{'Starting' if start_index == 0 else 'Resuming'}: "
        f"{remaining} posts remaining (target: {total})"
    )

    # ── Translation loop ──────────────────────────────────────────────────────
    try:
        for i, post in enumerate(sampled[start_index:], start=start_index):
            translation = translate_post(co, post["post"])
            failed      = translation is None

            # Carry all prepared-dataset fields through to the output
            entry = {
                "index":              post.get("index", i),
                "original":           post["post"],
                "translation":        translation,
                "label":              post["label"],
                "dialect":            post.get("dialect", ""),
                "source":             post.get("source", ""),
                "translation_failed": failed,
            }
            translated.append(entry)

            n_done = i + 1
            logger.info(f"[{n_done:>5}/{total}]  {'FAIL' if failed else post['label']}"
                        f"  [{post.get('dialect','')}]")

            if n_done % SAVE_EVERY == 0:
                save_checkpoint(translated, total, args.samples, args.seed)
                logger.info(f"  ↳ checkpoint saved ({n_done}/{total})")

            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress…")
        save_checkpoint(translated, total, args.samples, args.seed)
        save_output(translated, args.samples, args.seed, "partial")
        logger.info(f"Saved {len(translated)}/{total}. Re-run same command to resume.")
        sys.exit(0)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(translated, total, args.samples, args.seed)
    out    = save_output(translated, args.samples, args.seed, "complete")
    n_dep  = sum(1 for t in translated if t["label"] == "depression")
    n_fail = sum(1 for t in translated if t["translation_failed"])

    print(f"\n{'='*60}")
    print(f"  Translation complete")
    print(f"  Posts translated : {len(translated)}")
    print(f"  Depressed        : {n_dep}")
    print(f"  Normal           : {len(translated) - n_dep}")
    print(f"  Failed           : {n_fail}")
    print(f"  Output           : {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
