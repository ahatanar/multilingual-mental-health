"""
Translate the Phase 2 Chinese sample set to English and auto-filter for ethics.

Each post is processed in a single DeepSeek API call that simultaneously:
  1. Translates Chinese → English
  2. Flags problematic content (explicit sexual, graphic violence, PII, or
     translation failure)

Parallel workers dramatically reduce wall-clock time:
  • 1 worker  @ 0.5 s delay → ~50 min  for 6 000 posts
  • 5 workers @ 0.5 s delay → ~10 min
  • 10 workers               → ~5  min

Outputs:
  • data/phase2/translated/chinese_6000samples_seed42_translated.json
      — all posts with translation + filter_status
  • data/phase2/translated/filtered/chinese_6000samples_seed42_filtered.json
      — KEEP posts only  (ready for prepare_experiment1.py --lang chinese)

Usage:
    # Smoke test (10 posts, 3 workers)
    python scripts/phase2/translate_chinese.py --samples 10

    # Full run
    python scripts/phase2/translate_chinese.py --samples 6000

    # Full run with more parallelism
    python scripts/phase2/translate_chinese.py --samples 6000 --workers 10

    # Resume after interruption
    python scripts/phase2/translate_chinese.py --samples 6000

    # Start over from scratch
    python scripts/phase2/translate_chinese.py --samples 6000 --fresh
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
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
PHASE2_DIR      = PROJECT_ROOT / "data" / "phase2"
TRANSLATED_DIR  = PHASE2_DIR / "translated"
FILTERED_DIR    = TRANSLATED_DIR / "filtered"
PROGRESS_DIR    = PHASE2_DIR / "translation_progress"

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL           = "deepseek-chat"
DEFAULT_WORKERS = 5
DEFAULT_DELAY   = 0.5    # seconds between requests per worker
SAVE_EVERY      = 50     # checkpoint every N completions
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5.0    # seconds, multiplied by attempt number on retry

DEFAULT_SAMPLES = 6000
DEFAULT_SEED    = 42

# ── Filter categories ─────────────────────────────────────────────────────────
REMOVE_CATEGORIES = {
    "REMOVE_SEXUAL":     "explicit sexual content",
    "REMOVE_GORE":       "graphic gore / extreme physical violence (NOT mental health content)",
    "REMOVE_PII":        "personal identifying information",
    "REMOVE_SPAM":       "advertising, promotional, or bot-generated content",
    "REMOVE_INCOMPLETE": "too short, only emojis/symbols/URLs, or no meaningful text",
    "REMOVE_FAILURE":    "translation failure / garbled / non-Chinese text",
}


# ─────────────────────────────────────────────────────────────────────────────
# Input loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prepared(n_samples: int, seed: int) -> List[Dict]:
    path = PHASE2_DIR / f"chinese_{n_samples}samples_seed{seed}.json"
    if not path.exists():
        logger.error(
            f"Prepared dataset not found: {path}\n"
            f"  Run first: python scripts/phase2/prepare_chinese.py "
            f"--samples {n_samples} --seed {seed}"
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} posts  (seed={seed}, source: {path.name})")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(n_samples: int, seed: int) -> Path:
    return PROGRESS_DIR / f"chinese_translation_{n_samples}s_seed{seed}.json"


def load_checkpoint(n_samples: int, seed: int) -> Dict[int, Dict]:
    """Return dict of {index: result} for already-completed posts."""
    path = _checkpoint_path(n_samples, seed)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = {item["index"]: item for item in data.get("items", [])}
        logger.info(f"Resumed checkpoint: {len(items)}/{data.get('total', n_samples)} done")
        return items
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return {}


def save_checkpoint(
    results: Dict[int, Dict], total: int, n_samples: int, seed: int
) -> None:
    path = _checkpoint_path(n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(results.values(), key=lambda x: x["index"])
    payload = {
        "total":         total,
        "done_count":    len(items),
        "keep_count":    sum(1 for i in items if i["filter_status"] == "KEEP"),
        "remove_count":  sum(1 for i in items if i["filter_status"] != "KEEP"),
        "failed_count":  sum(1 for i in items if i["translation_failed"]),
        "seed":          seed,
        "last_updated":  datetime.now().isoformat(),
        "items":         items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Translate + filter (single API call)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are assisting with a mental health research ethics review. "
    "Translate Chinese social media posts to English and flag content that is "
    "unsuitable for academic research."
)

USER_TEMPLATE = """\
Translate the following Chinese social media post to English, then classify it for research ethics review.

Chinese text:
{text}

Respond in EXACTLY this format (no other text):
TRANSLATION: <single-line English translation>
STATUS: <see options below>

STATUS options:
- KEEP              — usable research content (includes depression, sadness, anxiety, \
suicidal ideation — these are valid for mental health research)
- REMOVE_SEXUAL     — explicit pornographic or sexual content
- REMOVE_GORE       — graphic gore or extreme physical violence (NOT mental health \
content — do NOT flag posts about emotional pain, self-harm feelings, or suicidal thoughts)
- REMOVE_PII        — real full names with contact info, phone numbers, home addresses, \
or other identifying information
- REMOVE_SPAM       — advertising, product promotions, or clearly bot-generated content
- REMOVE_INCOMPLETE — post has no meaningful text (only emojis, punctuation, URLs, \
or fewer than 5 meaningful words after translation)
- REMOVE_FAILURE    — text is untranslatable, garbled, or contains no Chinese characters"""


def _parse_response(text: str) -> Tuple[Optional[str], str]:
    """
    Parse 'TRANSLATION: ...\nSTATUS: ...' response.
    Returns (translation, filter_status).  filter_status defaults to 'KEEP'.
    """
    translation   = None
    filter_status = "KEEP"

    lines = text.strip().splitlines()
    trans_lines: List[str] = []
    found_status = False

    for line in lines:
        if line.startswith("STATUS:"):
            status_val = line[len("STATUS:"):].strip().upper()
            if status_val in REMOVE_CATEGORIES:
                filter_status = status_val
            else:
                filter_status = "KEEP"
            found_status = True
        elif line.startswith("TRANSLATION:"):
            trans_lines = [line[len("TRANSLATION:"):].strip()]
        elif trans_lines and not found_status:
            # Multi-line translation (before STATUS: appears)
            trans_lines.append(line)

    if trans_lines:
        translation = " ".join(trans_lines).strip() or None

    return translation, filter_status


def process_post(client, post: Dict, delay: float) -> Dict:
    """Translate + filter one post. Returns result dict. Thread-safe."""
    text = post["post"]
    prompt = USER_TEMPLATE.format(text=text)

    translation   = None
    filter_status = "KEEP"
    failed        = False

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content
            translation, filter_status = _parse_response(raw)

            # If translation came back empty, treat as failure
            if not translation:
                filter_status = "REMOVE_FAILURE"
                failed = True

            time.sleep(delay)
            break

        except Exception as e:
            logger.warning(f"[index {post.get('index')}] attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
            else:
                failed        = True
                filter_status = "REMOVE_FAILURE"

    return {
        "index":              post.get("index", -1),
        "original":           text,
        "translation":        translation,
        "label":              post["label"],
        "filter_status":      filter_status,
        "translation_failed": failed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output saving
# ─────────────────────────────────────────────────────────────────────────────

def save_translated(results: Dict[int, Dict], n_samples: int, seed: int, status: str) -> Path:
    items  = sorted(results.values(), key=lambda x: x["index"])
    n_dep  = sum(1 for i in items if i["label"] == "depression")
    n_keep = sum(1 for i in items if i["filter_status"] == "KEEP")

    path = TRANSLATED_DIR / f"chinese_{n_samples}samples_seed{seed}_translated.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    removed_breakdown = {
        cat: sum(1 for i in items if i["filter_status"] == cat)
        for cat in REMOVE_CATEGORIES
    }

    output = {
        "metadata": {
            "language":          "chinese",
            "dataset":           "Weibo Depression Dataset",
            "model":             MODEL,
            "seed":              seed,
            "total_requested":   n_samples,
            "total_processed":   len(items),
            "depressed_count":   n_dep,
            "normal_count":      len(items) - n_dep,
            "keep_count":        n_keep,
            "removed_count":     len(items) - n_keep,
            "removed_breakdown": removed_breakdown,
            "failed_count":      sum(1 for i in items if i["translation_failed"]),
            "status":            status,
            "timestamp":         datetime.now().isoformat(),
        },
        "samples": items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Translated output → {path}")
    return path


def save_filtered(results: Dict[int, Dict], n_samples: int, seed: int) -> Path:
    """Save only KEEP posts — this is what prepare_experiment1.py reads."""
    keep_items = sorted(
        (i for i in results.values() if i["filter_status"] == "KEEP"),
        key=lambda x: x["index"],
    )
    n_dep  = sum(1 for i in keep_items if i["label"] == "depression")
    n_norm = len(keep_items) - n_dep

    path = FILTERED_DIR / f"chinese_{n_samples}samples_seed{seed}_filtered.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "language":       "chinese",
            "dataset":        "Weibo Depression Dataset",
            "model":          MODEL,
            "seed":           seed,
            "total_filtered": len(keep_items),
            "depressed_count": n_dep,
            "normal_count":   n_norm,
            "filter_reason":  (
                "AI-filtered: translation failures, explicit sexual content, "
                "gore, PII, spam, incomplete posts"
            ),
            "status":         "filtered",
            "timestamp":      datetime.now().isoformat(),
        },
        "samples": keep_items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Filtered output  → {path}  ({len(keep_items)} posts kept)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate + auto-filter Phase 2 Chinese posts via DeepSeek."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel translation workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--delay",   type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between requests per worker (default: {DEFAULT_DELAY})")
    parser.add_argument("--fresh",   action="store_true",
                        help="Ignore any saved checkpoint and start from scratch.")
    args = parser.parse_args()

    # ── DeepSeek client ───────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed.  Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=get_api_key("deepseek"), base_url="https://api.deepseek.com")
    logger.info(f"DeepSeek client ready — model: {MODEL}, workers: {args.workers}")

    # ── Load data + checkpoint ────────────────────────────────────────────────
    sampled   = load_prepared(args.samples, args.seed)
    total     = len(sampled)
    results   = {} if args.fresh else load_checkpoint(args.samples, args.seed)
    pending   = [p for p in sampled if p.get("index", sampled.index(p)) not in results]

    if not pending:
        logger.info("Already complete. Use --fresh to re-run from scratch.")
        save_translated(results, args.samples, args.seed, "complete")
        save_filtered(results, args.samples, args.seed)
        return

    logger.info(
        f"{'Starting' if not results else 'Resuming'}: "
        f"{len(pending)} posts remaining  (target: {total})"
    )

    # ── Thread-safe progress state ────────────────────────────────────────────
    lock        = threading.Lock()
    done_count  = [len(results)]   # mutable counter shared across threads

    def _on_done(post: Dict, result: Dict) -> None:
        with lock:
            results[result["index"]] = result
            done_count[0] += 1
            n = done_count[0]
            keep = result["filter_status"] == "KEEP"
            logger.info(
                f"[{n:>5}/{total}]  {result['label']:<12}  "
                f"{'✓ KEEP' if keep else '✗ ' + result['filter_status']}"
            )
            if n % SAVE_EVERY == 0:
                save_checkpoint(results, total, args.samples, args.seed)
                logger.info(f"  ↳ checkpoint saved ({n}/{total})")

    # ── Parallel processing ───────────────────────────────────────────────────
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_post, client, post, args.delay): post
                for post in pending
            }
            for future in as_completed(futures):
                post = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for index {post.get('index')}: {e}")
                    result = {
                        "index":              post.get("index", -1),
                        "original":           post["post"],
                        "translation":        None,
                        "label":              post["label"],
                        "filter_status":      "REMOVE_FAILURE",
                        "translation_failed": True,
                    }
                _on_done(post, result)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress…")
        save_checkpoint(results, total, args.samples, args.seed)
        save_translated(results, args.samples, args.seed, "partial")
        save_filtered(results, args.samples, args.seed)
        logger.info(f"Saved {len(results)}/{total}. Re-run same command to resume.")
        sys.exit(0)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(results, total, args.samples, args.seed)
    trans_path   = save_translated(results, args.samples, args.seed, "complete")
    filter_path  = save_filtered(results, args.samples, args.seed)

    items      = list(results.values())
    n_keep     = sum(1 for i in items if i["filter_status"] == "KEEP")
    n_removed  = len(items) - n_keep
    n_dep_kept = sum(1 for i in items if i["filter_status"] == "KEEP" and i["label"] == "depression")
    n_nor_kept = n_keep - n_dep_kept

    print(f"\n{'='*60}")
    print(f"  Translation + filtering complete")
    print(f"  Total processed : {len(items)}")
    print(f"  Kept            : {n_keep}  (dep={n_dep_kept}, normal={n_nor_kept})")
    print(f"  Removed         : {n_removed}")
    for cat, desc in REMOVE_CATEGORIES.items():
        n = sum(1 for i in items if i["filter_status"] == cat)
        if n:
            print(f"    {cat:<22} {n}  ({desc})")
    print(f"\n  Translated file : {trans_path.name}")
    print(f"  Filtered file   : {filter_path}")
    print(f"\n  Next: python scripts/phase2/prepare_experiment1.py --lang chinese")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
