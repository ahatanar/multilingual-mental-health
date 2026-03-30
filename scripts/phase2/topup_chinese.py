"""
Top-up the Chinese filtered dataset to reach 2 500 depressed + 2 500 normal.

Reads the existing filtered file to see what's already kept, samples only the
additional posts needed (with a safety buffer), translates + auto-filters them
via DeepSeek, then merges with the existing filtered file.

No re-doing already-translated posts.

Usage:
    # Dry run — shows how many posts will be sampled
    python scripts/phase2/topup_chinese.py --dry-run

    # Full top-up
    python scripts/phase2/topup_chinese.py

    # More parallelism
    python scripts/phase2/topup_chinese.py --workers 10
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
from typing import Dict, List, Optional, Set, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_api_key
from evaluation.parsers import ChineseParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHASE2_DIR     = PROJECT_ROOT / "data" / "phase2"
FILTERED_DIR   = PHASE2_DIR / "translated" / "filtered"
PROGRESS_DIR   = PHASE2_DIR / "translation_progress"

FILTERED_FILE  = FILTERED_DIR / "chinese_6000samples_seed42_filtered.json"
TOPUP_PROGRESS = PROGRESS_DIR / "chinese_topup_progress.json"

# ── Settings ──────────────────────────────────────────────────────────────────
TARGET_EACH    = 2500   # 2500 dep + 2500 normal
BUFFER_MULT    = 2.0    # sample this many × the gap to account for filtering
MODEL          = "deepseek-chat"
DEFAULT_WORKERS = 5
DEFAULT_DELAY   = 0.5
SAVE_EVERY      = 50
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5.0

REMOVE_CATEGORIES = {
    "REMOVE_SEXUAL":     "explicit sexual content",
    "REMOVE_GORE":       "graphic gore / extreme physical violence",
    "REMOVE_PII":        "personal identifying information",
    "REMOVE_SPAM":       "advertising, promotional, or bot-generated content",
    "REMOVE_INCOMPLETE": "too short, only emojis/symbols/URLs, or no meaningful text",
    "REMOVE_FAILURE":    "translation failure / garbled / non-Chinese text",
}

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_filtered() -> Tuple[List[Dict], int, int]:
    """Return (existing_kept_posts, dep_count, normal_count)."""
    if not FILTERED_FILE.exists():
        logger.error(f"Filtered file not found: {FILTERED_FILE}")
        sys.exit(1)
    with open(FILTERED_FILE, encoding="utf-8") as f:
        data = json.load(f)
    samples   = data["samples"]
    dep_count = sum(1 for s in samples if s["label"] == "depression")
    nor_count = len(samples) - dep_count
    return samples, dep_count, nor_count


def load_already_seen() -> Set[str]:
    """Collect post texts already in the filtered file (to avoid duplicates)."""
    if not FILTERED_FILE.exists():
        return set()
    with open(FILTERED_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {s["original"] for s in data["samples"]}


def sample_new_posts(
    n_dep: int, n_normal: int, already_seen: Set[str]
) -> List[Dict]:
    """Sample fresh posts from the cleaned cache, excluding already-seen texts."""
    import random

    logger.info("Loading cleaned Chinese data (this may take ~10s)…")
    parser = ChineseParser(data_dir=str(PROJECT_ROOT / "data"))
    all_posts = parser.get_data()

    dep_pool    = [p for p in all_posts if p["label"] == "depression" and p["post"] not in already_seen]
    normal_pool = [p for p in all_posts if p["label"] == "normal"     and p["post"] not in already_seen]

    logger.info(f"Available after exclusions: {len(dep_pool)} dep, {len(normal_pool)} normal")

    rng = random.Random(43)  # different seed from the original run
    sampled_dep    = rng.sample(dep_pool,    min(n_dep,    len(dep_pool)))
    sampled_normal = rng.sample(normal_pool, min(n_normal, len(normal_pool)))

    combined = sampled_dep + sampled_normal
    rng.shuffle(combined)
    for i, entry in enumerate(combined):
        entry["topup_index"] = i

    logger.info(f"Sampled {len(sampled_dep)} dep + {len(sampled_normal)} normal for top-up")
    return combined


def load_topup_progress() -> Dict[int, Dict]:
    if not TOPUP_PROGRESS.exists():
        return {}
    try:
        with open(TOPUP_PROGRESS, encoding="utf-8") as f:
            data = json.load(f)
        items = {item["topup_index"]: item for item in data.get("items", [])}
        logger.info(f"Resumed top-up checkpoint: {len(items)} done")
        return items
    except Exception as e:
        logger.warning(f"Could not load top-up checkpoint: {e}")
        return {}


def save_topup_progress(results: Dict[int, Dict], total: int) -> None:
    TOPUP_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(results.values(), key=lambda x: x["topup_index"])
    with open(TOPUP_PROGRESS, "w", encoding="utf-8") as f:
        json.dump({
            "total":        total,
            "done":         len(items),
            "last_updated": datetime.now().isoformat(),
            "items":        items,
        }, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Translate + filter (same logic as translate_chinese.py)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> Tuple[Optional[str], str]:
    translation, filter_status = None, "KEEP"
    trans_lines: List[str] = []
    found_status = False
    for line in text.strip().splitlines():
        if line.startswith("STATUS:"):
            val = line[len("STATUS:"):].strip().upper()
            filter_status = val if val in REMOVE_CATEGORIES else "KEEP"
            found_status = True
        elif line.startswith("TRANSLATION:"):
            trans_lines = [line[len("TRANSLATION:"):].strip()]
        elif trans_lines and not found_status:
            trans_lines.append(line)
    if trans_lines:
        translation = " ".join(trans_lines).strip() or None
    return translation, filter_status


def process_post(client, post: Dict, delay: float) -> Dict:
    prompt  = USER_TEMPLATE.format(text=post["post"])
    translation, filter_status, failed = None, "KEEP", False

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
            translation, filter_status = _parse_response(response.choices[0].message.content)
            if not translation:
                filter_status, failed = "REMOVE_FAILURE", True
            time.sleep(delay)
            break
        except Exception as e:
            logger.warning(f"[topup_index {post.get('topup_index')}] attempt {attempt}: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
            else:
                filter_status, failed = "REMOVE_FAILURE", True

    return {
        "topup_index":        post.get("topup_index", -1),
        "original":           post["post"],
        "translation":        translation,
        "label":              post["label"],
        "filter_status":      filter_status,
        "translation_failed": failed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Merge + save updated filtered file
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(existing: List[Dict], new_kept: List[Dict]) -> Tuple[Path, int, int]:
    combined  = existing + new_kept
    dep_count = sum(1 for s in combined if s["label"] == "depression")
    nor_count = len(combined) - dep_count

    # Re-index for cleanliness
    for i, s in enumerate(combined):
        s["index"] = i
        s.pop("topup_index", None)

    output = {
        "metadata": {
            "language":        "chinese",
            "dataset":         "Weibo Depression Dataset",
            "model":           MODEL,
            "seed":            "42+43 (topup)",
            "total_filtered":  len(combined),
            "depressed_count": dep_count,
            "normal_count":    nor_count,
            "filter_reason":   (
                "AI-filtered: translation failures, explicit sexual content, "
                "gore, PII, spam, incomplete posts"
            ),
            "status":          "filtered+topup",
            "timestamp":       datetime.now().isoformat(),
        },
        "samples": combined,
    }
    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Updated filtered file → {FILTERED_FILE}")
    logger.info(f"Total kept: {len(combined)}  (dep={dep_count}, normal={nor_count})")
    return FILTERED_FILE, dep_count, nor_count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-up Chinese filtered dataset to 2500 dep + 2500 normal."
    )
    parser.add_argument("--workers",  type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--delay",    type=float, default=DEFAULT_DELAY)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show how many posts will be sampled without translating.")
    parser.add_argument("--fresh",    action="store_true",
                        help="Ignore top-up checkpoint and start fresh.")
    args = parser.parse_args()

    # ── Check what we already have ────────────────────────────────────────────
    existing, dep_have, nor_have = load_filtered()
    dep_need  = max(0, TARGET_EACH - dep_have)
    nor_need  = max(0, TARGET_EACH - nor_have)

    logger.info(f"Current:  dep={dep_have}, normal={nor_have}")
    logger.info(f"Target:   dep={TARGET_EACH}, normal={TARGET_EACH}")
    logger.info(f"Gap:      dep={dep_need}, normal={nor_need}")

    if dep_need == 0 and nor_need == 0:
        logger.info("Already at target! Run prepare_experiment1.py --lang chinese")
        return

    # ── Calculate sample sizes with buffer ───────────────────────────────────
    # Use observed removal rates: ~21% dep, ~42% normal
    dep_removal_rate    = 0.21
    normal_removal_rate = 0.42

    n_sample_dep    = int(dep_need    / (1 - dep_removal_rate)    * BUFFER_MULT) if dep_need    else 0
    n_sample_normal = int(nor_need    / (1 - normal_removal_rate) * BUFFER_MULT) if nor_need    else 0

    print(f"\n{'='*55}")
    print(f"  Top-up plan")
    print(f"  Need {dep_need} more depressed  → sampling {n_sample_dep}")
    print(f"  Need {nor_need} more normal     → sampling {n_sample_normal}")
    print(f"  Total to translate: {n_sample_dep + n_sample_normal}")
    print(f"{'='*55}\n")

    if args.dry_run:
        return

    # ── Sample new posts ──────────────────────────────────────────────────────
    already_seen = load_already_seen()
    new_posts    = sample_new_posts(n_sample_dep, n_sample_normal, already_seen)

    if not new_posts:
        logger.error("No new posts available to sample.")
        return

    total = len(new_posts)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    results: Dict[int, Dict] = {} if args.fresh else load_topup_progress()
    pending = [p for p in new_posts if p["topup_index"] not in results]

    if not pending:
        logger.info("Top-up translation already complete (checkpoint found).")
    else:
        logger.info(f"Translating {len(pending)} posts  ({args.workers} workers)…")

    # ── DeepSeek client ───────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed.  Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=get_api_key("deepseek"), base_url="https://api.deepseek.com")

    lock       = threading.Lock()
    done_count = [len(results)]

    def _on_done(result: Dict) -> None:
        with lock:
            results[result["topup_index"]] = result
            done_count[0] += 1
            n    = done_count[0]
            keep = result["filter_status"] == "KEEP"
            logger.info(
                f"[{n:>4}/{total}]  {result['label']:<12}  "
                f"{'✓ KEEP' if keep else '✗ ' + result['filter_status']}"
            )
            if n % SAVE_EVERY == 0:
                save_topup_progress(results, total)

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
                    logger.error(f"Error for topup_index {post.get('topup_index')}: {e}")
                    result = {
                        "topup_index":        post.get("topup_index", -1),
                        "original":           post["post"],
                        "translation":        None,
                        "label":              post["label"],
                        "filter_status":      "REMOVE_FAILURE",
                        "translation_failed": True,
                    }
                _on_done(result)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — saving progress…")
        save_topup_progress(results, total)
        logger.info("Re-run same command to resume.")
        sys.exit(0)

    save_topup_progress(results, total)

    # ── Merge new KEEP posts into filtered file ────────────────────────────────
    new_kept = [r for r in results.values() if r["filter_status"] == "KEEP"]
    logger.info(f"New posts kept after filtering: {len(new_kept)}")

    _, dep_final, nor_final = merge_and_save(existing, new_kept)

    dep_still_need = max(0, TARGET_EACH - dep_final)
    nor_still_need = max(0, TARGET_EACH - nor_final)

    print(f"\n{'='*55}")
    print(f"  Top-up complete")
    print(f"  Filtered file now has: dep={dep_final}, normal={nor_final}")
    if dep_still_need or nor_still_need:
        print(f"\n  Still short: dep={dep_still_need}, normal={nor_still_need}")
        print(f"  Re-run this script to top-up again.")
    else:
        print(f"\n  Ready! Run:")
        print(f"    python scripts/phase2/prepare_experiment1.py --lang chinese")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
