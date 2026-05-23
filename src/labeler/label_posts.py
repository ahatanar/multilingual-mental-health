"""
Silver-label labeling pipeline for Chinese & Spanish datasets.

Generates post-level depression labels using two independent classifiers
(XLM-RoBERTa + Grok). Only keeps posts where both classifiers agree.
Collects 250 depressed + 250 not-depressed consensus posts per language.

Usage:
    py labeler/label_posts.py                          # label both languages
    py labeler/label_posts.py --language chinese        # one language only
    py labeler/label_posts.py --resume                  # resume from checkpoint
    py labeler/label_posts.py --dry-run --limit 10      # test with 10 posts
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import get_api_key
from labeler.classifiers import XLMRobertaClassifier, GrokClassifier
from labeler.translator import CachedTranslator
from labeler.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
SAMPLED_DIR = os.path.join(DATA_DIR, "sampled")

SUPPORTED_LANGUAGES = ["chinese", "spanish"]
TARGET_PER_CLASS = 250
SEED = 42

# Length limits (same as evaluation/sampler.py)
MAX_WORD_COUNT = 240
MAX_CHAR_COUNT = 240
CJK_LANGUAGES = {"chinese"}


def load_cleaned_data(language: str) -> List[Dict]:
    """Load parsed data from data/cleaned/<language>.json."""
    path = os.path.join(CLEANED_DIR, f"{language}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No cleaned data for '{language}'. Run `py prepare_data.py` first.\n"
            f"  Expected: {path}"
        )
    logger.info(f"[{language}] Loading cleaned data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[{language}] Loaded {len(data)} posts")
    return data


def measure_length(text: str, language: str) -> int:
    """Measure post length — chars for CJK, words otherwise."""
    if language in CJK_LANGUAGES:
        return len(text)
    return len(text.split())


def is_too_long(text: str, language: str) -> bool:
    """Check if a post exceeds the length limit."""
    limit = MAX_CHAR_COUNT if language in CJK_LANGUAGES else MAX_WORD_COUNT
    return measure_length(text, language) > limit


TRANSLATE_BATCH_SIZE = 20


def _build_pools(data: List[Dict], language: str) -> Dict[str, List[Dict]]:
    """
    Split cleaned data into depressed / not-depressed candidate pools.

    Each pool is filtered for length and shuffled with a fixed seed.
    The user-level label tells us which pool to draw from so we don't
    waste API calls classifying posts that are unlikely to match the
    quota we still need.
    """
    pools: Dict[str, List[Dict]] = {"depression": [], "normal": []}
    for i, row in enumerate(data):
        post = row.get("post", "").strip()
        label = row.get("label", "").strip().lower()
        if not post or label not in pools:
            continue
        if is_too_long(post, language):
            continue
        pools[label].append({"idx": i, "post": post, "word_count": measure_length(post, language)})

    random.seed(SEED)
    for pool in pools.values():
        random.shuffle(pool)

    print(f"  Candidate pools — depression: {len(pools['depression'])}, "
          f"normal: {len(pools['normal'])}")
    return pools


def _batch_translate(candidates: List[Dict], translator: CachedTranslator,
                     src: str) -> List[Dict]:
    """Translate a batch of candidates, skipping failures."""
    for cand in candidates:
        if "english_text" in cand:
            continue
        try:
            cand["english_text"] = translator.translate(cand["post"], src=src, dest="en")
        except Exception as e:
            logger.warning(f"[Translator] Skipping post (translate failed): {e}")
            cand["english_text"] = None
    return candidates


def label_language(language: str, xlm: XLMRobertaClassifier,
                   grok: GrokClassifier, translator: CachedTranslator,
                   resume: bool = False, dry_run: bool = False,
                   limit: Optional[int] = None) -> Optional[Dict]:
    """
    Run the silver-label pipeline for a single language.

    Draws candidates from the matching user-level pool (depressed users
    for the depressed quota, control users for the not-depressed quota)
    to minimise wasted API calls.

    Returns metadata dict on success, None on failure.
    """
    print(f"\n{'='*60}")
    print(f"  Labeling: {language.upper()}")
    print(f"{'='*60}")

    # 1. Load cleaned data & build pools
    data = load_cleaned_data(language)
    pools = _build_pools(data, language)
    trans_src = "zh-cn" if language == "chinese" else "es"

    # 2. Load or create checkpoint
    ckpt = CheckpointManager(language)
    if resume:
        if ckpt.load():
            print(f"  Resumed: {ckpt.depressed_count} dep / "
                  f"{ckpt.not_depressed_count} not-dep / "
                  f"{ckpt.total_processed} processed")
            translator.load_cache(ckpt.accepted, src=trans_src)
        else:
            print("  No checkpoint found, starting fresh")

    if ckpt.is_done(TARGET_PER_CLASS) and not dry_run:
        print(f"  Already have {TARGET_PER_CLASS}+{TARGET_PER_CLASS} posts. Skipping.")
        return _export(language, ckpt, data)

    # 3. Process each pool towards its quota
    #    "depression" user-pool  -> fills the "depressed" quota
    #    "normal"     user-pool  -> fills the "not depressed" quota
    pool_map = [
        ("depression", "depressed", lambda: ckpt.depressed_count),
        ("normal", "not depressed", lambda: ckpt.not_depressed_count),
    ]

    total_errors = 0
    total_this_run = 0

    for pool_label, target_class, count_fn in pool_map:
        needed = TARGET_PER_CLASS - count_fn()
        if needed <= 0:
            print(f"  [{pool_label}] Already have {TARGET_PER_CLASS} — skipping pool")
            continue

        pool = pools[pool_label]
        pool_pos = 0  # current position in pool
        print(f"  [{pool_label}] Need {needed} more '{target_class}' posts "
              f"(pool size: {len(pool)})")

        while count_fn() < TARGET_PER_CLASS and pool_pos < len(pool):
            # Check limit
            if limit is not None and total_this_run >= limit:
                break

            # Grab next batch of candidates
            batch_end = min(pool_pos + TRANSLATE_BATCH_SIZE, len(pool))
            batch = []
            while pool_pos < batch_end:
                cand = pool[pool_pos]
                pool_pos += 1
                if ckpt.is_processed(cand["idx"]):
                    continue
                batch.append(cand)

            if not batch:
                continue

            # Batch-translate
            _batch_translate(batch, translator, src=trans_src)

            # Classify each candidate
            for cand in batch:
                if limit is not None and total_this_run >= limit:
                    break
                if count_fn() >= TARGET_PER_CLASS:
                    break

                idx = cand["idx"]
                post = cand["post"]
                english_text = cand.get("english_text")

                if english_text is None:
                    # Translation failed — skip
                    ckpt.add_result(idx, accepted=False)
                    ckpt.save()
                    total_this_run += 1
                    total_errors += 1
                    continue

                try:
                    # XLM-RoBERTa: original Spanish, English for Chinese
                    xlm_input = post if language != "chinese" else english_text
                    xlm_label = xlm.classify(xlm_input)

                    # Grok: always English
                    grok_label = grok.classify(english_text)
                except Exception as e:
                    logger.error(f"[{language}] Error on post {idx}: {e}")
                    total_errors += 1
                    ckpt.add_result(idx, accepted=False)
                    ckpt.save()
                    total_this_run += 1
                    continue

                # Consensus check
                if xlm_label == grok_label and xlm_label == target_class:
                    label_raw = "depression" if target_class == "depressed" else "normal"
                    entry = {
                        "post": post,
                        "label": label_raw,
                        "ground_truth": target_class,
                        "word_count": cand["word_count"],
                        "xlm_label": xlm_label,
                        "grok_label": grok_label,
                        "translated": english_text,
                    }
                    ckpt.add_result(idx, accepted=True, entry=entry, label=target_class)
                else:
                    ckpt.add_result(idx, accepted=False)

                ckpt.save()
                total_this_run += 1

                # Progress log
                if total_this_run % 10 == 0:
                    print(f"  [{language}] Processed {ckpt.total_processed} | "
                          f"Accepted {ckpt.depressed_count}d/{ckpt.not_depressed_count}nd | "
                          f"Disagreements {ckpt.disagreements} | "
                          f"Errors {total_errors}")

        filled = count_fn()
        print(f"  [{pool_label}] Filled {filled}/{TARGET_PER_CLASS}")

    # Final summary
    total = ckpt.total_processed
    agreement_rate = len(ckpt.accepted) / total if total > 0 else 0
    print(f"\n  [{language}] DONE:")
    print(f"    Total processed:  {total}")
    print(f"    Accepted:         {len(ckpt.accepted)} "
          f"({ckpt.depressed_count} dep / {ckpt.not_depressed_count} not-dep)")
    print(f"    Disagreements:    {ckpt.disagreements}")
    print(f"    Agreement rate:   {agreement_rate:.1%}")
    print(f"    Errors:           {total_errors}")

    if dry_run:
        print("  [DRY RUN] Skipping export")
        return None

    if not ckpt.is_done(TARGET_PER_CLASS):
        print(f"  WARNING: Did not reach {TARGET_PER_CLASS} per class. "
              f"Got {ckpt.depressed_count} dep / {ckpt.not_depressed_count} not-dep")

    return _export(language, ckpt, data)


def _export(language: str, ckpt: CheckpointManager,
            source_data: List[Dict]) -> Dict:
    """Export accepted posts to data/sampled/<language>.json."""
    os.makedirs(SAMPLED_DIR, exist_ok=True)
    output_path = os.path.join(SAMPLED_DIR, f"{language}.json")

    total = ckpt.total_processed
    agreement_rate = len(ckpt.accepted) / total if total > 0 else 0

    # Build samples in the format runner.py expects
    samples = []
    for entry in ckpt.accepted:
        samples.append({
            "post": entry["post"],
            "label": entry["label"],
            "ground_truth": entry["ground_truth"],
            "word_count": entry["word_count"],
        })

    gt_dist = {}
    for s in samples:
        gt_dist[s["ground_truth"]] = gt_dist.get(s["ground_truth"], 0) + 1

    export = {
        "metadata": {
            "language": language,
            "sample_size": len(samples),
            "requested_size": TARGET_PER_CLASS * 2,
            "seed": SEED,
            "created": datetime.now().isoformat(),
            "source_total": len(source_data),
            "label_distribution": gt_dist,
            "labeling_method": "silver_label_consensus",
            "classifiers": ["xlm-roberta-depression", "grok"],
            "agreement_rate": round(agreement_rate, 4),
        },
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"  Exported {len(samples)} samples -> {output_path}")
    return export["metadata"]


def main():
    parser = argparse.ArgumentParser(
        description="Silver-label labeling pipeline for Chinese & Spanish datasets"
    )
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None,
                        help="Label a specific language (default: both)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode — don't export final results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max posts to process (useful with --dry-run)")
    args = parser.parse_args()

    languages = [args.language] if args.language else SUPPORTED_LANGUAGES

    print("\n" + "=" * 60)
    print("  Silver-Label Consensus Pipeline")
    print("=" * 60)
    print(f"  Languages:  {', '.join(l.capitalize() for l in languages)}")
    print(f"  Target:     {TARGET_PER_CLASS} depressed + {TARGET_PER_CLASS} not-depressed per language")
    print(f"  Resume:     {args.resume}")
    print(f"  Dry run:    {args.dry_run}")
    if args.limit:
        print(f"  Limit:      {args.limit} posts")

    # Initialize classifiers (shared across languages)
    print("\n  Loading classifiers...")
    xlm = XLMRobertaClassifier()

    grok_key = get_api_key("grok")
    grok = GrokClassifier(api_key=grok_key)
    print("  Classifiers ready.")

    # Initialize translator
    translator = CachedTranslator()

    # Run for each language
    results = {}
    for lang in languages:
        try:
            meta = label_language(
                language=lang,
                xlm=xlm,
                grok=grok,
                translator=translator,
                resume=args.resume,
                dry_run=args.dry_run,
                limit=args.limit,
            )
            if meta:
                results[lang] = meta
        except Exception as e:
            logger.error(f"[{lang}] Failed: {e}", exc_info=True)
            print(f"\n  ERROR [{lang}]: {e}")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("  Pipeline complete!")
        print(f"{'='*60}")
        for lang, meta in results.items():
            print(f"  {lang.capitalize()}: {meta['sample_size']} samples "
                  f"({meta['label_distribution']}) "
                  f"agreement={meta.get('agreement_rate', 'N/A')}")
        print(f"\n  Output: data/sampled/<language>.json")
        print(f"  Next:   py runner.py")


if __name__ == "__main__":
    main()
