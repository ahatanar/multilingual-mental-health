"""
Repair Experiment 3 result files whose entries are marked ``agreement =
"no_translation"`` even though the English translation actually exists on
disk (i.e. the runner loaded the wrong translation fixture).

This happens when ``load_translations_for_exp3`` picks up a dev-size
``*_10samples_seed42_translated.json`` before the real 5 000/6 000-sample
file. That bug is fixed in ``runner.py`` (glob now prefers the largest
file), but existing broken result files still need to be backfilled —
this script does exactly that, touching only the broken rows.

Behaviour:
  1. Load the exp3 result file (with ``no_translation`` rows).
  2. Rebuild the translation lookup from ``data/phase2/…`` using the same
     priority order as the runner.
  3. For each row with ``agreement == "no_translation"``:
       - if ``post_full`` is NOT in the lookup  → leave it alone (real gap)
       - if it IS in the lookup                → re-run just that entry
  4. Merge repaired rows back into the full results list, recompute the
     metrics block, and save a new timestamped file next to the original.

All other rows (already-classified, LM-Studio errors, exp1-skipped, etc.)
are untouched — unlike the wider rerun script, this is surgical.

Usage:
    python scripts/phase2/repair_exp3_translations.py \\
        --file "results/phase2/experiment3/Local Models/gemma/gemma_urdu_20260407_052201.json"

    python scripts/phase2/repair_exp3_translations.py --file <path> --delay 0.2
    python scripts/phase2/repair_exp3_translations.py --file <path> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_api_key
from evaluation.prompts import PROMPTS


def _load_provider(model_key: str):
    """Lazy-import only the provider we actually need so missing SDKs for
    other providers don't block a repair run (e.g. no anthropic installed
    when repairing a gemma file)."""
    if model_key == "gemini":
        from models.gemini_provider import GeminiProvider
        return GeminiProvider, "gemini-2.0-flash", "Gemini 2.0 Flash", 0.0
    if model_key == "openai":
        from models.openai_provider import OpenAIProvider
        return OpenAIProvider, "gpt-4o-mini", "GPT-4o-mini", 0.0
    if model_key == "deepseek":
        from models.deepseek_provider import DeepSeekProvider
        return DeepSeekProvider, "deepseek-chat", "DeepSeek Chat", 0.0
    if model_key == "claude":
        from models.claude_provider import ClaudeProvider
        return ClaudeProvider, "claude-haiku-4-5", "Claude Haiku 4.5", 1.5
    if model_key == "llama":
        from models.lm_studio_provider import LMStudioProvider
        return LMStudioProvider, "meta-llama-3.1-8b-instruct", "Llama 3.1 8B (Local)", 0.0
    if model_key == "deepseek-local":
        from models.lm_studio_provider import LMStudioProvider
        return LMStudioProvider, "deepseek/deepseek-r1-0528-qwen3-8b", "Deepseek-r1-0528-qwen3-8b (Local)", 0.0
    if model_key == "gemma":
        from models.lm_studio_provider import LMStudioProvider
        return LMStudioProvider, "google/gemma-3-9b-it", "Gemma 9B (Local)", 0.0
    raise ValueError(f"Unknown model_key '{model_key}'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PHASE2_DATA_DIR = PROJECT_ROOT / "data" / "phase2"

_LANG_DISPLAY = {
    "arabic":  "Arabic (Egyptian dialect)",
    "urdu":    "Roman Urdu",
    "chinese": "Simplified Chinese (Weibo)",
}

SAMPLE_PATTERN = "{lang}_5000samples_seed42.json"


# ── translation lookup (copy of runner's loader, with the size-based glob) ────

def load_translations_for_exp3(lang: str) -> Dict[str, str]:
    """Build {original_post_text: english_translation}. Largest file wins."""
    lookup: Dict[str, str] = {}

    # Primary: main sample file (Arabic / Chinese have translations baked in)
    sample_path = PHASE2_DATA_DIR / SAMPLE_PATTERN.format(lang=lang)
    if sample_path.exists():
        with open(sample_path, encoding="utf-8") as f:
            data = json.load(f)
        for s in data["samples"]:
            if s.get("translation") and not s.get("translation_failed"):
                post_key = s.get("post") or s.get("original", "")
                if post_key:
                    lookup[post_key] = s["translation"]
        if lookup:
            logger.info(f"[{lang}] {len(lookup)} translations loaded from {sample_path.name}")
            return lookup

    # Fallback 1: translated/ subdir (biggest file first)
    translated_dir = PHASE2_DATA_DIR / "translated"
    if translated_dir.exists():
        for p in sorted(
            translated_dir.glob(f"{lang}_*samples_seed42_translated.json"),
            key=lambda x: x.stat().st_size, reverse=True,
        ):
            with open(p, encoding="utf-8") as f:
                td = json.load(f)
            for s in td.get("samples", []):
                if s.get("translation") and not s.get("translation_failed"):
                    lookup[s["original"]] = s["translation"]
            if lookup:
                logger.info(f"[{lang}] {len(lookup)} translations loaded from {p.name}")
                return lookup

    # Fallback 2: translation_progress checkpoint (biggest first)
    progress_dir = PHASE2_DATA_DIR / "translation_progress"
    if progress_dir.exists():
        for p in sorted(
            progress_dir.glob(f"{lang}_translation_*s_seed42.json"),
            key=lambda x: x.stat().st_size, reverse=True,
        ):
            with open(p, encoding="utf-8") as f:
                td = json.load(f)
            for s in td.get("items", []):
                if s.get("translation") and not s.get("translation_failed"):
                    lookup[s["original"]] = s["translation"]
            if lookup:
                logger.info(f"[{lang}] {len(lookup)} translations loaded from checkpoint {p.name}")
                return lookup

    logger.warning(f"[{lang}] No English translations found on disk.")
    return {}


# ── parser + single-entry call (mirrors runner.py) ────────────────────────────

def _parse_exp3_response(raw_response: str) -> dict:
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    keywords = [k.strip() for k in lines[0].split(",")] if lines else []
    agreement_raw = lines[1].lower().strip() if len(lines) >= 2 else ""
    if agreement_raw.startswith("yes"):
        agreement = "yes"
    elif agreement_raw.startswith("no"):
        agreement = "no"
    else:
        agreement = agreement_raw
    return {"keywords": keywords, "agreement": agreement}


def _classify_one(provider, entry: dict, language: str) -> dict:
    """Call the model on a single repaired entry and return the merged record."""
    prompt_template = PROMPTS["v3_exp3"]
    translation = entry["translation"]
    filled = (
        prompt_template
        .replace("{prediction}", entry.get("prediction", ""))
        .replace("{original_language}", entry.get("original_language") or _LANG_DISPLAY.get(language, language))
    )
    result = provider.classify(post=translation, prompt_template=filled)
    parsed = _parse_exp3_response(result["raw_response"])
    return {
        **entry,
        "keywords_exp3":     parsed["keywords"],
        "agreement":         parsed["agreement"],
        "raw_response_exp3": result["raw_response"],
        "error_exp3":        result["error"],
    }


# ── metrics block (recompute after repair) ────────────────────────────────────

def _recompute_metrics(results: List[dict]) -> dict:
    """Recompute the same metrics block the runner writes out."""
    tp = fp = tn = fn = 0
    unclear = errors = 0
    classified = 0
    agreed = 0
    total_agreement = 0

    for r in results:
        pred = r.get("prediction", "")
        gt = r.get("ground_truth", "")
        if pred == "unclear":
            unclear += 1
        elif pred == "error":
            errors += 1
        elif pred in ("depressed", "not depressed"):
            classified += 1
            if pred == "depressed" and gt == "depressed":
                tp += 1
            elif pred == "depressed" and gt == "not depressed":
                fp += 1
            elif pred == "not depressed" and gt == "not depressed":
                tn += 1
            elif pred == "not depressed" and gt == "depressed":
                fn += 1

        agr = r.get("agreement")
        if agr in ("yes", "no"):
            total_agreement += 1
            if agr == "yes":
                agreed += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / total if total else 0.0
    agreement_rate = agreed / total_agreement if total_agreement else 0.0

    return {
        "confusion_matrix": {
            "true_positives":  tp, "false_positives": fp,
            "true_negatives":  tn, "false_negatives": fn,
        },
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "accuracy":          round(accuracy, 4),
        "total_samples":     len(results),
        "total_classified":  classified,
        "unclear_responses": unclear,
        "error_responses":   errors,
        "agreement_rate":    round(agreement_rate, 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill Exp3 entries that were wrongly marked 'no_translation'."
    )
    parser.add_argument("--file", type=Path, required=True,
                        help="Path to the exp3 result JSON to repair.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between model calls (default: 0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change, but don't call the model or write anything.")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"  File not found: {args.file}")
        sys.exit(1)

    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)

    metadata  = data.get("metadata", {})
    results   = data.get("results", [])
    model_key = metadata.get("model", "")
    language  = metadata.get("language", "")

    # Count broken rows
    no_trans_rows = [r for r in results if r.get("agreement") == "no_translation"]

    print(f"\n  File:              {args.file.name}")
    print(f"  Model:             {model_key}  |  Language: {language}")
    print(f"  Total rows:        {len(results)}")
    print(f"  'no_translation':  {len(no_trans_rows)}")

    if not no_trans_rows:
        print("\n  Nothing to repair.")
        sys.exit(0)

    # Build the correct translation lookup
    lookup = load_translations_for_exp3(language)
    if not lookup:
        print(f"\n  No translations found on disk for '{language}'. Cannot repair.")
        sys.exit(1)

    # Separate actually-fixable vs real gaps
    fixable = [r for r in no_trans_rows if r.get("post_full", "") in lookup]
    real_gap = [r for r in no_trans_rows if r.get("post_full", "") not in lookup]

    print(f"  Fixable (in lookup): {len(fixable)}")
    print(f"  Real gap (missing) : {len(real_gap)}  (will be left as 'no_translation')")

    if not fixable:
        print("\n  Nothing fixable — all 'no_translation' rows are real gaps.")
        sys.exit(0)

    if args.dry_run:
        print("\n  --dry-run set. No model calls made, no file written.")
        sys.exit(0)

    try:
        provider_cls, default_model, display_name, provider_delay = _load_provider(model_key)
    except (ValueError, ImportError) as e:
        print(f"\n  Cannot load provider for '{model_key}': {e}")
        sys.exit(1)

    delay = max(args.delay, provider_delay)

    try:
        api_key = get_api_key(model_key)
    except EnvironmentError as e:
        print(f"\n  {e}")
        sys.exit(1)

    provider = provider_cls(api_key=api_key, model_name=default_model)

    confirm = input(f"\n  Repair {len(fixable)} entries with {display_name}? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        sys.exit(0)

    # Build new records for the fixable rows
    lang_display = _LANG_DISPLAY.get(language, language.capitalize())
    repaired_by_idx: Dict[int, dict] = {}

    for i, r in enumerate(fixable, 1):
        idx = r.get("index")
        post_full = r.get("post_full", "")
        entry = {
            **r,
            "translation":       lookup[post_full],
            "original_language": r.get("original_language") or lang_display,
            # Clear the stale error state before the retry
            "error_exp3":        "",
        }
        logger.info(f"[{model_key}][{language}] Repairing {i}/{len(fixable)} (index={idx})...")

        try:
            repaired = _classify_one(provider, entry, language)
        except Exception as e:
            logger.error(f"  failed on index={idx}: {e}")
            repaired = {**entry, "keywords_exp3": [], "agreement": "no_translation",
                        "raw_response_exp3": "", "error_exp3": f"repair_failed: {e}"}

        repaired_by_idx[idx] = repaired

        if i < len(fixable) and delay > 0:
            time.sleep(delay)

    # Merge back — only repaired rows overwrite
    merged_by_idx = {r.get("index"): r for r in results}
    merged_by_idx.update(repaired_by_idx)
    merged = sorted(merged_by_idx.values(), key=lambda x: x.get("index", 0))

    # Recompute metrics
    new_metrics = _recompute_metrics(merged)

    # Report
    still_nt = sum(1 for r in merged if r.get("agreement") == "no_translation")
    new_yes  = sum(1 for r in merged if r.get("agreement") == "yes")
    new_no   = sum(1 for r in merged if r.get("agreement") == "no")
    print(f"\n  Repair summary:")
    print(f"    Repaired:           {len(repaired_by_idx)}")
    print(f"    Still 'no_translation' (real gaps): {still_nt}")
    print(f"    New 'yes'/'no' total: {new_yes} / {new_no}")
    print(f"    New agreement_rate: {new_metrics['agreement_rate']}")

    # Save timestamped output next to original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_key.replace("/", "_").replace(":", "_")
    out_dir   = args.file.parent
    out_path  = out_dir / f"{safe_name}_{language}_{timestamp}.json"

    payload = {
        "metadata": {
            "model":         model_key,
            "language":      language,
            "experiment":    3,
            "timestamp":     timestamp,
            "sample_size":   len(merged),
            "repair_source": args.file.name,
            "repaired_rows": len(repaired_by_idx),
        },
        "metrics": new_metrics,
        "results": merged,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
