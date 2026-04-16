"""
Rerun only failed (LM Studio error) entries from a completed Experiment 3 result file.

Entries with "no English translation available" or "skipped" are intentionally
left alone — those are data gaps, not transient errors.

Usage:
    python scripts/phase2/rerun_failed_exp3.py --file results/phase2/experiment3/Local Models/gemma/gemma_chinese_20260413_120000.json
    python scripts/phase2/rerun_failed_exp3.py --file <path> --delay 0.5
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_api_key
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider, ClaudeProvider, LMStudioProvider
from evaluation.prompts import PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXP3_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment3"

MODELS = {
    "gemini":         {"class": GeminiProvider,   "name": "Gemini 2.0 Flash",                   "default_model": "gemini-2.0-flash"},
    "openai":         {"class": OpenAIProvider,   "name": "GPT-4o-mini",                         "default_model": "gpt-4o-mini"},
    "deepseek":       {"class": DeepSeekProvider, "name": "DeepSeek Chat",                       "default_model": "deepseek-chat"},
    "claude":         {"class": ClaudeProvider,   "name": "Claude Haiku 4.5",                    "default_model": "claude-haiku-4-5",
                       "max_workers": 1, "delay": 1.5},
    "llama":          {"class": LMStudioProvider, "name": "Llama 3.1 8B (Local)",                "default_model": "meta-llama-3.1-8b-instruct",
                       "max_workers": 1, "delay": 0},
    "qwen":           {"class": LMStudioProvider, "name": "Qwen 3.5 9B (Local)",                 "default_model": "qwen/qwen3.5-9b",
                       "max_workers": 1, "delay": 0},
    "deepseek-local": {"class": LMStudioProvider, "name": "Deepseek-r1-0528-qwen3-8b (Local)",  "default_model": "deepseek/deepseek-r1-0528-qwen3-8b",
                       "max_workers": 1, "delay": 0},
    "gemma":          {"class": LMStudioProvider, "name": "Gemma 9B (Local)",                    "default_model": "google/gemma-3-9b-it",
                       "max_workers": 1, "delay": 0},
}

# Errors that are data gaps — not worth retrying
_SKIP_ERRORS = {"no English translation available", "skipped", "skipped (no valid Exp 1 prediction)"}


def _is_retryable(entry: dict) -> bool:
    """True if the entry failed due to a transient error (e.g. LM Studio dropped)."""
    err = entry.get("error_exp3")
    if not err:
        return False
    return err not in _SKIP_ERRORS


def _parse_exp3_response(raw_response: str) -> dict:
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    keywords      = [k.strip() for k in lines[0].split(",")] if lines else []
    agreement_raw = lines[1].lower().strip() if len(lines) >= 2 else ""
    if agreement_raw.startswith("yes"):
        agreement = "yes"
    elif agreement_raw.startswith("no"):
        agreement = "no"
    else:
        agreement = agreement_raw
    return {"keywords": keywords, "agreement": agreement}


def rerun_entries(provider, failed: list, language: str, delay: float) -> dict:
    """
    Retry each failed exp3 entry. Returns {index: updated_entry}.
    The translation and prediction are already embedded in each entry.
    """
    prompt_template = PROMPTS["v3_exp3"]
    total = len(failed)
    new_results = {}

    for i, entry in enumerate(failed, 1):
        idx         = entry["index"]
        translation = entry.get("translation", "")
        prediction  = entry.get("prediction", "")
        orig_lang   = entry.get("original_language", language)

        logger.info(f"[gemma][{language}] Retrying index {idx} ({i}/{total})...")

        if not translation:
            new_results[idx] = {**entry, "keywords_exp3": [], "agreement": "no_translation",
                                "raw_response_exp3": "", "error_exp3": "no English translation available"}
            continue

        filled = (prompt_template
                  .replace("{prediction}", prediction)
                  .replace("{original_language}", orig_lang))

        result = provider.classify(post=translation, prompt_template=filled)
        parsed = _parse_exp3_response(result["raw_response"])

        new_results[idx] = {
            **entry,
            "keywords_exp3":     parsed["keywords"],
            "agreement":         parsed["agreement"],
            "raw_response_exp3": result["raw_response"],
            "error_exp3":        result["error"],
        }

        if i < total:
            time.sleep(delay)

    return new_results


def compute_consistency_stats(results: list) -> dict:
    agreed      = sum(1 for r in results if r.get("agreement") == "yes")
    disagreed   = sum(1 for r in results if r.get("agreement") == "no")
    no_trans    = sum(1 for r in results if r.get("agreement") == "no_translation")
    skipped     = sum(1 for r in results if r.get("agreement") in ("skipped", ""))
    total_valid = agreed + disagreed
    return {
        "agreed":        agreed,
        "disagreed":     disagreed,
        "no_translation": no_trans,
        "skipped":       skipped,
        "total_valid":   total_valid,
        "agreement_pct": round(agreed / total_valid * 100, 2) if total_valid else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Rerun LM Studio error entries from an Experiment 3 result file.")
    parser.add_argument("--file",  type=Path, required=True, help="Path to the exp3 result JSON to fix")
    parser.add_argument("--delay", type=float, default=0.0,  help="Seconds between retries (default: 0)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"  File not found: {args.file}")
        sys.exit(1)

    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results  = data.get("results", [])
    model_key = metadata.get("model", "")
    language  = metadata.get("language", "")

    failed        = [r for r in results if _is_retryable(r)]
    no_trans      = sum(1 for r in results if r.get("error_exp3") == "no English translation available")
    skipped_count = sum(1 for r in results if r.get("agreement") == "skipped")

    print(f"\n  File:             {args.file.name}")
    print(f"  Model:            {model_key}  |  Language: {language}")
    print(f"  Total entries:    {len(results)}")
    print(f"  Retryable errors: {len(failed)}")
    print(f"  No translation:   {no_trans}  (skipped — data gap)")
    print(f"  Skipped (exp1):   {skipped_count}  (skipped — no valid exp1 prediction)")

    if not failed:
        print("\n  Nothing to retry.")
        sys.exit(0)

    if model_key not in MODELS:
        print(f"\n  Unknown model key '{model_key}'. Add it to MODELS dict in this script.")
        sys.exit(1)

    info = MODELS[model_key]
    delay = max(args.delay, info.get("delay", 0))

    try:
        api_key = get_api_key(model_key)
    except EnvironmentError as e:
        print(f"\n  {e}")
        sys.exit(1)

    provider = info["class"](api_key=api_key, model_name=info["default_model"])

    confirm = input(f"\n  Retry {len(failed)} entries with {info['name']}? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        sys.exit(0)

    new_results = rerun_entries(provider, failed, language, delay)

    still_failed = sum(1 for r in new_results.values() if _is_retryable(r))
    resolved     = len(new_results) - still_failed
    print(f"\n  Resolved:      {resolved}/{len(new_results)}")
    print(f"  Still failed:  {still_failed}/{len(new_results)}")

    # Merge back
    idx_map = {r["index"]: r for r in results}
    idx_map.update(new_results)
    merged = sorted(idx_map.values(), key=lambda x: x.get("index", 0))

    stats = compute_consistency_stats(merged)
    print(f"\n  Consistency after retry:")
    print(f"    Agreed:      {stats['agreed']} / {stats['total_valid']}  ({stats['agreement_pct']}%)")
    print(f"    Disagreed:   {stats['disagreed']}")
    print(f"    No translation: {stats['no_translation']}")

    # Save as new timestamped file next to the original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_key.replace("/", "_").replace(":", "_")
    out_dir   = args.file.parent
    out_path  = out_dir / f"{safe_name}_{language}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model":        model_key,
                "language":     language,
                "experiment":   3,
                "timestamp":    timestamp,
                "sample_size":  len(merged),
                "rerun_source": args.file.name,
            },
            "consistency": stats,
            "results": merged,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
