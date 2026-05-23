"""Rerun failed (error/unclear) entries from a completed Experiment 1 result file.

Usage:
    python scripts/rerun_failed.py                          # interactive file picker
    python scripts/rerun_failed.py --file results/phase2/experiment1/claude_arabic_20260501_120000.json
    python scripts/rerun_failed.py --file <path> --delay 1.0
    python scripts/rerun_failed.py --file <path> --unclear  # unclear only
    python scripts/rerun_failed.py --file <path> --errors   # errors only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_api_key
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider, ClaudeProvider, LMStudioProvider
from evaluation.metrics import EvaluationMetrics
from evaluation.prompts import PROMPTS, LANGUAGE_DEFAULT_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXP1_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment1"

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
}


def discover_results() -> list:
    if not EXP1_RESULTS_DIR.exists():
        return []
    found = []
    for p in sorted(EXP1_RESULTS_DIR.glob("*.json")):
        if p.name.startswith("comparison_"):
            continue
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        if not (parts[-2].isdigit() and parts[-1].isdigit()):
            continue
        lang      = parts[-3]
        model_key = "_".join(parts[:-3])
        found.append((model_key, lang, p))
    return found


def pick_result_file() -> Path:
    results = discover_results()
    if not results:
        print("  No completed result files found in", EXP1_RESULTS_DIR)
        sys.exit(1)
    print(f"\n  Completed result files in {EXP1_RESULTS_DIR}:")
    print("  " + "-" * 60)
    for i, (model_key, lang, path) in enumerate(results, 1):
        name = MODELS[model_key]["name"] if model_key in MODELS else model_key
        print(f"  [{i}] {name} x {lang.capitalize():<10}  {path.name}")
    print()
    while True:
        choice = input("  Select file number: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(results):
                return results[idx - 1][2]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(results)}.")


def identify_failed(results: list, rerun_errors: bool, rerun_unclear: bool) -> list:
    return [
        r for r in results
        if (rerun_errors  and r.get("prediction") == "error")
        or (rerun_unclear and r.get("prediction") == "unclear")
    ]


def rerun_entries(provider, failed: list, language: str, prompt_template: str,
                  model_name: str, delay: float) -> dict:
    total, new_results = len(failed), {}
    for i, entry in enumerate(failed, 1):
        idx  = entry["index"]
        post = entry.get("post_full") or entry.get("post_preview", "")
        logger.info(f"[{model_name}][{language}] Rerunning index {idx} ({i}/{total})...")
        result = provider.classify(post=post, prompt_template=prompt_template)
        new_results[idx] = {**entry, "prediction": result["prediction"],
                            "raw_response": result["raw_response"], "error": result["error"]}
        if i < total:
            time.sleep(delay)
    return new_results


def merge_results(original: list, new_results: dict) -> list:
    return [new_results.get(e["index"], e) for e in original]


def save_results(results: list, metrics: dict, model_key: str, language: str,
                 experiment: int, original_path: Path) -> Path:
    EXP1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_key.replace("/", "_").replace(":", "_")
    path      = EXP1_RESULTS_DIR / f"{safe_name}_{language}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": model_key, "language": language,
                "experiment": experiment, "timestamp": ts,
                "sample_size": len(results), "rerun_source": original_path.name,
            },
            "metrics": metrics, "results": results,
        }, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file",    type=Path, help="Path to result JSON to fix")
    parser.add_argument("--delay",   type=float, default=0.0)
    parser.add_argument("--unclear", action="store_true", help="Rerun unclear only")
    parser.add_argument("--errors",  action="store_true", help="Rerun errors only")
    args = parser.parse_args()

    rerun_errors  = args.errors  or not args.unclear
    rerun_unclear = args.unclear or not args.errors

    result_path = args.file or pick_result_file()
    if not result_path.exists():
        print(f"  File not found: {result_path}"); sys.exit(1)

    data       = json.loads(result_path.read_text(encoding="utf-8"))
    metadata   = data.get("metadata", {})
    results    = data.get("results", [])
    model_key  = metadata.get("model", "")
    language   = metadata.get("language", "")
    experiment = metadata.get("experiment", 1)

    failed = identify_failed(results, rerun_errors, rerun_unclear)
    print(f"\n  File:      {result_path.name}")
    print(f"  Model:     {model_key}  |  Language: {language}")
    print(f"  Total:     {len(results)}  |  To rerun: {len(failed)}")

    if not failed:
        print("\n  Nothing to rerun — all entries are already classified.")
        sys.exit(0)
    if model_key not in MODELS:
        print(f"\n  Unknown model key '{model_key}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    info  = MODELS[model_key]
    delay = max(args.delay, info.get("delay", 0))

    try:
        api_key = get_api_key(model_key)
    except EnvironmentError as e:
        print(f"\n  {e}"); sys.exit(1)

    provider        = info["class"](api_key=api_key, model_name=info["default_model"])
    prompt_key      = LANGUAGE_DEFAULT_PROMPTS.get(language)
    prompt_template = PROMPTS[prompt_key] if prompt_key else list(PROMPTS.values())[-1]

    if input(f"\n  Rerun {len(failed)} entries with {info['name']}? (Y/n): ").strip().lower() == "n":
        print("  Aborted."); sys.exit(0)

    new_results  = rerun_entries(provider, failed, language, prompt_template, info["name"], delay)
    still_failed = sum(1 for r in new_results.values() if r.get("prediction") in ("error", "unclear"))
    print(f"\n  Resolved: {len(new_results) - still_failed}/{len(new_results)}  |  Still failed: {still_failed}")

    merged   = merge_results(results, new_results)
    metrics  = EvaluationMetrics.compute(merged)
    out_path = save_results(merged, metrics, model_key, language, experiment, result_path)
    print(EvaluationMetrics.format_report(metrics, f"{info['name']} — {language.capitalize()} (after rerun)"))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
