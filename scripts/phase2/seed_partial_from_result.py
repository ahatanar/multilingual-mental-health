"""
Convert a completed (partial) result file into a .partial.json checkpoint
so the main runner can resume from where it left off.

Usage:
    python scripts/phase2/seed_partial_from_result.py \
        results/phase2/experiment1/deepseek-local_arabic_20260330_043919.json \
        --total 5000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP1_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment1"

# Must match the model_name the LMStudioProvider reports (provider.name = model_name)
MODEL_KEY_TO_MODEL_NAME = {
    "deepseek-local": "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama":          "meta-llama-3.1-8b-instruct",
    "qwen":           "qwen/qwen3.5-9b",
    "gemini":         "gemini-2.0-flash",
    "openai":         "gpt-4o-mini",
    "deepseek":       "deepseek-chat",
    "claude":         "claude-haiku-4-5",
}


def main():
    parser = argparse.ArgumentParser(
        description="Seed a .partial.json from a completed result file so the runner can resume."
    )
    parser.add_argument("result_file", type=Path, help="Path to the completed result JSON file")
    parser.add_argument("--total", type=int, default=5000,
                        help="Total number of samples in the full dataset (default: 5000)")
    args = parser.parse_args()

    if not args.result_file.exists():
        print(f"File not found: {args.result_file}")
        sys.exit(1)

    with open(args.result_file, encoding="utf-8") as f:
        data = json.load(f)

    meta     = data.get("metadata", {})
    results  = data.get("results", [])
    model_key = meta.get("model", "")
    language  = meta.get("language", "")

    if not model_key or not language:
        print("Could not read model/language from file metadata.")
        sys.exit(1)

    if model_key not in MODEL_KEY_TO_MODEL_NAME:
        print(f"Unknown model key '{model_key}'. Add it to MODEL_KEY_TO_MODEL_NAME in this script.")
        sys.exit(1)

    model_name = MODEL_KEY_TO_MODEL_NAME[model_key]
    safe_name  = model_name.replace("/", "_").replace(":", "_")
    partial_path = EXP1_RESULTS_DIR / f"{safe_name}_{language}.partial.json"

    completed = len(results)
    if completed >= args.total:
        print(f"File already has {completed} entries — nothing to resume (total={args.total}).")
        sys.exit(0)

    if partial_path.exists():
        existing_completed = json.load(open(partial_path))["metadata"].get("completed", 0)
        answer = input(
            f"  Partial file already exists with {existing_completed} entries.\n"
            f"  Overwrite with {completed} entries from {args.result_file.name}? (y/N): "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    partial = {
        "metadata": {
            "model":      model_name,
            "language":   language,
            "status":     "partial",
            "completed":  completed,
            "total":      args.total,
            "last_saved": datetime.now().isoformat(),
        },
        "results": results,
    }

    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)

    print(f"\n  Created partial checkpoint:")
    print(f"    {partial_path}")
    print(f"    {completed}/{args.total} entries already done")
    print(f"\n  Now run the normal runner and select deepseek-local + arabic:")
    print(f"    python scripts/phase2/runner.py")
    print(f"  It will automatically resume from entry {completed + 1}.")


if __name__ == "__main__":
    main()
