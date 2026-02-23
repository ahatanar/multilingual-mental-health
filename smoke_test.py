"""
Smoke test: send 1 classification request per language per API.

Verifies all API keys work, all providers can classify, and results save correctly.
Run this before committing to a full 500-sample evaluation.

Usage:
    py smoke_test.py
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from config import get_api_key
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider, ClaudeProvider
from evaluation.prompts import CLASSIFICATION_PROMPT
from models.base import ModelProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLED_DIR = os.path.join(os.path.dirname(__file__), "data", "sampled")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PROVIDERS = {
    "gemini":   {"class": GeminiProvider,   "model": "gemini-2.0-flash"},
    "deepseek": {"class": DeepSeekProvider, "model": "deepseek-chat"},
    "openai":   {"class": OpenAIProvider,   "model": "gpt-4o-mini"},
    "claude":   {"class": ClaudeProvider,   "model": "claude-haiku-4-5"},
}


def load_one_sample(language: str) -> dict:
    """Load the first sample from a pre-sampled language file."""
    path = os.path.join(SAMPLED_DIR, f"{language}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No sampled data for '{language}'. Run prepare_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"][0]


def main():
    parser = argparse.ArgumentParser(description="Smoke test — 1 request per language per API")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()),
                        help="Test only this provider (default: all)")
    args = parser.parse_args()

    providers_to_test = {args.provider: PROVIDERS[args.provider]} if args.provider else PROVIDERS

    print("\n" + "=" * 60)
    print("  🧪 Smoke Test — 1 request per language per API")
    print("=" * 60)

    # Discover languages
    if not os.path.isdir(SAMPLED_DIR):
        print("\n❌ No sampled data. Run `py prepare_data.py` first.")
        return

    languages = sorted(
        f.replace(".json", "")
        for f in os.listdir(SAMPLED_DIR)
        if f.endswith(".json")
    )
    print(f"\n  Languages: {', '.join(languages)}")
    print(f"  Providers: {', '.join(providers_to_test.keys())}")

    # Load one sample per language
    samples = {}
    for lang in languages:
        try:
            samples[lang] = load_one_sample(lang)
            print(f"  ✓ {lang}: loaded sample ({samples[lang]['ground_truth']})")
        except FileNotFoundError as e:
            print(f"  ❌ {lang}: {e}")

    if not samples:
        print("\n❌ No samples loaded. Aborting.")
        return

    # Test each provider × language
    results = []
    total_pass = 0
    total_fail = 0
    total_skip = 0

    for prov_key, prov_info in providers_to_test.items():
        print(f"\n{'─'*50}")
        print(f"  🔌 Provider: {prov_key} ({prov_info['model']})")
        print(f"{'─'*50}")

        # Check API key
        try:
            api_key = get_api_key(prov_key)
        except EnvironmentError as e:
            print(f"  ⚠ SKIPPED — {e}")
            for lang in samples:
                results.append({
                    "provider": prov_key, "language": lang,
                    "status": "SKIP", "reason": "No API key",
                })
                total_skip += 1
            continue

        # Init provider
        try:
            provider = prov_info["class"](api_key=api_key, model_name=prov_info["model"])
        except Exception as e:
            print(f"  ❌ Failed to init provider: {e}")
            for lang in samples:
                results.append({
                    "provider": prov_key, "language": lang,
                    "status": "FAIL", "reason": f"Init error: {e}",
                })
                total_fail += 1
            continue

        # Classify one sample per language
        for lang, sample in samples.items():
            try:
                result = provider.classify(
                    post=sample["post"],
                    prompt_template=CLASSIFICATION_PROMPT,
                )
                pred = result["prediction"]
                gt = sample["ground_truth"]

                # Check if the API actually returned a valid prediction
                if pred == "error" or not result["raw_response"]:
                    print(f"  {lang:<10} ❌ API ERROR (returned '{pred}', raw=\"{result['raw_response'][:40]}\")")
                    results.append({
                        "provider": prov_key, "language": lang,
                        "status": "FAIL", "reason": "API returned error prediction",
                        "prediction": pred,
                        "raw_response": result["raw_response"],
                    })
                    total_fail += 1
                    continue

                match = "✓" if pred == gt else "✗"
                correct = pred == gt

                print(f"  {lang:<10} gt={gt:<15} pred={pred:<15} {match}  raw=\"{result['raw_response'][:40]}\"")
                results.append({
                    "provider": prov_key, "language": lang,
                    "status": "PASS", "prediction": pred,
                    "ground_truth": gt, "correct": correct,
                    "raw_response": result["raw_response"],
                })
                total_pass += 1

            except Exception as e:
                print(f"  {lang:<10} ❌ ERROR: {e}")
                results.append({
                    "provider": prov_key, "language": lang,
                    "status": "FAIL", "reason": str(e),
                })
                total_fail += 1

    # Save smoke test results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"smoke_test_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {"pass": total_pass, "fail": total_fail, "skip": total_skip},
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  🧪 Smoke Test Results")
    print(f"{'='*60}")
    print(f"  ✅ Pass: {total_pass}    ❌ Fail: {total_fail}    ⚠ Skip: {total_skip}")
    print(f"  💾 Saved: {out_path}")

    if total_fail == 0 and total_skip == 0:
        print(f"\n  🎉 All clear! Safe to run full evaluation.")
    elif total_fail > 0:
        print(f"\n  ⚠ Some tests failed. Check API keys and provider configs.")
    print()


if __name__ == "__main__":
    main()
