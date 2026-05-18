"""
Translate Urdu sampled data to English and save as data/sampled/urdu_english.json.

This creates a parallel dataset with the same labels/ground truth but
English-translated posts, so we can compare model performance on
native Urdu vs translated English.

Usage:
    py translate_urdu.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

from googletrans import Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "phase1", "sampled", "urdu.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "phase1", "sampled", "urdu_english.json")

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def translate_text(translator: Translator, text: str) -> str:
    """Translate a single text with retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            coro = translator.translate(text, src="ur", dest="en")
            result = asyncio.get_event_loop().run_until_complete(coro)
            return result.text
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                translator = Translator()
            else:
                raise
    return text


def main():
    # Load source data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Translating {len(samples)} Urdu posts to English...")

    translator = Translator()
    translated_samples = []
    errors = 0

    for i, sample in enumerate(samples):
        try:
            english_post = translate_text(translator, sample["post"])
        except Exception as e:
            logger.error(f"Failed to translate post {i}: {e}")
            english_post = sample["post"]  # fallback to original
            errors += 1

        translated_samples.append({
            "post": english_post,
            "original_post": sample["post"],
            "label": sample["label"],
            "ground_truth": sample["ground_truth"],
            "word_count": len(english_post.split()),
            **({"severity": sample["severity"]} if "severity" in sample else {}),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)} translated ({errors} errors)")

    # Save
    output = {
        "metadata": {
            "language": "urdu_english",
            "sample_size": len(translated_samples),
            "requested_size": data["metadata"]["requested_size"],
            "seed": data["metadata"]["seed"],
            "created": datetime.now().isoformat(),
            "source_total": data["metadata"]["source_total"],
            "label_distribution": data["metadata"]["label_distribution"],
            "source_language": "urdu",
            "translated_to": "english",
            "translation_method": "googletrans",
        },
        "samples": translated_samples,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(translated_samples)} posts translated ({errors} errors)")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nNext: py -X utf8 runner.py  (select 'urdu_english')")


if __name__ == "__main__":
    main()
