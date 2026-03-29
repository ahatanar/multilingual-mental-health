"""
Phase 2 main evaluation runner.

Supports Experiment 1 (monolingual classification: Arabic / Urdu / Chinese).
Experiments 2 and 3 are shown in the menu but not yet implemented.

Prerequisites:
    python scripts/phase2/prepare_experiment1.py

Usage:
    python scripts/phase2/runner.py                  # interactive mode
    python scripts/phase2/runner.py --fresh          # ignore partial results
    python scripts/phase2/runner.py --delay 2.0      # slower API calls
    python scripts/phase2/runner.py --workers 3      # parallel requests
    python scripts/phase2/runner.py --prompt v2      # override prompt version
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_api_key, DEFAULT_MODELS
from models import GeminiProvider, DeepSeekProvider, OpenAIProvider, ClaudeProvider, LMStudioProvider
from evaluation.metrics import EvaluationMetrics
from evaluation.prompts import PROMPTS, LANGUAGE_DEFAULT_PROMPTS, LANGUAGE_DEFAULT_PROMPTS_EXP2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Registry ─────────────────────────────────────────────────────────────────

MODELS = {
    "gemini":   {"class": GeminiProvider,   "name": "Gemini 2.0 Flash",       "default_model": "gemini-2.0-flash"},
    "deepseek": {"class": DeepSeekProvider, "name": "DeepSeek Chat",           "default_model": "deepseek-chat"},
    "openai":   {"class": OpenAIProvider,   "name": "ChatGPT (GPT-4o-mini)",   "default_model": "gpt-4o-mini"},
    "claude":   {"class": ClaudeProvider,   "name": "Claude Haiku 4.5",        "default_model": "claude-haiku-4-5",
                 "max_workers": 1, "delay": 1.5},
    "lmstudio": {"class": LMStudioProvider, "name": "LM Studio (Local)",         "default_model": "local-model",
                 "max_workers": 1, "delay": 0},
}

# ── Paths ─────────────────────────────────────────────────────────────────────

PHASE2_DATA_DIR = PROJECT_ROOT / "data" / "phase2"
EXP1_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment1"
EXP2_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment2"

# Keep RESULTS_DIR pointing to exp1 for backward-compat with existing partial files
RESULTS_DIR = EXP1_RESULTS_DIR

# Languages for Experiments 1 & 2 (in display order)
EXP1_LANGUAGES = ["arabic", "urdu", "chinese"]
SAMPLE_PATTERN  = "{lang}_5000samples_seed42.json"

SAVE_EVERY = 10


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_experiment1_languages() -> Dict[str, Optional[Path]]:
    """
    Returns {language: path_or_None} for all Experiment 1 languages.
    Path is None if the prepared sample file doesn't exist yet.
    """
    result = {}
    for lang in EXP1_LANGUAGES:
        path = PHASE2_DATA_DIR / SAMPLE_PATTERN.format(lang=lang)
        result[lang] = path if path.exists() else None
    return result


def load_samples(lang: str) -> List[Dict]:
    path = PHASE2_DATA_DIR / SAMPLE_PATTERN.format(lang=lang)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared data for '{lang}'.\n"
            f"  Run: python scripts/phase2/prepare_experiment1.py\n"
            f"  Expected: {path}"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples  = data["samples"]
    metadata = data["metadata"]
    logger.info(
        f"[{lang}] Loaded {len(samples)} samples "
        f"(seed={metadata.get('seed')}, created={metadata.get('created', '?')})"
    )
    return samples


# ── Checkpoint helpers (identical logic to phase1/runner.py) ─────────────────

def _partial_path(model_name: str, language: str, out_dir: Path = None) -> Path:
    d = out_dir or EXP1_RESULTS_DIR
    return d / f"{model_name}_{language}.partial.json"


def _save_partial(results: List[Dict], model_name: str, language: str,
                  total: int, out_dir: Path = None) -> None:
    d = out_dir or EXP1_RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "model":      model_name,
            "language":   language,
            "status":     "partial",
            "completed":  len(results),
            "total":      total,
            "last_saved": datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(_partial_path(model_name, language, d), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _load_partial(model_name: str, language: str, out_dir: Path = None) -> List[Dict]:
    path = _partial_path(model_name, language, out_dir)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results   = data.get("results", [])
        completed = data["metadata"]["completed"]
        total     = data["metadata"]["total"]
        logger.info(f"Found partial results: {completed}/{total} completed")
        return results
    except Exception as e:
        logger.warning(f"Could not load partial results: {e}")
        return []


def _remove_partial(model_name: str, language: str, out_dir: Path = None) -> None:
    path = _partial_path(model_name, language, out_dir)
    if path.exists():
        path.unlink()


# ── Classification ────────────────────────────────────────────────────────────

def _classify_one(provider, sample: Dict, idx: int, total: int,
                  model_name: str, language: str, prompt_template: str) -> Dict:
    logger.info(f"[{model_name}][{language}] Classifying {idx}/{total}...")
    result = provider.classify(post=sample["post"], prompt_template=prompt_template)
    return {
        "index":        idx,
        "post_preview": sample["post"][:100] + ("..." if len(sample["post"]) > 100 else ""),
        "post_full":    sample["post"],
        "ground_truth": sample["ground_truth"],
        "prediction":   result["prediction"],
        "raw_response": result["raw_response"],
        "error":        result["error"],
        "word_count":   sample.get("word_count", 0),
    }


def _parse_exp2_response(raw_response: str) -> Dict:
    """
    Parse an Experiment 2 attribution response (2 lines).

    Expected format:
        word1, word2        ← original-language keywords
        trans1, trans2      ← one-word English translations

    Returns {"keywords": [...], "translations": [...]}.
    Missing or malformed lines produce empty lists.
    """
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    keywords     = [k.strip() for k in lines[0].split(",")] if len(lines) >= 1 else []
    translations = [t.strip() for t in lines[1].split(",")] if len(lines) >= 2 else []
    return {"keywords": keywords, "translations": translations}


def run_evaluation(provider, samples: List[Dict], language: str,
                   prompt_template: str,
                   delay: float = 1.0, fresh: bool = False,
                   workers: int = 1,
                   classify_fn=None,
                   results_dir: Path = None) -> List[Dict]:
    """Run classification with crash-resume support."""
    model_name = provider.name if hasattr(provider, "name") else str(provider)
    total = len(samples)

    out_dir = results_dir or EXP1_RESULTS_DIR

    if fresh:
        _remove_partial(model_name, language, out_dir)
        print("  Resetting — ignoring any previous partial results")
        results, start_idx = [], 0
    else:
        existing  = _load_partial(model_name, language, out_dir)
        start_idx = len(existing)
        if 0 < start_idx < total:
            print(f"  Resuming from {start_idx}/{total}")
            results = existing
        elif start_idx >= total:
            print(f"  Already completed {total}/{total} — skipping")
            return existing
        else:
            results, start_idx = [], 0

    remaining = list(range(start_idx, total))

    do_classify = classify_fn or _classify_one
    out_dir     = results_dir or EXP1_RESULTS_DIR

    if workers <= 1:
        for i in remaining:
            r = do_classify(provider, samples[i], i + 1, total,
                            model_name, language, prompt_template)
            results.append(r)
            if len(results) % SAVE_EVERY == 0:
                _save_partial(results, model_name, language, total, out_dir)
                logger.info(f"  Checkpoint: {len(results)}/{total}")
            if i < total - 1:
                time.sleep(delay)
    else:
        print(f"  Parallel mode: {workers} workers")
        result_slots: Dict[int, Dict] = {}

        for batch_start in range(0, len(remaining), workers):
            batch = remaining[batch_start:batch_start + workers]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(do_classify, provider, samples[i], i + 1,
                                    total, model_name, language, prompt_template): i
                    for i in batch
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        result_slots[i] = future.result()
                    except Exception as e:
                        logger.error(f"Worker error on {i+1}: {e}")
                        post_text = samples[i].get("post") or samples[i].get("post_full", "")
                        result_slots[i] = {
                            "index": i + 1, "post_preview": post_text[:100],
                            "post_full": post_text,
                            "ground_truth": samples[i].get("ground_truth", ""),
                            "prediction": "error", "raw_response": "", "error": str(e),
                            "word_count": samples[i].get("word_count", 0),
                        }
            for i in batch:
                results.append(result_slots[i])
            if len(results) % SAVE_EVERY == 0 or batch_start + workers >= len(remaining):
                _save_partial(results, model_name, language, total, out_dir)
                logger.info(f"  Checkpoint: {len(results)}/{total}")
            if batch_start + workers < len(remaining):
                time.sleep(delay)

    _remove_partial(model_name, language, out_dir)
    return results


def save_results(results: List[Dict], metrics: Dict,
                 model_name: str, language: str,
                 out_dir: Path = None, experiment: int = 1) -> Path:
    out_dir = out_dir or EXP1_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{model_name}_{language}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model":       model_name,
                "language":    language,
                "experiment":  experiment,
                "timestamp":   timestamp,
                "sample_size": len(results),
            },
            "metrics": metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    return path


# ── Interactive menus ─────────────────────────────────────────────────────────

def print_header():
    print("\n" + "=" * 58)
    print("  Multilingual Mental Health — Research Framework")
    print("=" * 58)


def select_experiment() -> int:
    """Top-level experiment selection menu."""
    print("\n  Select Experiment:")
    print("  " + "-" * 52)
    print("  [1]  Experiment 1 — Monolingual classification")
    print("       (Arabic / Urdu / Chinese, classification only)")
    print("  [2]  Experiment 2 — Classification + keyword extraction")
    print("       (same languages, outputs key words + translations per post)")
    print("  [3]  Experiment 3 — ...                          [COMING SOON]")
    print()

    while True:
        choice = input("  Enter experiment number: ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        print("  Please enter 1, 2, or 3.")


def select_models() -> List[str]:
    model_keys = list(MODELS.keys())
    print("\n  Available Models:")
    print("  " + "-" * 44)
    for i, key in enumerate(model_keys, 1):
        info = MODELS[key]
        print(f"  [{i}] {info['name']}  ({info['default_model']})")
    print(f"  [{len(model_keys) + 1}] All models")
    print()

    while True:
        choice = input("  Select model(s) (comma-separated, e.g. 1,2): ").strip()
        if not choice:
            continue
        if choice == str(len(model_keys) + 1):
            return model_keys
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(model_keys):
                    selected.append(model_keys[idx - 1])
                else:
                    print(f"  Invalid number: {idx}")
                    selected = []
                    break
            if selected:
                print(f"  Selected: {', '.join(MODELS[k]['name'] for k in selected)}")
                return selected
        except ValueError:
            print("  Please enter numbers separated by commas.")


def select_languages(available: Dict[str, Optional[Path]], label: str = "Experiment 1") -> List[str]:
    lang_keys = list(available.keys())
    print(f"\n  Available Languages ({label}):")
    print("  " + "-" * 44)
    for i, lang in enumerate(lang_keys, 1):
        path = available[lang]
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                n    = data["metadata"]["total_sampled"]
                seed = data["metadata"]["seed"]
                print(f"  [{i}] {lang.capitalize():<10}  ({n} samples, seed={seed})")
            except Exception:
                print(f"  [{i}] {lang.capitalize()}")
        else:
            print(f"  [{i}] {lang.capitalize():<10}  [not ready — run prepare_experiment1.py]")
    print(f"  [{len(lang_keys) + 1}] All available languages")
    print()

    ready = [l for l in lang_keys if available[l] is not None]

    while True:
        choice = input("  Select language(s) (comma-separated): ").strip()
        if not choice:
            continue
        if choice == str(len(lang_keys) + 1):
            if not ready:
                print("  No languages are ready yet. Run prepare_experiment1.py first.")
                continue
            print(f"  Selected: {', '.join(l.capitalize() for l in ready)}")
            return ready
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(lang_keys):
                    lang = lang_keys[idx - 1]
                    if available[lang] is None:
                        print(f"  '{lang}' is not ready. Run prepare_experiment1.py --lang {lang}")
                        selected = []
                        break
                    selected.append(lang)
                else:
                    print(f"  Invalid number: {idx}")
                    selected = []
                    break
            if selected:
                print(f"  Selected: {', '.join(l.capitalize() for l in selected)}")
                return selected
        except ValueError:
            print("  Please enter numbers separated by commas.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment1(args) -> None:
    available = discover_experiment1_languages()
    if not any(available.values()):
        print("\n  No prepared data found in data/phase2/")
        print("  Run: python scripts/phase2/prepare_experiment1.py")
        return

    selected_models    = select_models()
    selected_languages = select_languages(available)

    # Resolve prompt: CLI override beats language default
    if args.prompt:
        prompt_key = args.prompt
        prompt_map = {lang: PROMPTS[prompt_key] for lang in selected_languages}
        prompt_label = args.prompt.upper()
    else:
        prompt_map  = {lang: PROMPTS[LANGUAGE_DEFAULT_PROMPTS[lang]] for lang in selected_languages}
        prompt_label = "language-specific V3 (auto)"

    print(f"\n{'='*58}")
    print(f"  Experiment 1 — Evaluation Plan:")
    print(f"  Models:     {', '.join(MODELS[k]['name'] for k in selected_models)}")
    print(f"  Languages:  {', '.join(l.capitalize() for l in selected_languages)}")
    print(f"  Prompt:     {prompt_label}")
    print(f"  Results:    {EXP1_RESULTS_DIR}")
    print(f"{'='*58}")

    confirm = input("\n  Proceed? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        return

    all_output_files = []
    all_metrics      = {}
    print_lock       = threading.Lock()

    def _run_one(model_key: str, lang: str) -> None:
        info         = MODELS[model_key]
        model_delay  = max(args.delay, info.get("delay", 0))
        model_workers = min(args.workers, info.get("max_workers", args.workers))
        template     = prompt_map[lang]

        try:
            api_key = get_api_key(model_key)
        except EnvironmentError as e:
            with print_lock:
                print(f"  [{info['name']}] {e} — skipping")
            return

        provider = info["class"](api_key=api_key, model_name=info["default_model"])

        try:
            samples = load_samples(lang)
        except FileNotFoundError as e:
            with print_lock:
                print(f"  [{info['name']}][{lang}] {e}")
            return

        with print_lock:
            note = f" [rate-limited: {model_workers}w, {model_delay}s]" if model_delay != args.delay else ""
            print(f"\n  Started: {info['name']} x {lang.capitalize()} "
                  f"({len(samples)} samples){note}")

        results = run_evaluation(
            provider, samples, lang, template,
            delay=model_delay, fresh=args.fresh, workers=model_workers,
            results_dir=EXP1_RESULTS_DIR,
        )
        metrics = EvaluationMetrics.compute(results)

        report = EvaluationMetrics.format_report(metrics, f"{info['name']} — {lang.capitalize()}")
        with print_lock:
            print(report)

        path = save_results(results, metrics, model_key, lang,
                            out_dir=EXP1_RESULTS_DIR, experiment=1)
        with print_lock:
            all_output_files.append(str(path))
            all_metrics[(info["name"], lang)] = metrics
            print(f"  Saved: {path}")

    _run_jobs(selected_models, selected_languages, _run_one,
              all_output_files, all_metrics, EXP1_RESULTS_DIR, experiment=1)


def discover_exp1_results() -> Dict[tuple, Path]:
    """
    Scan results/phase2/experiment1/ for completed result files.

    Filenames follow the pattern: {model_key}_{language}_{timestamp}.json
    Returns {(model_key, language): path_to_latest_file}.
    """
    if not EXP1_RESULTS_DIR.exists():
        return {}
    found: Dict[tuple, list] = {}
    for p in EXP1_RESULTS_DIR.glob("*.json"):
        if p.name.startswith("comparison_"):
            continue
        parts = p.stem.split("_")
        # stem = model_key + "_" + language + "_" + YYYYMMDD + "_" + HHMMSS
        # model_key and language are single words; timestamp is two parts
        if len(parts) < 4:
            continue
        # timestamp = last two parts, language = second-to-last before timestamp
        timestamp_parts = parts[-2:]
        if not (timestamp_parts[0].isdigit() and timestamp_parts[1].isdigit()):
            continue
        lang      = parts[-3]
        model_key = "_".join(parts[:-3])
        key = (model_key, lang)
        found.setdefault(key, []).append(p)
    # Keep only the latest file per (model, language)
    return {key: sorted(paths)[-1] for key, paths in found.items()}


def select_exp1_results(available: Dict[tuple, Path]) -> List[tuple]:
    """
    Interactive menu: let the user pick which (model, language) Exp 1 results
    to run attribution on.

    Returns list of (model_key, language) tuples.
    """
    if not available:
        return []
    items = sorted(available.keys())
    print("\n  Available Experiment 1 results:")
    print("  " + "-" * 52)
    for i, (model_key, lang) in enumerate(items, 1):
        model_name = MODELS[model_key]["name"] if model_key in MODELS else model_key
        print(f"  [{i}] {model_name} x {lang.capitalize()}")
    print(f"  [{len(items) + 1}] All listed above")
    print()

    while True:
        choice = input("  Select result(s) (comma-separated): ").strip()
        if not choice:
            continue
        if choice == str(len(items) + 1):
            return items
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(items):
                    selected.append(items[idx - 1])
                else:
                    print(f"  Invalid number: {idx}")
                    selected = []
                    break
            if selected:
                for mk, lang in selected:
                    name = MODELS[mk]["name"] if mk in MODELS else mk
                    print(f"  Selected: {name} x {lang.capitalize()}")
                return selected
        except ValueError:
            print("  Please enter numbers separated by commas.")


def _call_attribution_one(provider, entry: Dict, idx: int, total: int,
                          model_name: str, language: str,
                          prompt_template: str) -> Dict:
    """
    Call the attribution prompt for one already-classified entry.

    Substitutes {prediction} and {post_text} into the template, calls the
    provider, parses the 2-line response (keywords + translations), and
    returns the original entry augmented with attribution fields.
    """
    logger.info(f"[{model_name}][{language}] Attribution {idx}/{total}...")

    filled_template = prompt_template.replace("{prediction}", entry["prediction"])
    result = provider.classify(post=entry["post_full"], prompt_template=filled_template)

    parsed = _parse_exp2_response(result["raw_response"])
    return {
        **entry,
        "keywords":     parsed["keywords"],
        "translations": parsed["translations"],
        "raw_response_exp2": result["raw_response"],
        "error_exp2":        result["error"],
    }


def run_experiment2(args) -> None:
    """
    Experiment 2 — Attribution: explain the model's own Experiment 1 labels.

    For each (model, language) pair already run in Experiment 1, load that
    result file, then ask the same model to identify the word(s) that drove
    each of its predictions.  The model receives the post text and its own
    label as input; it outputs exactly two lines:
        Line 1: key word(s) from the post in the original language, comma-separated
        Line 2: one-word English translation of each keyword, same order

    Each output entry is the original Experiment 1 entry augmented with
    'keywords' and 'translations' fields.
    Results saved to: results/phase2/experiment2/
    """
    exp1_available = discover_exp1_results()
    if not exp1_available:
        print("\n  No Experiment 1 results found.")
        print(f"  Expected location: {EXP1_RESULTS_DIR}")
        print("  Run Experiment 1 first, then come back.")
        return

    selected_pairs = select_exp1_results(exp1_available)
    if not selected_pairs:
        return

    # Resolve attribution prompt per language
    if args.prompt:
        prompt_map   = {lang: PROMPTS[args.prompt]
                        for _, lang in selected_pairs}
        prompt_label = args.prompt.upper()
    else:
        prompt_map   = {lang: PROMPTS[LANGUAGE_DEFAULT_PROMPTS_EXP2.get(lang, "v3_exp2")]
                        for _, lang in selected_pairs}
        prompt_label = "language-specific attribution V3 (auto)"

    print(f"\n{'='*58}")
    print(f"  Experiment 2 — Attribution Plan:")
    for mk, lang in selected_pairs:
        name = MODELS[mk]["name"] if mk in MODELS else mk
        print(f"    {name} x {lang.capitalize()}")
    print(f"  Prompt:   {prompt_label}")
    print(f"  Results:  {EXP2_RESULTS_DIR}")
    print(f"{'='*58}")

    confirm = input("\n  Proceed? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        return

    all_output_files = []
    print_lock       = threading.Lock()

    def _run_one_attr(model_key: str, lang: str) -> None:
        info          = MODELS.get(model_key, {"name": model_key, "class": None,
                                               "default_model": None})
        model_delay   = max(args.delay, info.get("delay", 0))
        model_workers = min(args.workers, info.get("max_workers", args.workers))
        template      = prompt_map.get(lang, PROMPTS[LANGUAGE_DEFAULT_PROMPTS_EXP2.get(lang, "v3_exp2")])

        # Load Exp 1 results
        exp1_path = exp1_available[(model_key, lang)]
        with open(exp1_path, encoding="utf-8") as f:
            exp1_data = json.load(f)
        all_entries = exp1_data["results"]

        # Skip error entries — no valid prediction to attribute
        entries = [e for e in all_entries if e.get("prediction") not in ("error", "unclear", None)]
        skipped = len(all_entries) - len(entries)
        if skipped:
            logger.info(f"[{model_key}][{lang}] Skipping {skipped} error/unclear entries")

        try:
            api_key = get_api_key(model_key)
        except EnvironmentError as e:
            with print_lock:
                print(f"  [{info['name']}] {e} — skipping")
            return

        provider = info["class"](api_key=api_key, model_name=info["default_model"])

        with print_lock:
            print(f"\n  Started attribution: {info['name']} x {lang.capitalize()} "
                  f"({len(entries)} entries)")

        results = run_evaluation(
            provider, entries, lang, template,
            delay=model_delay, fresh=args.fresh, workers=model_workers,
            classify_fn=_call_attribution_one,
            results_dir=EXP2_RESULTS_DIR,
        )

        # Restore skipped entries (no keywords/translations) so file is complete
        skipped_entries = [
            {**e, "keywords": [], "translations": [], "raw_response_exp2": "", "error_exp2": "skipped"}
            for e in all_entries if e.get("prediction") in ("error", "unclear", None)
        ]
        full_results = results + skipped_entries
        full_results.sort(key=lambda x: x.get("index", 0))

        # Exp 2 has no new ground-truth predictions — reuse Exp 1 metrics
        metrics = EvaluationMetrics.compute(full_results)
        report  = EvaluationMetrics.format_report(metrics, f"{info['name']} — {lang.capitalize()} (Exp 1 labels)")
        with print_lock:
            print(report)

        path = save_results(full_results, metrics, model_key, lang,
                            out_dir=EXP2_RESULTS_DIR, experiment=2)
        with print_lock:
            all_output_files.append(str(path))
            print(f"  Saved: {path}")

    jobs = selected_pairs
    if len(jobs) == 1:
        _run_one_attr(jobs[0][0], jobs[0][1])
    else:
        print(f"\n  Running {len(jobs)} attribution jobs")
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for f in [executor.submit(_run_one_attr, mk, lang) for mk, lang in jobs]:
                f.result()

    print(f"\n{'='*58}")
    print("  Experiment 2 complete!")
    print(f"{'='*58}")
    for fp in all_output_files:
        print(f"    -> {fp}")
    print()


# ── Shared job dispatcher ─────────────────────────────────────────────────────

def _run_jobs(selected_models, selected_languages, run_one_fn,
              all_output_files, all_metrics, out_dir: Path, experiment: int) -> None:
    """Dispatch model×language jobs, then print comparison tables."""
    jobs = [(mk, lang) for mk in selected_models for lang in selected_languages]

    if len(jobs) == 1:
        run_one_fn(jobs[0][0], jobs[0][1])
    else:
        print(f"\n  Running {len(jobs)} evaluations in parallel")
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for f in [executor.submit(run_one_fn, mk, lang) for mk, lang in jobs]:
                f.result()

    if len(selected_languages) > 1:
        for model_key in selected_models:
            model_name   = MODELS[model_key]["name"]
            lang_metrics = {lang: m for (mn, lang), m in all_metrics.items() if mn == model_name}
            if len(lang_metrics) > 1:
                print(f"\n{'='*70}")
                print(f"  CROSS-LINGUAL: {model_name}")
                print(f"{'='*70}")
                print(f"  {'Language':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
                print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                for lang, m in lang_metrics.items():
                    print(f"  {lang.capitalize():<12} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                          f"{m['recall']:>8.4f} {m['f1_score']:>8.4f}")

    if len(selected_models) > 1:
        for lang in selected_languages:
            model_metrics = {mn: m for (mn, l), m in all_metrics.items() if l == lang}
            if len(model_metrics) > 1:
                print(f"\n  Model Comparison — {lang.upper()}:")
                print(EvaluationMetrics.comparison_table(model_metrics))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = out_dir / f"comparison_{timestamp}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison_data: Dict = {}
        for (mn, lang), m in all_metrics.items():
            comparison_data.setdefault(mn, {})[lang] = m
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": experiment, "models": comparison_data}, f, indent=2)
        all_output_files.append(str(comp_path))
        print(f"  Comparison saved: {comp_path}")

    print(f"\n{'='*58}")
    print("  Evaluation complete!")
    print(f"{'='*58}")
    for fp in all_output_files:
        print(f"    -> {fp}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 evaluation runner — Experiments 1/2/3"
    )
    parser.add_argument("--fresh",   action="store_true", help="Ignore partial results")
    parser.add_argument("--delay",   type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--workers", type=int,   default=1,   help="Parallel requests per model")
    parser.add_argument("--prompt",  type=str,   default=None, choices=list(PROMPTS.keys()),
                        help="Override prompt version (default: language-specific V3)")
    args = parser.parse_args()

    print_header()

    while True:
        exp = select_experiment()

        if exp == 1:
            run_experiment1(args)
            break
        elif exp == 2:
            run_experiment2(args)
            break
        elif exp == 3:
            print("\n  Experiment 3 is not yet implemented.")
            print("  Returning to menu...\n")


if __name__ == "__main__":
    main()
