"""Experiment 4 — fresh classification + justification on the 15-sample CSVs.

Reads results/all_models_wrong/{language}_all_wrong.csv (15 rows each), prompts
each requested model with v3_exp4, parses Classification + Justification, and
writes results to results/phase2/experiment4/{model}/{model}_{language}_<ts>.json.

Examples:
    # Run one model on one language (smoke test)
    python scripts/phase2/run_exp4.py --model claude --language urdu --limit 1

    # Run all online models on all 3 languages
    python scripts/phase2/run_exp4.py --models claude,openai,gemini --languages all

    # Run a single local model (LM Studio must be serving the right model)
    python scripts/phase2/run_exp4.py --model gemma --language arabic

    # Run everything
    python scripts/phase2/run_exp4.py
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from config import get_api_key  # noqa: E402
from evaluation.prompts import PROMPTS  # noqa: E402
from models import (  # noqa: E402
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    LMStudioProvider,
    OpenAIProvider,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("exp4")

CSV_DIR = REPO / "results" / "all_models_wrong"
OUT_DIR = REPO / "results" / "phase2" / "experiment4"
LANGUAGES = ["arabic", "chinese", "urdu"]
PROMPT_KEY = "v3_exp4"
MAX_TOKENS = 500  # ~enough for 4 sentences of justification + the classification line

# Model registry mirrors scripts/phase2/runner.py but adds an explicit api_key
# (online) / lm_studio (local) flag plus the max_workers we'll run with here.
MODELS = {
    "gemini":         {"class": GeminiProvider,   "default_model": "gemini-2.0-flash",                    "online": True,  "workers": 5,  "delay": 0.0},
    "openai":         {"class": OpenAIProvider,   "default_model": "gpt-4o-mini",                         "online": True,  "workers": 5,  "delay": 0.0},
    "claude":         {"class": ClaudeProvider,   "default_model": "claude-haiku-4-5",                    "online": True,  "workers": 1,  "delay": 1.5},
    "deepseek":       {"class": DeepSeekProvider, "default_model": "deepseek-chat",                       "online": True,  "workers": 3,  "delay": 0.5},
    "llama":          {"class": LMStudioProvider, "default_model": "meta-llama-3.1-8b-instruct",          "online": False, "workers": 1,  "delay": 0.0},
    "gemma":          {"class": LMStudioProvider, "default_model": "google/gemma-3-9b-it",                "online": False, "workers": 1,  "delay": 0.0},
    "deepseek-local": {"class": LMStudioProvider, "default_model": "deepseek/deepseek-r1-0528-qwen3-8b",  "online": False, "workers": 1,  "delay": 0.0},
}


def load_rows(language: str) -> list[dict]:
    path = CSV_DIR / f"{language}_all_wrong.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


CLASSIFICATION_LINE = re.compile(r"^\s*classification\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
JUSTIFICATION_LINE  = re.compile(r"^\s*justification\s*:\s*(.+)$",      re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_exp4(raw: str) -> dict:
    """Pull Classification + Justification out of the model's response."""
    if not raw:
        return {"classification": "error", "justification": "", "parse_note": "empty response"}

    m = CLASSIFICATION_LINE.search(raw)
    label = "unclear"
    if m:
        v = m.group(1).strip().lower()
        if "not" in v and "depress" in v:
            label = "not depressed"
        elif "depress" in v:
            label = "depressed"

    j = JUSTIFICATION_LINE.search(raw)
    justification = ""
    if j:
        # Take everything after "Justification:" until end. Strip excess whitespace and
        # cap at 600 chars in case the model rambles (the prompt asks for 300).
        body = j.group(1).strip()
        # Stop at "Post:" or "Response:" if it bleeds into another section
        for stop in ("\nPost:", "\nResponse:", "\nClassification:"):
            if stop in body:
                body = body.split(stop)[0]
        justification = " ".join(body.split())[:600]

    return {
        "classification": label,
        "justification": justification,
        "parse_note": "" if (m and j) else f"missing fields: clf={bool(m)} just={bool(j)}",
    }


def init_provider(model_key: str):
    spec = MODELS[model_key]
    api_key = "lm-studio" if not spec["online"] else get_api_key(model_key)
    return spec["class"](
        api_key=api_key,
        model_name=spec["default_model"],
        max_tokens=MAX_TOKENS,
    )


def _classify_one(provider, row: dict, prompt: str) -> dict:
    """Call provider on one row, returning enriched result."""
    post = row["conversation"]
    result = provider.classify(post=post, prompt_template=prompt)
    parsed = parse_exp4(result["raw_response"])
    return {
        "index": int(row["index"]),
        "post_full": post,
        "translation": row.get("translation", ""),
        "ground_truth": row["actual_value"],
        "prior_prediction": row.get(f"{provider.__class__.__name__.lower().replace('provider','')}_prediction", ""),
        "exp4_classification": parsed["classification"],
        "exp4_justification": parsed["justification"],
        "raw_response": result["raw_response"],
        "error": result.get("error"),
        "parse_note": parsed["parse_note"],
    }


def run_for(model_key: str, language: str) -> Path:
    """Run one model on one language; write a single JSON; return its path."""
    spec = MODELS[model_key]
    rows = load_rows(language)
    if not rows:
        raise SystemExit(f"No rows for {language} in {CSV_DIR}")

    log.info(f"[{model_key}/{language}] {len(rows)} rows, workers={spec['workers']}, delay={spec['delay']}")
    provider = init_provider(model_key)
    prompt = PROMPTS[PROMPT_KEY]

    results: list[dict] = [None] * len(rows)
    t0 = time.time()
    if spec["workers"] == 1:
        for i, row in enumerate(rows):
            try:
                results[i] = _classify_one(provider, row, prompt)
            except Exception as e:
                results[i] = {"index": int(row["index"]), "error": f"hard fail: {e}", "exp4_classification": "error", "exp4_justification": "", "raw_response": ""}
                log.warning(f"[{model_key}/{language}] row {i+1} failed: {e}")
            if spec["delay"]:
                time.sleep(spec["delay"])
            log.info(f"[{model_key}/{language}] {i+1}/{len(rows)} idx={row['index']} -> {results[i].get('exp4_classification')}")
    else:
        with ThreadPoolExecutor(max_workers=spec["workers"]) as ex:
            futs = {ex.submit(_classify_one, provider, row, prompt): i for i, row in enumerate(rows)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    results[i] = f.result()
                except Exception as e:
                    results[i] = {"index": int(rows[i]["index"]), "error": f"hard fail: {e}", "exp4_classification": "error", "exp4_justification": "", "raw_response": ""}
                    log.warning(f"[{model_key}/{language}] row {i+1} failed: {e}")
                log.info(f"[{model_key}/{language}] {sum(1 for r in results if r)}/{len(rows)} done")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r and r.get("exp4_classification") not in ("error", "unclear"))
    n_unclear = sum(1 for r in results if r and r.get("exp4_classification") == "unclear")
    n_err = sum(1 for r in results if r and r.get("exp4_classification") == "error")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_subdir = OUT_DIR / model_key
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{model_key}_{language}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "metadata": {
                "model": model_key, "language": language, "experiment": 4,
                "prompt": PROMPT_KEY, "model_id": spec["default_model"],
                "n": len(rows), "ok": n_ok, "unclear": n_unclear, "error": n_err,
                "elapsed_sec": round(elapsed, 1),
                "timestamp": ts,
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=2)
    log.info(f"[{model_key}/{language}] DONE in {elapsed:.1f}s  ok={n_ok} unclear={n_unclear} err={n_err}  -> {out_path.relative_to(REPO)}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     help="Single model key (e.g. claude). Mutually exclusive with --models.")
    p.add_argument("--models",    help="Comma-separated model keys, or 'online'/'local'/'all'. Default: all.")
    p.add_argument("--language",  help="Single language (urdu/arabic/chinese). Mutually exclusive with --languages.")
    p.add_argument("--languages", help="Comma-separated languages, or 'all'. Default: all.")
    p.add_argument("--limit",     type=int, help="Smoke-test: only process the first N rows per (model,lang).")
    return p.parse_args()


def resolve_models(args) -> list[str]:
    if args.model: return [args.model]
    raw = (args.models or "all").lower()
    if raw == "all":    return list(MODELS)
    if raw == "online": return [m for m, s in MODELS.items() if s["online"]]
    if raw == "local":  return [m for m, s in MODELS.items() if not s["online"]]
    return [m.strip() for m in raw.split(",") if m.strip()]


def resolve_languages(args) -> list[str]:
    if args.language: return [args.language]
    raw = (args.languages or "all").lower()
    if raw == "all": return LANGUAGES
    return [l.strip() for l in raw.split(",") if l.strip()]


def main():
    args = parse_args()
    models = resolve_models(args)
    langs  = resolve_languages(args)

    for m in models:
        if m not in MODELS:
            raise SystemExit(f"Unknown model: {m}. Valid: {sorted(MODELS)}")
    for l in langs:
        if l not in LANGUAGES:
            raise SystemExit(f"Unknown language: {l}. Valid: {LANGUAGES}")
        if not (CSV_DIR / f"{l}_all_wrong.csv").exists():
            raise SystemExit(f"Missing input CSV: {CSV_DIR / f'{l}_all_wrong.csv'}")

    # Apply --limit by temporarily monkey-patching load_rows
    if args.limit:
        global load_rows
        _orig = load_rows
        load_rows = lambda lang, _o=_orig, _n=args.limit: _o(lang)[:_n]

    log.info(f"Models: {models}")
    log.info(f"Languages: {langs}")

    for m in models:
        for l in langs:
            try:
                run_for(m, l)
            except Exception as e:
                log.error(f"[{m}/{l}] FAILED: {e}")


if __name__ == "__main__":
    main()
