"""Experiment 4 — fresh classification + justification.

Default mode: 15-sample hardest-wrong CSVs (error analysis).
Full mode (--full): 5 000-sample phase2 JSONs per language.

Examples:
    # Smoke test (15-row mode)
    python scripts/run_exp4.py --model claude --language urdu --limit 1

    # All online models, all languages (15-row few-shot)
    python scripts/run_exp4.py --models claude,openai,gemini --languages all

    # Full 5k dataset, one model at a time (resumes automatically)
    python scripts/run_exp4.py --model claude --language arabic --full
    python scripts/run_exp4.py --model openai --language arabic --full

    # Full 5k zero-shot
    python scripts/run_exp4.py --model claude --language arabic --full --zeroshot

    # Debug: print one row's full response, write nothing
    python scripts/run_exp4.py --model claude --language arabic --debug 42
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import get_api_key  # noqa: E402
from evaluation.prompts import LANGUAGE_DEFAULT_PROMPTS_EXP4, LANGUAGE_DEFAULT_PROMPTS_EXP4_ZEROSHOT, PROMPTS  # noqa: E402
from models import (  # noqa: E402
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    LMStudioProvider,
    OpenAIProvider,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("exp4")

# --- directories ---
CSV_DIR         = REPO / "results" / "all_models_wrong"          # 15-row error-analysis CSVs
FULL_DATA_DIR   = REPO / "data" / "phase2"                       # 5k-sample JSONs
OUT_DIR         = REPO / "results" / "phase2" / "experiment4"
OUT_DIR_ZS      = REPO / "results" / "phase2" / "experiment4_zeroshot"
FULL_OUT_DIR    = REPO / "results" / "phase2" / "experiment4_full"
FULL_OUT_DIR_ZS = REPO / "results" / "phase2" / "experiment4_full_zeroshot"

LANGUAGES  = ["arabic", "chinese", "urdu"]
MAX_TOKENS = 500
SAVE_EVERY = 50   # checkpoint interval for --full runs

MODELS = {
    "gemini": {"class": GeminiProvider,   "default_model": "gemini-3.1-flash-lite",               "online": True,  "workers": 5, "delay": 0.0, "max_tokens": 500},
    "openai": {"class": OpenAIProvider,   "default_model": "gpt-4o-mini",                          "online": True,  "workers": 5, "delay": 0.0, "max_tokens": 500},
    "claude": {"class": ClaudeProvider,   "default_model": "claude-haiku-4-5",                     "online": True,  "workers": 1, "delay": 1.5, "max_tokens": 500},
    "llama":  {"class": LMStudioProvider, "default_model": "BeaverAI/Llama-3.3-8B-Instruct-GGUF", "online": False, "workers": 1, "delay": 0.0, "max_tokens": 500},
    "gemma":  {"class": LMStudioProvider, "default_model": "google/gemma-4-e2b",                   "online": False, "workers": 1, "delay": 0.0, "max_tokens": 4000},
    "qwen":   {"class": LMStudioProvider, "default_model": "qwen/qwen3.5-9b",                      "online": False, "workers": 1, "delay": 0.0, "max_tokens": 500, "no_think": True},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(language: str) -> list[dict]:
    """15-row error-analysis CSV."""
    path = CSV_DIR / f"{language}_all_wrong.csv"
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_rows_full(language: str) -> list[dict]:
    """5 000-sample phase2 JSON — normalised to the same dict shape as load_rows."""
    path = FULL_DATA_DIR / f"{language}_5000samples_seed42.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        {
            "index":        s["index"],
            "conversation": s["post"],
            "translation":  s.get("translation", ""),
            "actual_value": s["ground_truth"],
        }
        for s in data["samples"]
    ]


def _partial_path(model_key: str, language: str, out_dir: Path) -> Path:
    return out_dir / model_key / f"{model_key}_{language}.partial.json"


def _save_partial(model_key: str, language: str, out_dir: Path, results: list) -> None:
    path = _partial_path(model_key, language, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=2)


def load_done(model_key: str, language: str, out_dir: Path):
    """Return (done_indices, existing_results, existing_path|None).

    Checks for an in-progress .partial.json checkpoint first (interrupted run),
    then falls back to the newest completed timestamped file. done_indices only
    contains clean results; errors/uncleareds are retried on resume.
    """
    partial = _partial_path(model_key, language, out_dir)
    if partial.exists():
        try:
            with open(partial, encoding="utf-8") as fh:
                data = json.load(fh)
            existing = [r for r in data.get("results", []) if r]
            done = {
                int(r["index"]) for r in existing
                if r.get("exp4_classification") not in ("error", "unclear", None)
            }
            log.info(f"[{model_key}/{language}] Resuming from partial checkpoint ({len(existing)} rows so far)")
            return done, existing, partial
        except Exception as e:
            log.warning(f"[{model_key}/{language}] Could not read partial checkpoint, ignoring: {e}")

    model_dir = out_dir / model_key
    # Use [0-9]* so we never accidentally match .partial.json
    matches = sorted(model_dir.glob(f"{model_key}_{language}_[0-9]*.json")) if model_dir.is_dir() else []
    if not matches:
        return set(), [], None
    with open(matches[-1], encoding="utf-8") as fh:
        data = json.load(fh)
    existing = [r for r in data.get("results", []) if r]
    done = {
        int(r["index"]) for r in existing
        if r.get("exp4_classification") not in ("error", "unclear", None)
    }
    return done, existing, matches[-1]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

CLASSIFICATION_LINE = re.compile(r"^\s*classification\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
JUSTIFICATION_LINE  = re.compile(r"^\s*justification\s*:\s*(.+)$",      re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_exp4(raw: str) -> dict:
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
        body = j.group(1).strip()
        for stop in ("\nPost:", "\nResponse:", "\nClassification:"):
            if stop in body:
                body = body.split(stop)[0]
        justification = " ".join(body.split())[:600]

    return {
        "classification": label,
        "justification":  justification,
        "parse_note":     "" if (m and j) else f"missing fields: clf={bool(m)} just={bool(j)}",
    }


# ---------------------------------------------------------------------------
# Provider init
# ---------------------------------------------------------------------------

def init_provider(model_key: str):
    spec = MODELS[model_key]
    api_key = "lm-studio" if not spec["online"] else get_api_key(model_key)
    return spec["class"](
        api_key=api_key,
        model_name=spec["default_model"],
        max_tokens=spec.get("max_tokens", MAX_TOKENS),
        **({} if spec["online"] else {"no_think": spec.get("no_think", False)}),
    )


# ---------------------------------------------------------------------------
# Per-row classification
# ---------------------------------------------------------------------------

def _classify_one(provider, row: dict, prompt: str) -> dict:
    post   = row["conversation"]
    result = provider.classify(post=post, prompt_template=prompt)
    parsed = parse_exp4(result["raw_response"])
    return {
        "index":               int(row["index"]),
        "post_full":           post,
        "translation":         row.get("translation", ""),
        "ground_truth":        row["actual_value"],
        "prior_prediction":    row.get(f"{provider.__class__.__name__.lower().replace('provider','')}_prediction", ""),
        "exp4_classification": parsed["classification"],
        "exp4_justification":  parsed["justification"],
        "raw_response":        result["raw_response"],
        "error":               result.get("error"),
        "parse_note":          parsed["parse_note"],
    }


def _log_result(prefix: str, done: int, total: int, idx: int, r: dict):
    clf  = r.get("exp4_classification", "?")
    just = (r.get("exp4_justification") or "")[:120]
    note = r.get("parse_note") or ""
    log.info(
        f"{prefix} {done}/{total} idx={idx} | {clf}"
        + (f" | {just}{'…' if len(just) == 120 else ''}" if just else "")
        + (f"  ⚠ {note}" if note else "")
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_for(
    model_key: str,
    language: str,
    zeroshot: bool = False,
    full: bool = False,
    fresh: bool = False,
    limit: int | None = None,
) -> Path:
    """Run one model on one language; write a timestamped JSON; return its path.

    Resumable: a .partial.json checkpoint is saved every SAVE_EVERY rows so
    crashes don't lose progress. On completion the partial is deleted and a
    clean timestamped file is written.

    Args:
        fresh: ignore any existing results and start from scratch.
        limit: only process the first N rows (smoke-test).
    """
    spec = MODELS[model_key]
    prefix = f"[{model_key}/{language}]"

    if full:
        rows    = load_rows_full(language)
        out_dir = FULL_OUT_DIR_ZS if zeroshot else FULL_OUT_DIR
    else:
        rows    = load_rows(language)
        out_dir = OUT_DIR_ZS if zeroshot else OUT_DIR

    if limit:
        rows = rows[:limit]

    if not rows:
        raise SystemExit(f"No rows for {language}")

    prompt_map = LANGUAGE_DEFAULT_PROMPTS_EXP4_ZEROSHOT if zeroshot else LANGUAGE_DEFAULT_PROMPTS_EXP4

    done_indices, existing_results, existing_path = load_done(model_key, language, out_dir)
    if fresh:
        if done_indices:
            log.info(f"{prefix} --fresh: discarding {len(done_indices)} existing results")
        done_indices, existing_results = set(), []
        partial = _partial_path(model_key, language, out_dir)
        if partial.exists():
            partial.unlink()

    total = len(rows)
    if done_indices:
        log.info(f"{prefix} resuming — {len(done_indices)}/{total} already done, retrying errors/uncleareds")
        rows = [r for r in rows if int(r["index"]) not in done_indices]

    if not rows:
        log.info(f"{prefix} all {total} rows complete — nothing to do")
        return existing_path

    log.info(
        f"{prefix} processing {len(rows)} rows  workers={spec['workers']} "
        f"delay={spec['delay']} full={full} zeroshot={zeroshot}"
    )
    provider   = init_provider(model_key)
    prompt_key = prompt_map[language]
    prompt     = PROMPTS[prompt_key]

    all_accumulated: list[dict] = list(existing_results)
    done_count = len(done_indices)
    t0 = time.time()

    if spec["workers"] == 1:
        for row in rows:
            try:
                r = _classify_one(provider, row, prompt)
            except Exception as e:
                r = {
                    "index": int(row["index"]), "error": f"hard fail: {e}",
                    "exp4_classification": "error", "exp4_justification": "", "raw_response": "",
                }
                log.warning(f"{prefix} idx={row['index']} hard fail: {e}")
            all_accumulated.append(r)
            if spec["delay"]:
                time.sleep(spec["delay"])
            done_count += 1
            _log_result(prefix, done_count, total, int(row["index"]), r)
            if len(all_accumulated) % SAVE_EVERY == 0:
                _save_partial(model_key, language, out_dir, all_accumulated)
                log.info(f"{prefix} checkpoint: {len(all_accumulated)} rows saved")
    else:
        with ThreadPoolExecutor(max_workers=spec["workers"]) as ex:
            futs = {ex.submit(_classify_one, provider, row, prompt): i for i, row in enumerate(rows)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    r = f.result()
                except Exception as e:
                    r = {
                        "index": int(rows[i]["index"]), "error": f"hard fail: {e}",
                        "exp4_classification": "error", "exp4_justification": "", "raw_response": "",
                    }
                    log.warning(f"{prefix} idx={rows[i]['index']} hard fail: {e}")
                all_accumulated.append(r)
                done_count += 1
                _log_result(prefix, done_count, total, int(rows[i]["index"]), r)
                if len(all_accumulated) % SAVE_EVERY == 0:
                    _save_partial(model_key, language, out_dir, all_accumulated)
                    log.info(f"{prefix} checkpoint: {len(all_accumulated)} rows saved")

    all_accumulated.sort(key=lambda r: r.get("index", 0))

    elapsed   = time.time() - t0
    n_ok      = sum(1 for r in all_accumulated if r.get("exp4_classification") not in ("error", "unclear"))
    n_unclear = sum(1 for r in all_accumulated if r.get("exp4_classification") == "unclear")
    n_err     = sum(1 for r in all_accumulated if r.get("exp4_classification") == "error")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_subdir = out_dir / model_key
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{model_key}_{language}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": {
                    "model": model_key, "language": language, "experiment": 4,
                    "prompt": prompt_key, "model_id": spec["default_model"],
                    "n": len(all_accumulated), "ok": n_ok, "unclear": n_unclear, "error": n_err,
                    "elapsed_sec": round(elapsed, 1), "timestamp": ts,
                    "full_dataset": full,
                },
                "results": all_accumulated,
            },
            fh, ensure_ascii=False, indent=2,
        )

    partial = _partial_path(model_key, language, out_dir)
    if partial.exists():
        partial.unlink()

    log.info(
        f"{prefix} DONE in {elapsed:.1f}s  total={len(all_accumulated)} ok={n_ok} unclear={n_unclear} err={n_err}"
        f"  -> {out_path.relative_to(REPO)}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

def debug_one(model_key: str, language: str, index: int, zeroshot: bool = False, full: bool = False):
    """Run one specific row by index, print everything, write no files."""
    rows = load_rows_full(language) if full else load_rows(language)
    row  = next((r for r in rows if int(r["index"]) == index), None)
    if row is None:
        raise SystemExit(f"Index {index} not found in {language} {'full' if full else 'CSV'}.")

    prompt_map = LANGUAGE_DEFAULT_PROMPTS_EXP4_ZEROSHOT if zeroshot else LANGUAGE_DEFAULT_PROMPTS_EXP4
    prompt_key = prompt_map[language]
    prompt     = PROMPTS[prompt_key]
    provider   = init_provider(model_key)

    print(f"\n{'='*60}")
    print(f"MODEL: {model_key}  LANGUAGE: {language}  INDEX: {index}  FULL: {full}")
    print(f"PROMPT KEY: {prompt_key}")
    print(f"{'='*60}")
    print(f"POST:\n{row['conversation']}\n")

    result = provider.classify(post=row["conversation"], prompt_template=prompt)
    parsed = parse_exp4(result["raw_response"])

    print(f"RAW RESPONSE:\n{result['raw_response']}\n")
    print(f"{'='*60}")
    print(f"PARSED CLASSIFICATION : {parsed['classification']}")
    print(f"PARSED JUSTIFICATION  : {parsed['justification']}")
    print(f"PARSE NOTE            : {parsed['parse_note']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     help="Single model key (e.g. claude). Mutually exclusive with --models.")
    p.add_argument("--models",    help="Comma-separated model keys, or 'online'/'local'/'all'. Default: all.")
    p.add_argument("--language",  help="Single language (urdu/arabic/chinese). Mutually exclusive with --languages.")
    p.add_argument("--languages", help="Comma-separated languages, or 'all'. Default: all.")
    p.add_argument("--limit",     type=int, help="Smoke-test: only process the first N rows per (model, lang).")
    p.add_argument("--full",      action="store_true", help="Use full 5k-sample dataset from data/phase2/ instead of 15-row CSVs.")
    p.add_argument("--zeroshot",  action="store_true", help="Use universal zero-shot prompt. Saves to experiment4[_full]_zeroshot/.")
    p.add_argument("--fresh",     action="store_true", help="Ignore existing results and partial checkpoints; start from scratch.")
    p.add_argument("--debug",     type=int, metavar="INDEX", help="Debug: run one row by index, print full response, write nothing.")
    return p.parse_args()


def resolve_models(args) -> list[str]:
    raw = (args.model or args.models or "all").lower()
    if raw == "all":    return list(MODELS)
    if raw == "online": return [m for m, s in MODELS.items() if s["online"]]
    if raw == "local":  return [m for m, s in MODELS.items() if not s["online"]]
    return [m.strip() for m in raw.split(",") if m.strip()]


def resolve_languages(args) -> list[str]:
    raw = (args.language or args.languages or "all").lower()
    if raw == "all": return LANGUAGES
    return [l.strip() for l in raw.split(",") if l.strip()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    models = resolve_models(args)
    langs  = resolve_languages(args)

    for m in models:
        if m not in MODELS:
            raise SystemExit(f"Unknown model: {m}. Valid: {sorted(MODELS)}")
    for l in langs:
        if l not in LANGUAGES:
            raise SystemExit(f"Unknown language: {l}. Valid: {LANGUAGES}")
        if args.full:
            src = FULL_DATA_DIR / f"{l}_5000samples_seed42.json"
        else:
            src = CSV_DIR / f"{l}_all_wrong.csv"
        if not src.exists():
            raise SystemExit(f"Missing input: {src}")

    if args.debug is not None:
        if len(models) != 1 or len(langs) != 1:
            raise SystemExit("--debug requires exactly one --model and one --language.")
        debug_one(models[0], langs[0], args.debug, zeroshot=args.zeroshot, full=args.full)
        return

    log.info(f"Models: {models}  Languages: {langs}  full={args.full}  zeroshot={args.zeroshot}")

    for m in models:
        for l in langs:
            try:
                run_for(m, l, zeroshot=args.zeroshot, full=args.full,
                        fresh=args.fresh, limit=args.limit)
            except Exception as e:
                log.error(f"[{m}/{l}] FAILED: {e}")


if __name__ == "__main__":
    main()
