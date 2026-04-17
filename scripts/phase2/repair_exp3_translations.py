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

Splitting a big repair across multiple machines (shard mode):
    # PC 1
    python scripts/phase2/repair_exp3_translations.py --file <path> --shard 1/3
    # PC 2
    python scripts/phase2/repair_exp3_translations.py --file <path> --shard 2/3
    # PC 3
    python scripts/phase2/repair_exp3_translations.py --file <path> --shard 3/3
    # After all shards land (git pull on one machine):
    python scripts/phase2/repair_exp3_translations.py --merge-shards <path>
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
EXP3_RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment3"

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


# ── candidate discovery ───────────────────────────────────────────────────────

def _audit_file(path: Path) -> dict | None:
    """
    Peek at an exp3 result file and summarise how broken it is.
    Returns None for files that can't be read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    meta = data.get("metadata", {})
    results = data.get("results", [])
    if not results:
        return None

    no_trans = [r for r in results if r.get("agreement") == "no_translation"]
    return {
        "path":      path,
        "model":     meta.get("model", ""),
        "language":  meta.get("language", ""),
        "total":     len(results),
        "no_trans":  len(no_trans),
    }


def find_candidate_files(model_filter: str | None = None) -> list[dict]:
    """
    Scan the exp3 results tree for files with any 'no_translation' rows.
    Optionally filter to a single model key (e.g. 'gemma', 'deepseek-local').
    """
    if not EXP3_RESULTS_DIR.exists():
        return []

    candidates = []
    for p in EXP3_RESULTS_DIR.rglob("*.json"):
        if p.name.endswith(".partial.json"):
            continue
        audit = _audit_file(p)
        if audit is None or audit["no_trans"] == 0:
            continue
        if model_filter and audit["model"] != model_filter:
            continue
        candidates.append(audit)

    # Newest first (by mtime) so we prefer the most recent result per model/lang
    candidates.sort(key=lambda a: a["path"].stat().st_mtime, reverse=True)
    return candidates


# ── shard helpers ─────────────────────────────────────────────────────────────

def _parse_shard(spec: str | None) -> tuple[int, int] | None:
    """Parse a --shard spec like '1/3'. Returns (shard_num, total) or None."""
    if not spec:
        return None
    try:
        a, b = spec.split("/")
        n, m = int(a), int(b)
    except ValueError:
        raise ValueError(f"--shard must be 'N/M' (e.g. '1/3'), got: {spec!r}")
    if not (1 <= n <= m):
        raise ValueError(f"--shard '{spec}' out of range (need 1 <= N <= M)")
    return n, m


def _shard_slice(count: int, shard: tuple[int, int]) -> tuple[int, int]:
    """Given 'shard 1/3' and count=4971, return (start, end) half-open."""
    n, m = shard
    base, rem = divmod(count, m)
    # Shards 1..rem get one extra row each (remainder distribution)
    start = (n - 1) * base + min(n - 1, rem)
    end   = start + base + (1 if n <= rem else 0)
    return start, end


def _shard_suffix(shard: tuple[int, int] | None) -> str:
    """'shard_2of3' or '' if no shard."""
    if shard is None:
        return ""
    return f"shard_{shard[0]}of{shard[1]}"


# ── checkpoint (resume) helpers ───────────────────────────────────────────────

CHECKPOINT_EVERY = 50  # save progress every N repaired entries


def _checkpoint_path(original: Path, shard: tuple[int, int] | None = None) -> Path:
    suffix = ".repair_checkpoint"
    if shard is not None:
        suffix += f"_{_shard_suffix(shard)}"
    suffix += ".json"
    return original.with_suffix(original.suffix + suffix)


def _load_checkpoint(original: Path, shard: tuple[int, int] | None = None) -> Dict[int, dict]:
    """Return {index: repaired_entry} from an existing checkpoint, or {}."""
    cp = _checkpoint_path(original, shard)
    if not cp.exists():
        return {}
    try:
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("repaired", {}).items()}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  Could not read checkpoint {cp.name}: {e} — starting fresh.")
        return {}


def _save_checkpoint(original: Path, repaired_by_idx: Dict[int, dict],
                     shard: tuple[int, int] | None = None) -> None:
    cp = _checkpoint_path(original, shard)
    tmp = cp.with_suffix(cp.suffix + ".tmp")
    payload = {
        "source_file":    original.name,
        "shard":          f"{shard[0]}/{shard[1]}" if shard else None,
        "last_updated":   datetime.now().isoformat(timespec="seconds"),
        "repaired_count": len(repaired_by_idx),
        # keys must be strings for JSON
        "repaired":       {str(k): v for k, v in repaired_by_idx.items()},
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(cp)  # atomic


def _clear_checkpoint(original: Path, shard: tuple[int, int] | None = None) -> None:
    cp = _checkpoint_path(original, shard)
    if cp.exists():
        cp.unlink()


# ── per-file repair ───────────────────────────────────────────────────────────

def repair_file(path: Path, delay_override: float, dry_run: bool,
                shard: tuple[int, int] | None = None) -> bool:
    """
    Repair a single exp3 result file. Returns True if something was done
    (or dry-run reported fixable rows), False if nothing to do / skipped.

    When ``shard`` is given (e.g. (2, 3)), only the shard's slice of the
    fixable rows is processed and the output is a shard file meant to be
    stitched together later with --merge-shards.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    metadata  = data.get("metadata", {})
    results   = data.get("results", [])
    model_key = metadata.get("model", "")
    language  = metadata.get("language", "")

    no_trans_rows = [r for r in results if r.get("agreement") == "no_translation"]

    print(f"\n  File:              {path.name}")
    print(f"  Model:             {model_key}  |  Language: {language}")
    print(f"  Total rows:        {len(results)}")
    print(f"  'no_translation':  {len(no_trans_rows)}")
    if shard is not None:
        print(f"  Shard:             {shard[0]}/{shard[1]}")

    if not no_trans_rows:
        print("  Nothing to repair.")
        return False

    lookup = load_translations_for_exp3(language)
    if not lookup:
        print(f"  No translations found on disk for '{language}'. Cannot repair.")
        return False

    fixable_all = [r for r in no_trans_rows if r.get("post_full", "") in lookup]
    real_gap    = [r for r in no_trans_rows if r.get("post_full", "") not in lookup]

    print(f"  Fixable (in lookup): {len(fixable_all)}")
    print(f"  Real gap (missing) : {len(real_gap)}  (left as 'no_translation')")

    # Deterministic order so every shard sees the same indexing regardless of
    # which PC it runs on.
    fixable_all.sort(key=lambda r: r.get("index", 0))

    if shard is not None:
        start, end = _shard_slice(len(fixable_all), shard)
        fixable = fixable_all[start:end]
        print(f"  Shard slice:       [{start}:{end}]  "
              f"({len(fixable)} rows of {len(fixable_all)} total fixable)")
    else:
        start, end = 0, len(fixable_all)
        fixable = fixable_all

    # Load checkpoint (shard-scoped if we're in shard mode)
    already_repaired = _load_checkpoint(path, shard)
    if already_repaired:
        print(f"  Resume: found checkpoint with {len(already_repaired)} entries already "
              f"done — those will be skipped.")
        remaining = [r for r in fixable if r.get("index") not in already_repaired]
    else:
        remaining = fixable

    print(f"  To process this run : {len(remaining)}")

    if not fixable_all:
        print("  Nothing fixable — all 'no_translation' rows are real gaps.")
        return False

    if not fixable:
        print("  Nothing in this shard — slice is empty.")
        return False

    if not remaining:
        print("  Nothing left to do — checkpoint already covers this shard's fixable rows.")
        print("  Finalising output...")

    if dry_run:
        print("  --dry-run set. No model calls, no file written.")
        return True

    try:
        provider_cls, default_model, display_name, provider_delay = _load_provider(model_key)
    except (ValueError, ImportError) as e:
        print(f"  Cannot load provider for '{model_key}': {e}")
        return False

    delay = max(delay_override, provider_delay)

    try:
        api_key = get_api_key(model_key)
    except EnvironmentError as e:
        print(f"  {e}")
        return False

    provider = provider_cls(api_key=api_key, model_name=default_model)

    prompt_label = f"Repair {len(remaining)} entries"
    if already_repaired:
        prompt_label += f" (resuming; {len(already_repaired)} already done)"
    confirm = input(f"  {prompt_label} with {display_name}? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Skipped.")
        return False

    # Seed with anything already on the checkpoint, then process remaining
    lang_display = _LANG_DISPLAY.get(language, language.capitalize())
    repaired_by_idx: Dict[int, dict] = dict(already_repaired)

    total_new = len(remaining)
    try:
        for i, r in enumerate(remaining, 1):
            idx = r.get("index")
            post_full = r.get("post_full", "")
            entry = {
                **r,
                "translation":       lookup[post_full],
                "original_language": r.get("original_language") or lang_display,
                "error_exp3":        "",
            }
            overall = len(already_repaired) + i
            logger.info(
                f"[{model_key}][{language}] Repairing {i}/{total_new} this run "
                f"({overall}/{len(fixable)} overall, index={idx})..."
            )

            try:
                repaired = _classify_one(provider, entry, language)
            except Exception as e:
                logger.error(f"  failed on index={idx}: {e}")
                repaired = {**entry, "keywords_exp3": [], "agreement": "no_translation",
                            "raw_response_exp3": "", "error_exp3": f"repair_failed: {e}"}

            repaired_by_idx[idx] = repaired

            # Periodic checkpoint
            if i % CHECKPOINT_EVERY == 0:
                _save_checkpoint(path, repaired_by_idx, shard)

            if i < total_new and delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n  Interrupted. Saving checkpoint so you can resume later...")
        _save_checkpoint(path, repaired_by_idx, shard)
        print(f"  Checkpoint saved: {_checkpoint_path(path, shard).name}  "
              f"({len(repaired_by_idx)} / {len(fixable)} done)")
        print("  Re-run the same command to pick up where you left off.")
        return False
    except Exception:
        # Any unexpected crash — still save progress
        _save_checkpoint(path, repaired_by_idx, shard)
        raise

    # Full run succeeded — no need to keep checkpoint
    _save_checkpoint(path, repaired_by_idx, shard)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_key.replace("/", "_").replace(":", "_")
    out_dir   = path.parent

    # --- Shard mode: write only the repaired slice, let --merge-shards stitch ---
    if shard is not None:
        out_path = out_dir / f"{safe_name}_{language}_{timestamp}_{_shard_suffix(shard)}.json"
        payload = {
            "metadata": {
                "model":          model_key,
                "language":       language,
                "experiment":     3,
                "timestamp":      timestamp,
                "repair_source":  path.name,
                "shard":          f"{shard[0]}/{shard[1]}",
                "shard_range":    [start, end],   # half-open, into sorted fixable_all
                "fixable_total":  len(fixable_all),
                "repaired_rows":  len(repaired_by_idx),
            },
            # Indexed by the original row's "index" field — merge stitches by this.
            "repaired": {str(k): v for k, v in repaired_by_idx.items()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n  Shard summary:")
        print(f"    Shard:              {shard[0]}/{shard[1]}")
        print(f"    Slice:              [{start}:{end}]")
        print(f"    Repaired in shard:  {len(repaired_by_idx)}")
        print(f"  Saved shard output: {out_path.relative_to(PROJECT_ROOT)}")
        print(f"  When all shards are in place, run:")
        print(f"    python scripts/phase2/repair_exp3_translations.py "
              f"--merge-shards \"{path.relative_to(PROJECT_ROOT)}\"")

        _clear_checkpoint(path, shard)
        return True

    # --- Non-shard mode: merge back into full results and write canonical file ---
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

    out_path = out_dir / f"{safe_name}_{language}_{timestamp}.json"

    payload = {
        "metadata": {
            "model":         model_key,
            "language":      language,
            "experiment":    3,
            "timestamp":     timestamp,
            "sample_size":   len(merged),
            "repair_source": path.name,
            "repaired_rows": len(repaired_by_idx),
        },
        "metrics": new_metrics,
        "results": merged,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Final output is committed — clean up the checkpoint
    _clear_checkpoint(path)
    return True


# ── merge shards ──────────────────────────────────────────────────────────────

def merge_shards(source_path: Path) -> bool:
    """
    Find every ``*_shard_NofM.json`` file next to ``source_path`` whose
    metadata points back at ``source_path.name``, stitch their repaired
    rows together, merge into the original results, recompute metrics,
    and write a canonical repaired file.
    """
    if not source_path.exists():
        print(f"  Source file not found: {source_path}")
        return False

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata  = data.get("metadata", {})
    results   = data.get("results", [])
    model_key = metadata.get("model", "")
    language  = metadata.get("language", "")

    # Discover shard outputs in the same directory
    shard_entries: list[tuple[Path, dict]] = []
    for p in source_path.parent.glob("*.json"):
        if p == source_path:
            continue
        if "_shard_" not in p.stem:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                sd = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        smeta = sd.get("metadata", {})
        if smeta.get("repair_source") != source_path.name:
            continue
        shard_entries.append((p, sd))

    if not shard_entries:
        print(f"  No shard output files found for {source_path.name} in "
              f"{source_path.parent}.")
        return False

    # De-dup: if multiple outputs exist for the same 'shard' string (e.g.
    # someone re-ran a shard), keep the newest mtime.
    best_by_shard: Dict[str, tuple[Path, dict]] = {}
    for p, sd in shard_entries:
        shard_str = sd.get("metadata", {}).get("shard", "?")
        if shard_str not in best_by_shard or p.stat().st_mtime > best_by_shard[shard_str][0].stat().st_mtime:
            best_by_shard[shard_str] = (p, sd)

    print(f"\n  Merging {len(best_by_shard)} shard file(s) for {source_path.name}:")
    for shard_str, (p, sd) in sorted(best_by_shard.items()):
        smeta = sd.get("metadata", {})
        print(f"    shard={shard_str:>5}  repaired={smeta.get('repaired_rows'):>4}  "
              f"range={smeta.get('shard_range')}  ({p.name})")

    # Gather every repaired row from every shard
    repaired_by_idx: Dict[int, dict] = {}
    collisions = 0
    for _, sd in best_by_shard.values():
        for k, v in sd.get("repaired", {}).items():
            idx = int(k)
            if idx in repaired_by_idx:
                collisions += 1
            repaired_by_idx[idx] = v
    if collisions:
        print(f"  Note: {collisions} row index(es) appeared in multiple shards "
              f"(slices should be disjoint — check your --shard args).")

    # Shards can also tell us how many fixable rows the source actually has
    fixable_totals = {sd.get("metadata", {}).get("fixable_total")
                      for _, sd in best_by_shard.values()
                      if sd.get("metadata", {}).get("fixable_total") is not None}
    expected_total = max(fixable_totals) if fixable_totals else None
    if expected_total is not None:
        print(f"  Repaired rows collected: {len(repaired_by_idx)} / {expected_total} "
              f"fixable in source.")
        if len(repaired_by_idx) < expected_total:
            missing = expected_total - len(repaired_by_idx)
            print(f"  WARNING: {missing} fixable row(s) not covered by any shard. "
                  f"Missing shard(s)? Merging anyway; the rest stay 'no_translation'.")

    # Merge into original results and recompute
    merged_by_idx = {r.get("index"): r for r in results}
    merged_by_idx.update(repaired_by_idx)
    merged = sorted(merged_by_idx.values(), key=lambda x: x.get("index", 0))
    new_metrics = _recompute_metrics(merged)

    still_nt = sum(1 for r in merged if r.get("agreement") == "no_translation")
    new_yes  = sum(1 for r in merged if r.get("agreement") == "yes")
    new_no   = sum(1 for r in merged if r.get("agreement") == "no")
    print(f"\n  Merge summary:")
    print(f"    Repaired (from shards): {len(repaired_by_idx)}")
    print(f"    Still 'no_translation': {still_nt}")
    print(f"    New 'yes'/'no' total:   {new_yes} / {new_no}")
    print(f"    New agreement_rate:     {new_metrics['agreement_rate']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_key.replace("/", "_").replace(":", "_")
    out_path  = source_path.parent / f"{safe_name}_{language}_{timestamp}.json"

    payload = {
        "metadata": {
            "model":           model_key,
            "language":        language,
            "experiment":      3,
            "timestamp":       timestamp,
            "sample_size":     len(merged),
            "repair_source":   source_path.name,
            "merged_shards":   sorted(best_by_shard.keys()),
            "repaired_rows":   len(repaired_by_idx),
        },
        "metrics": new_metrics,
        "results": merged,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  Saved merged: {out_path.relative_to(PROJECT_ROOT)}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill Exp3 entries that were wrongly marked 'no_translation'."
    )
    parser.add_argument("--file",  type=Path, default=None,
                        help="Specific exp3 result JSON to repair. If omitted, the script "
                             "scans results/phase2/experiment3/ and repairs every file it finds "
                             "with fixable 'no_translation' rows.")
    parser.add_argument("--model", type=str, default=None,
                        help="When auto-scanning, only repair files for this model key "
                             "(e.g. 'gemma', 'deepseek-local', 'llama').")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between model calls (default: 0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change, but don't call the model or write anything.")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard N/M of the fixable rows (e.g. --shard 1/3). "
                             "Writes a shard-specific output file that --merge-shards can stitch. "
                             "Use the SAME --file path on every PC so shard boundaries match.")
    parser.add_argument("--merge-shards", dest="merge_shards", type=Path, default=None,
                        help="Path to the ORIGINAL broken source file. Stitches every shard "
                             "output next to it into one canonical repaired file.")
    args = parser.parse_args()

    # Merge mode short-circuits everything else
    if args.merge_shards is not None:
        merge_shards(args.merge_shards)
        return

    try:
        shard = _parse_shard(args.shard)
    except ValueError as e:
        print(f"  {e}")
        sys.exit(1)

    if args.file is not None:
        # Explicit single-file mode
        if not args.file.exists():
            print(f"  File not found: {args.file}")
            sys.exit(1)
        repair_file(args.file, args.delay, args.dry_run, shard=shard)
        return

    # Auto-scan mode
    candidates = find_candidate_files(model_filter=args.model)
    if not candidates:
        suffix = f" for model '{args.model}'" if args.model else ""
        print(f"\n  No exp3 files with 'no_translation' rows found{suffix}.")
        return

    print(f"\n  Found {len(candidates)} candidate file(s) with 'no_translation' rows:")
    for i, c in enumerate(candidates, 1):
        rel = c["path"].relative_to(PROJECT_ROOT)
        print(f"    [{i}] {c['model']:15} / {c['language']:8} — "
              f"{c['no_trans']:>4}/{c['total']} broken   ({rel})")

    choice = input(
        f"\n  Enter file number (1-{len(candidates)}), 'a' for all, or 'q' to quit: "
    ).strip().lower()

    if choice in ("q", "n", ""):
        print("  Aborted.")
        return

    if choice in ("a", "all"):
        if shard is not None:
            print("  --shard can only run against one file at a time. Pick a number.")
            return
        selected = candidates
    else:
        try:
            n = int(choice)
            if not 1 <= n <= len(candidates):
                raise ValueError
            selected = [candidates[n - 1]]
        except ValueError:
            print(f"  Invalid choice: {choice!r}. Aborted.")
            return

    repaired_files = 0
    for c in selected:
        did_work = repair_file(c["path"], args.delay, args.dry_run, shard=shard)
        if did_work:
            repaired_files += 1

    print(f"\n  Done. Processed {repaired_files}/{len(selected)} file(s).")


if __name__ == "__main__":
    main()
