"""Merge Experiment 4 JSON outputs back into the 15-row per-language CSVs.

For each language:
    1. Load results/all_models_wrong/{language}_all_wrong.csv (15 rows from Exp2 baseline).
    2. Scan results/phase2/experiment4/{model}/ for the NEWEST JSON per (model, language).
    3. Join by `index`, and insert two columns per model immediately after that
       model's existing `_keyword_evaluation` column:
            {csv_prefix}_exp4_classification
            {csv_prefix}_exp4_justification
    4. Write the CSV back in place.

Run anytime after Exp4 has produced at least one JSON. Missing models leave their
new columns blank. Re-running picks up newer JSONs automatically.

Note: `deepseek` here refers to the LOCAL DeepSeek-R1 model served by LM Studio,
matching the Exp2 baseline. (run_exp4.py uses the same convention.)
"""

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
CSV_DIR    = REPO / "results" / "all_models_wrong"
EXP4_DIR   = REPO / "results" / "phase2" / "experiment4"
EXP4_ZS_DIR = REPO / "results" / "phase2" / "experiment4_zeroshot"
LANGUAGES  = ["arabic", "chinese", "urdu"]

# CSV column prefixes — includes qwen as a new model for Exp4 onwards
CSV_MODELS = ["claude", "openai", "gemini", "gemma", "llama", "qwen"]

EXP4_TO_CSV = {m: m for m in CSV_MODELS}
EXP4_KEY_ORDER = list(EXP4_TO_CSV)

# Drop per-model keyword columns only (not human_eval_* columns)
DROP_COLS = {f"{m}{s}" for m in CSV_MODELS for s in ("_keywords", "_keywords_en", "_keyword_evaluation")}


def newest_json(model_key: str, language: str, zeroshot: bool = False) -> Optional[Path]:
    folder = (EXP4_ZS_DIR if zeroshot else EXP4_DIR) / model_key
    if not folder.is_dir():
        return None
    matches = sorted(glob.glob(str(folder / f"{model_key}_{language}_*.json")))
    return Path(matches[-1]) if matches else None


def load_exp4(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        return {int(r["index"]): r for r in json.load(fh)["results"]}


def reorder_fieldnames(existing: list[str]) -> list[str]:
    """Insert {prefix}_exp4_classification and _justification after each prefix's
    _prediction column. Idempotent."""
    out = []
    inserted = set()
    for col in existing:
        out.append(col)
        if col.endswith("_prediction"):
            prefix = col[: -len("_prediction")]
            for new in (f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"):
                if new not in existing and new not in out:
                    out.append(new)
                    inserted.add(prefix)
    # Append exp4 columns for models with no _prediction column (e.g. qwen)
    for prefix in CSV_MODELS:
        if prefix not in inserted:
            for new in (f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"):
                if new not in out:
                    out.append(new)
    return out


def process_language(language: str, zeroshot: bool = False) -> tuple[int, dict[str, int]]:
    base_csv = CSV_DIR / f"{language}_all_wrong.csv"
    if not base_csv.exists():
        print(f"  [{language}] base CSV missing: {base_csv}")
        return 0, {}

    # Zero-shot writes to a separate file; few-shot updates in place.
    out_csv = CSV_DIR / f"{language}_zeroshot.csv" if zeroshot else base_csv

    with open(base_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        original_fields = reader.fieldnames or []
        rows = list(reader)

    exp4_by_csv_prefix: dict[str, dict[int, dict]] = {}
    seen_files: dict[str, str] = {}
    for exp4_key in EXP4_KEY_ORDER:
        path = newest_json(exp4_key, language, zeroshot=zeroshot)
        if not path:
            continue
        csv_prefix = EXP4_TO_CSV[exp4_key]
        exp4_by_csv_prefix[csv_prefix] = load_exp4(path)
        seen_files[csv_prefix] = f"{exp4_key}/{path.name}"

    if not exp4_by_csv_prefix:
        variant = "zero-shot" if zeroshot else "few-shot"
        print(f"  [{language}] no {variant} Exp4 JSONs found yet — skipping")
        return 0, {}

    # Base CSV is empty (e.g. wiped by a failed previous run) — rebuild rows from JSONs.
    if not rows:
        print(f"  [{language}] base CSV is empty — rebuilding from JSON files")
        seen_indices = {}
        for prefix, results in exp4_by_csv_prefix.items():
            for idx, r in results.items():
                if idx not in seen_indices:
                    seen_indices[idx] = {
                        "index": idx,
                        "conversation": r.get("post_full", ""),
                        "translation": r.get("translation", ""),
                        "actual_value": r.get("ground_truth", ""),
                    }
        rows = [seen_indices[i] for i in sorted(seen_indices)]

    kept_fields = [f for f in original_fields if f not in DROP_COLS]
    # If original_fields was empty (wiped CSV), build a minimal field list.
    if not kept_fields:
        kept_fields = ["index", "conversation", "translation", "actual_value"]
    new_fields = reorder_fieldnames(kept_fields)

    per_model_filled = {p: 0 for p in CSV_MODELS}
    for row in rows:
        idx = int(row["index"])
        for csv_prefix in CSV_MODELS:
            cls_col = f"{csv_prefix}_exp4_classification"
            just_col = f"{csv_prefix}_exp4_justification"
            row.setdefault(cls_col, "")
            row.setdefault(just_col, "")
            results = exp4_by_csv_prefix.get(csv_prefix)
            if not results:
                continue
            hit = results.get(idx)
            if hit:
                row[cls_col] = hit.get("exp4_classification", "")
                row[just_col] = hit.get("exp4_justification", "")
                per_model_filled[csv_prefix] += 1

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), per_model_filled, seen_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeroshot", action="store_true", help="Merge zero-shot results (reads from experiment4_zeroshot/, writes {language}_zeroshot.csv).")
    args = p.parse_args()

    total_rows = 0
    for lang in LANGUAGES:
        out = process_language(lang, zeroshot=args.zeroshot)
        if not out:
            continue
        n_rows = out[0]
        if n_rows == 0:
            continue
        _, per_model, seen_files = out
        total_rows += n_rows
        filled = ", ".join(f"{m}={n}" for m, n in per_model.items() if n)
        print(f"  [{lang}] merged {n_rows} rows; filled: {filled or '(nothing)'}")
        for prefix, fname in seen_files.items():
            print(f"      {prefix:9s} <- {fname}")

    variant = "zero-shot" if args.zeroshot else "few-shot"
    print(f"\nDone ({variant}). Touched {total_rows} rows across all languages.")
    print(f"CSVs at: {CSV_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
