"""Merge Experiment 4 JSON outputs into the per-language error-analysis CSVs.

For each language:
    1. Load results/all_models_wrong/{language}_all_wrong.csv (15-row Exp 2 baseline).
    2. Scan results/phase2/experiment4/{model}/ for the newest JSON per (model, language).
    3. Join on `index` and insert two columns per model after its _prediction column:
            {model}_exp4_classification
            {model}_exp4_justification
    4. Write the CSV back in place (few-shot) or to {language}_zeroshot.csv (--zeroshot).

Run any time after Exp 4 has produced at least one JSON. Missing models leave
their new columns blank. Re-running picks up newer JSONs automatically.

Usage:
    python scripts/merge_exp4.py
    python scripts/merge_exp4.py --zeroshot
"""

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Optional

REPO        = Path(__file__).resolve().parent.parent
CSV_DIR     = REPO / "results" / "all_models_wrong"
EXP4_DIR    = REPO / "results" / "phase2" / "experiment4"
EXP4_ZS_DIR = REPO / "results" / "phase2" / "experiment4_zeroshot"
LANGUAGES   = ["arabic", "chinese", "urdu"]

CSV_MODELS     = ["claude", "openai", "gemini", "gemma", "llama", "qwen"]
EXP4_TO_CSV   = {m: m for m in CSV_MODELS}
EXP4_KEY_ORDER = list(EXP4_TO_CSV)

DROP_COLS = {f"{m}{s}" for m in CSV_MODELS for s in ("_keywords", "_keywords_en", "_keyword_evaluation")}


def newest_json(model_key: str, language: str, zeroshot: bool = False) -> Optional[Path]:
    folder  = (EXP4_ZS_DIR if zeroshot else EXP4_DIR) / model_key
    matches = sorted(glob.glob(str(folder / f"{model_key}_{language}_*.json"))) if folder.is_dir() else []
    return Path(matches[-1]) if matches else None


def load_exp4(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        return {int(r["index"]): r for r in json.load(fh)["results"]}


def reorder_fieldnames(existing: list[str]) -> list[str]:
    """Insert exp4 columns after each model's _prediction column. Idempotent."""
    out, inserted = [], set()
    for col in existing:
        out.append(col)
        if col.endswith("_prediction"):
            prefix = col[: -len("_prediction")]
            for new in (f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"):
                if new not in existing and new not in out:
                    out.append(new)
                    inserted.add(prefix)
    for prefix in CSV_MODELS:
        if prefix not in inserted:
            for new in (f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"):
                if new not in out:
                    out.append(new)
    return out


def process_language(language: str, zeroshot: bool = False) -> tuple[int, dict, dict]:
    base_csv = CSV_DIR / f"{language}_all_wrong.csv"
    if not base_csv.exists():
        print(f"  [{language}] base CSV missing: {base_csv}")
        return 0, {}, {}

    out_csv = CSV_DIR / f"{language}_zeroshot.csv" if zeroshot else base_csv

    with open(base_csv, encoding="utf-8-sig", newline="") as fh:
        reader          = csv.DictReader(fh)
        original_fields = reader.fieldnames or []
        rows            = list(reader)

    exp4_data: dict[str, dict[int, dict]] = {}
    seen_files: dict[str, str]            = {}
    for key in EXP4_KEY_ORDER:
        path = newest_json(key, language, zeroshot=zeroshot)
        if not path:
            continue
        prefix             = EXP4_TO_CSV[key]
        exp4_data[prefix]  = load_exp4(path)
        seen_files[prefix] = f"{key}/{path.name}"

    if not exp4_data:
        variant = "zero-shot" if zeroshot else "few-shot"
        print(f"  [{language}] no {variant} Exp 4 JSONs found — skipping")
        return 0, {}, {}

    if not rows:
        print(f"  [{language}] base CSV is empty — rebuilding from JSON files")
        seen: dict[int, dict] = {}
        for prefix, results in exp4_data.items():
            for idx, r in results.items():
                if idx not in seen:
                    seen[idx] = {
                        "index": idx, "conversation": r.get("post_full", ""),
                        "translation": r.get("translation", ""), "actual_value": r.get("ground_truth", ""),
                    }
        rows = [seen[i] for i in sorted(seen)]

    kept_fields = [f for f in original_fields if f not in DROP_COLS] or \
                  ["index", "conversation", "translation", "actual_value"]
    new_fields  = reorder_fieldnames(kept_fields)

    per_model_filled = {p: 0 for p in CSV_MODELS}
    for row in rows:
        idx = int(row["index"])
        for prefix in CSV_MODELS:
            cls_col, just_col = f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"
            row.setdefault(cls_col, "")
            row.setdefault(just_col, "")
            results = exp4_data.get(prefix)
            if not results:
                continue
            hit = results.get(idx)
            if hit:
                row[cls_col]  = hit.get("exp4_classification", "")
                row[just_col] = hit.get("exp4_justification", "")
                per_model_filled[prefix] += 1

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), per_model_filled, seen_files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zeroshot", action="store_true",
                   help="Merge zero-shot results (experiment4_zeroshot/ → {language}_zeroshot.csv)")
    args = p.parse_args()

    total_rows = 0
    for lang in LANGUAGES:
        result = process_language(lang, zeroshot=args.zeroshot)
        n_rows = result[0]
        if not n_rows:
            continue
        _, per_model, seen_files = result
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
