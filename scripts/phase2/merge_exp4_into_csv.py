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

Mapping notes:
    - exp4 model key "deepseek-local" -> CSV column "deepseek"
      (matches the Exp2 baseline which also used local DeepSeek-R1)
    - exp4 model key "deepseek"        -> also CSV column "deepseek"
      (if you ran the online API; it will overwrite the local one)
"""

import csv
import glob
import json
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
CSV_DIR = REPO / "results" / "all_models_wrong"
EXP4_DIR = REPO / "results" / "phase2" / "experiment4"
LANGUAGES = ["arabic", "chinese", "urdu"]

# CSV column prefixes that exist in {language}_all_wrong.csv
CSV_MODELS = ["claude", "openai", "gemini", "gemma", "llama", "deepseek"]

# Map exp4 model key -> CSV column prefix.
EXP4_TO_CSV = {
    "claude":         "claude",
    "openai":         "openai",
    "gemini":         "gemini",
    "gemma":          "gemma",
    "llama":          "llama",
    "deepseek-local": "deepseek",  # local DeepSeek-R1 (matches Exp2 baseline)
    "deepseek":       "deepseek",  # online API — overrides local if both ran
}

# Order in which to apply exp4-model -> CSV-column mappings. If multiple exp4
# models target the same CSV column, the LAST one wins. We list deepseek (online)
# BEFORE deepseek-local so that, if both are present, the local result wins (it's
# what Exp2 used). Reverse the two entries if you'd rather keep the online one.
EXP4_KEY_ORDER = ["claude", "openai", "gemini", "gemma", "llama", "deepseek", "deepseek-local"]


def newest_json(model_key: str, language: str) -> Optional[Path]:
    folder = EXP4_DIR / model_key
    if not folder.is_dir():
        return None
    matches = sorted(glob.glob(str(folder / f"{model_key}_{language}_*.json")))
    return Path(matches[-1]) if matches else None


def load_exp4(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        return {int(r["index"]): r for r in json.load(fh)["results"]}


def reorder_fieldnames(existing: list[str]) -> list[str]:
    """Insert {prefix}_exp4_classification and _justification after each prefix's
    _keyword_evaluation column. Idempotent — if they're already there, leave alone."""
    out = []
    for col in existing:
        out.append(col)
        if col.endswith("_keyword_evaluation"):
            prefix = col[: -len("_keyword_evaluation")]
            for new in (f"{prefix}_exp4_classification", f"{prefix}_exp4_justification"):
                if new not in existing and new not in out:
                    out.append(new)
    return out


def process_language(language: str) -> tuple[int, dict[str, int]]:
    csv_path = CSV_DIR / f"{language}_all_wrong.csv"
    if not csv_path.exists():
        print(f"  [{language}] base CSV missing: {csv_path}")
        return 0, {}

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        original_fields = reader.fieldnames or []
        rows = list(reader)

    # Pull in the newest JSON per exp4 model that has been run for this language.
    exp4_by_csv_prefix: dict[str, dict[int, dict]] = {}
    seen_files: dict[str, str] = {}
    for exp4_key in EXP4_KEY_ORDER:
        path = newest_json(exp4_key, language)
        if not path:
            continue
        csv_prefix = EXP4_TO_CSV[exp4_key]
        exp4_by_csv_prefix[csv_prefix] = load_exp4(path)
        seen_files[csv_prefix] = f"{exp4_key}/{path.name}"

    if not exp4_by_csv_prefix:
        print(f"  [{language}] no Exp4 JSONs found yet — leaving CSV unchanged")
        return 0, {}

    new_fields = reorder_fieldnames(original_fields)

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

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), per_model_filled, seen_files


def main():
    total_rows = 0
    for lang in LANGUAGES:
        out = process_language(lang)
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

    print(f"\nDone. Touched {total_rows} rows across all languages.")
    print(f"CSVs updated in place at: {CSV_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
