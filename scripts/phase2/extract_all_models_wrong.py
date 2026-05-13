"""Produce one CSV per language: 15 rows where all 6 models predicted wrong AND
every model produced non-empty keywords.

Sampling is balanced by ground truth (target half not-depressed, half depressed;
fills from majority class if minority pool is too small). Selection is
deterministic — sorted by index within each class — so re-runs are reproducible.

These 15-row CSVs feed Experiment 4 (re-prompting each model on the same rows
to elicit a justification for its classification).

Output: results/all_models_wrong/{language}_all_wrong.csv (UTF-8 with BOM for Excel).
"""

import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E2 = REPO / "results" / "phase2" / "experiment2"
E3 = REPO / "results" / "phase2" / "experiment3"
OUT_DIR = REPO / "results" / "all_models_wrong"

MODELS = ["claude", "openai", "gemini", "gemma", "llama", "deepseek"]
LIMIT_PER_LANG = 15

MODEL_DIR = {
    "claude":   E2 / "Online Models" / "claude",
    "gemini":   E2 / "Online Models" / "gemini",
    "openai":   E2 / "Online Models" / "openai",
    "gemma":    E2 / "Local Models" / "gemma",
    "llama":    E2 / "Local Models" / "meta-llama-3.1-8b-instruct",
    "deepseek": E2 / "Local Models" / "deepseek-r1-0528-qwen3-8b",
}

MODEL_PREFIX = {
    "claude":   "claude",
    "gemini":   "gemini",
    "openai":   "openai",
    "gemma":    "gemma",
    "llama":    "llama",
    "deepseek": "deepseek-local",
}

EXP3_TRANSLATION_DIR = E3 / "Online Models" / "gemini"


def newest(folder: Path, pattern: str) -> Path:
    matches = sorted(glob.glob(str(folder / pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matched {folder}/{pattern}")
    return Path(matches[-1])


def load_results(path: Path) -> dict[int, dict]:
    with open(path) as fh:
        return {r["index"]: r for r in json.load(fh)["results"]}


def norm(label) -> str:
    return str(label).strip().lower() if label is not None else ""


def fmt_list(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def balanced_sample(rows: list[dict], limit: int) -> list[dict]:
    """Pick up to `limit` rows balanced by actual_value, deterministic by index."""
    not_dep = sorted([r for r in rows if norm(r["actual_value"]) == "not depressed"], key=lambda r: r["index"])
    dep     = sorted([r for r in rows if norm(r["actual_value"]) == "depressed"],     key=lambda r: r["index"])

    target_not = limit // 2
    target_dep = limit - target_not

    n_not = min(target_not, len(not_dep))
    n_dep = min(target_dep, len(dep))
    # If one class is short, top up from the other so we hit `limit` if possible.
    shortfall = limit - n_not - n_dep
    if shortfall > 0:
        if n_dep < len(dep):
            extra = min(shortfall, len(dep) - n_dep)
            n_dep += extra
            shortfall -= extra
        if shortfall > 0 and n_not < len(not_dep):
            n_not += min(shortfall, len(not_dep) - n_not)

    chosen = not_dep[:n_not] + dep[:n_dep]
    chosen.sort(key=lambda r: r["index"])
    return chosen


def process_language(language: str) -> tuple[Path, int, int, int]:
    model_data = {}
    for m in MODELS:
        path = newest(MODEL_DIR[m], f"{MODEL_PREFIX[m]}_{language}_*.json")
        model_data[m] = load_results(path)

    translation_src = load_results(newest(EXP3_TRANSLATION_DIR, f"gemini_{language}_*.json"))

    indices = sorted(set.intersection(*[set(d.keys()) for d in model_data.values()]))

    candidates = []
    for idx in indices:
        rows = [model_data[m][idx] for m in MODELS]
        gt_set = {norm(r["ground_truth"]) for r in rows}
        if len(gt_set) != 1:
            continue
        gt = next(iter(gt_set))
        if not all(norm(r["prediction"]) != gt for r in rows):
            continue
        # Require every model to have produced non-empty keywords.
        if not all(r.get("keywords") for r in rows):
            continue

        first = rows[0]
        record = {
            "index": idx,
            "conversation": first["post_full"],
            "translation": translation_src.get(idx, {}).get("translation", ""),
            "actual_value": first["ground_truth"],
        }
        for m, r in zip(MODELS, rows):
            record[f"{m}_prediction"] = r["prediction"]
            record[f"{m}_keywords"] = fmt_list(r.get("keywords"))
            record[f"{m}_keywords_en"] = fmt_list(r.get("translations"))
            record[f"{m}_keyword_evaluation"] = ""
        record["human_eval_response"] = ""
        record["human_eval_keywords"] = ""
        candidates.append(record)

    pool_total = len(candidates)
    chosen = balanced_sample(candidates, LIMIT_PER_LANG)
    n_dep = sum(1 for r in chosen if norm(r["actual_value"]) == "depressed")
    n_not = len(chosen) - n_dep

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{language}_all_wrong.csv"
    fieldnames = (
        ["index", "conversation", "translation", "actual_value"]
        + [f"{m}_{suffix}" for m in MODELS for suffix in ("prediction", "keywords", "keywords_en", "keyword_evaluation")]
        + ["human_eval_response", "human_eval_keywords"]
    )
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen)

    return out_path, pool_total, n_dep, n_not


def main():
    for lang in ["arabic", "chinese", "urdu"]:
        out_path, pool_total, n_dep, n_not = process_language(lang)
        rel = out_path.relative_to(REPO)
        print(f"{lang:8s}  pool(all-wrong & all-kw)={pool_total:4d}  picked={n_dep + n_not:2d}  (dep={n_dep}, not_dep={n_not})  {rel}")


if __name__ == "__main__":
    main()
