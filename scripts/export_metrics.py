"""Export Phase 2 Experiment 1 metrics to a compact CSV.

Scans results/phase2/experiment1/, ignores partial files, and emits:

    Model, Language, Accuracy, Precision, Recall, F1 Score

One row per (model, language), keeping only the latest run for each pair.

Usage:
    python scripts/export_metrics.py
    python scripts/export_metrics.py --out results/phase2/experiment1/exp1_metrics_summary.csv
    python scripts/export_metrics.py --models gemini,claude,openai
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_DIR     = PROJECT_ROOT / "results" / "phase2" / "experiment1"


@dataclass(frozen=True)
class Row:
    model:       str
    language:    str
    accuracy:    float
    precision:   float
    recall:      float
    f1_score:    float
    timestamp:   str
    source_file: Path


def _iter_json_files(root: Path) -> Iterable[Path]:
    if root.exists():
        yield from root.rglob("*.json")


def _is_partial(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".partial.json") or name.startswith("comparison_")


def _model_matches(model: str, selected: List[str]) -> bool:
    m = (model or "").strip().lower()
    tokens = {s.strip().lower() for s in selected}
    if "gemini"   in tokens and m == "gemini":           return True
    if "claude"   in tokens and m == "claude":           return True
    if "openai"   in tokens and m == "openai":           return True
    if "deepseek" in tokens and m.startswith("deepseek"): return True
    if any(t in ("llama", "meta-llama") for t in tokens) and m.startswith("meta-llama"): return True
    return False


def _load_row(path: Path) -> Optional[Row]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    meta    = data.get("metadata") or {}
    metrics = data.get("metrics")  or {}
    if meta.get("experiment") != 1:
        return None
    model, language, timestamp = meta.get("model"), meta.get("language"), meta.get("timestamp")
    if not (model and language and timestamp):
        return None
    try:
        return Row(
            model=str(model), language=str(language), timestamp=str(timestamp),
            accuracy=float(metrics["accuracy"]),   precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),        f1_score=float(metrics["f1_score"]),
            source_file=path,
        )
    except Exception:
        return None


def latest_per_pair(rows: List[Row]) -> List[Row]:
    best: Dict[Tuple[str, str], Row] = {}
    for r in rows:
        key = (r.model, r.language)
        if key not in best or r.timestamp > best[key].timestamp:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r.model, r.language))


def write_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Language", "Accuracy", "Precision", "Recall", "F1 Score"])
        for r in rows:
            w.writerow([r.model, r.language,
                        f"{r.accuracy:.4f}", f"{r.precision:.4f}",
                        f"{r.recall:.4f}",   f"{r.f1_score:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(EXP1_DIR / "exp1_metrics_summary.csv"))
    parser.add_argument("--models", default="gemini,claude,openai,deepseek,meta-llama")
    args = parser.parse_args()

    selected = [s.strip() for s in args.models.split(",") if s.strip()]
    rows: List[Row] = []
    for p in _iter_json_files(EXP1_DIR):
        if _is_partial(p):
            continue
        row = _load_row(p)
        if row and _model_matches(row.model, selected):
            rows.append(row)

    out_rows = latest_per_pair(rows)
    out_path = PROJECT_ROOT / Path(args.out)
    write_csv(out_rows, out_path)

    print(f"Wrote {len(out_rows)} rows → {out_path}")
    for r in out_rows:
        print(f"  {r.model} / {r.language}: acc={r.accuracy:.4f}  f1={r.f1_score:.4f}  ({r.source_file.name})")


if __name__ == "__main__":
    main()
