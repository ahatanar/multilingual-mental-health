"""Export Phase 2 Experiment 1 metrics to a compact CSV.

Scans results/phase2/experiment1/ (including Local Models subfolders),
ignores partial files, and emits a CSV with the columns:

    Model, Language, Accuracy, Precision, Recall, F1 Score

By default, it exports the latest run per (model, language) for:
- gemini
- claude
- openai
- deepseek (includes deepseek-local if present)
- meta-llama (any model starting with "meta-llama")

Usage (from repo root):
    py scripts/phase2/export_experiment1_metrics_csv.py

Optional:
    py scripts/phase2/export_experiment1_metrics_csv.py --out results/phase2/experiment1/exp1_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP1_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment1"


@dataclass(frozen=True)
class Row:
    model: str
    language: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: str
    source_file: Path


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    yield from root.rglob("*.json")


def _is_partial_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".partial.json"):
        return True
    if name.endswith(".json.retry.partial"):
        return True
    if name.startswith("comparison_"):
        return True
    return False


def _model_is_selected(model: str, selected: List[str]) -> bool:
    m = (model or "").strip().lower()

    # normalize user-friendly tokens
    normalized = set(s.strip().lower() for s in selected)

    if "gemini" in normalized and m == "gemini":
        return True
    if "claude" in normalized and m == "claude":
        return True
    if "openai" in normalized and m == "openai":
        return True

    # DeepSeek: accept "deepseek" provider key, and "deepseek-local" runs
    if "deepseek" in normalized and (m == "deepseek" or m.startswith("deepseek-")):
        return True

    # Meta Llama: local runs store the concrete model id
    if ("meta llama" in normalized or "meta-llama" in normalized or "llama" in normalized) and m.startswith("meta-llama"):
        return True

    return False


def _load_row(path: Path) -> Optional[Row]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    meta = data.get("metadata") or {}
    metrics = data.get("metrics") or {}

    if meta.get("experiment") != 1:
        return None

    model = meta.get("model")
    language = meta.get("language")
    timestamp = meta.get("timestamp")

    # Required fields
    if not model or not language or not timestamp:
        return None

    try:
        return Row(
            model=str(model),
            language=str(language),
            accuracy=float(metrics["accuracy"]),
            precision=float(metrics["precision"]),
            recall=float(metrics["recall"]),
            f1_score=float(metrics["f1_score"]),
            timestamp=str(timestamp),
            source_file=path,
        )
    except Exception:
        return None


def _latest_per_model_language(rows: List[Row]) -> List[Row]:
    latest: Dict[Tuple[str, str], Row] = {}
    for r in rows:
        key = (r.model, r.language)
        prev = latest.get(key)
        if prev is None or r.timestamp > prev.timestamp:
            latest[key] = r
    return sorted(latest.values(), key=lambda r: (r.model, r.language))


def write_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Language", "Accuracy", "Precision", "Recall", "F1 Score"])
        for r in rows:
            w.writerow([
                r.model,
                r.language,
                f"{r.accuracy:.4f}",
                f"{r.precision:.4f}",
                f"{r.recall:.4f}",
                f"{r.f1_score:.4f}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Experiment 1 metrics to a compact CSV")
    parser.add_argument(
        "--out",
        type=str,
        default=str(EXP1_DIR / "exp1_metrics_summary.csv"),
        help="Output CSV path (default: results/phase2/experiment1/exp1_metrics_summary.csv)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemini,claude,openai,deepseek,meta-llama",
        help="Comma-separated list of model selectors (default: gemini,claude,openai,deepseek,meta-llama)",
    )
    args = parser.parse_args()

    selected = [s.strip() for s in args.models.split(",") if s.strip()]

    rows: List[Row] = []
    for p in _iter_json_files(EXP1_DIR):
        if _is_partial_file(p):
            continue
        row = _load_row(p)
        if not row:
            continue
        if not _model_is_selected(row.model, selected):
            continue
        rows.append(row)

    latest_rows = _latest_per_model_language(rows)
    out_path = PROJECT_ROOT / Path(args.out)
    write_csv(latest_rows, out_path)

    print(f"Wrote {len(latest_rows)} rows to: {out_path}")

    # Helpful summary to stdout
    for r in latest_rows:
        print(f"- {r.model} / {r.language}: acc={r.accuracy:.4f} f1={r.f1_score:.4f} ({r.source_file.name})")


if __name__ == "__main__":
    main()
