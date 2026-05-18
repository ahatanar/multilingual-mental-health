"""
loader.py — Load and normalise result JSON files into flat pandas DataFrames.

Provides:
  load_file(meta)        → (metrics_dict, samples_df)
  build_registry(files)  → Registry  (keyed by (exp, model, lang))
  build_master_summary() → DataFrame with one row per file
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from discovery import ResultFileMeta


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[loader] ERROR reading {path.name}: {e}")
        return None


def _flatten_metrics(raw: dict) -> dict[str, Any]:
    """Flatten confusion_matrix sub-dict into top-level keys."""
    flat = {}
    cm = raw.get("confusion_matrix", {})
    flat["tp"] = cm.get("true_positives")
    flat["fp"] = cm.get("false_positives")
    flat["tn"] = cm.get("true_negatives")
    flat["fn"] = cm.get("false_negatives")
    for k in ("precision", "recall", "f1_score", "accuracy",
              "total_samples", "total_classified",
              "unclear_responses", "error_responses", "agreement_rate"):
        flat[k] = raw.get(k)
    return flat


def _normalise_keywords(val: Any) -> list[str]:
    """Always return a list of strings from keywords/translations fields."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(k).strip() for k in val if k]
    if isinstance(val, str):
        return [k.strip() for k in val.split(",") if k.strip()]
    return []


def _row_from_result(entry: dict, meta: ResultFileMeta) -> dict[str, Any]:
    """Convert one result dict to a flat row dict."""
    row: dict[str, Any] = {
        "experiment": meta.experiment,
        "model": meta.model,
        "model_type": meta.model_type,
        "language": meta.language,
        "timestamp": meta.timestamp,
        # core classification fields
        "index": entry.get("index"),
        "ground_truth": entry.get("ground_truth"),
        "prediction": entry.get("prediction"),
        "correct": (
            entry.get("prediction") == entry.get("ground_truth")
            if entry.get("prediction") and entry.get("ground_truth")
            else None
        ),
        "word_count": entry.get("word_count", 0) or 0,
        "error": entry.get("error"),
    }

    # Experiment 2 extras
    if meta.experiment == 2:
        kws = _normalise_keywords(entry.get("keywords"))
        trans = _normalise_keywords(entry.get("translations"))
        row["keywords"] = kws
        row["translations"] = trans
        row["keyword_count"] = len(kws)
        row["error_exp2"] = entry.get("error_exp2")

    # Experiment 3 extras
    if meta.experiment == 3:
        kws3 = _normalise_keywords(entry.get("keywords_exp3"))
        row["translation"] = entry.get("translation", "")
        row["original_language"] = entry.get("original_language", "")
        row["keywords_exp3"] = kws3
        row["keyword_count_exp3"] = len(kws3)
        row["agreement"] = entry.get("agreement", "")
        row["error_exp3"] = entry.get("error_exp3")

    return row


# ── core load ─────────────────────────────────────────────────────────────────

def load_file(meta: ResultFileMeta) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """
    Load a single result file.

    Returns (metrics_dict, samples_df) or (None, None) on failure.
    metrics_dict includes all metadata + flattened metrics.
    samples_df has one row per result entry with columns described above.
    """
    raw = _safe_load_json(meta.path)
    if raw is None:
        return None, None

    try:
        # Metrics
        raw_metrics = raw.get("metrics", {})
        metrics = {
            "experiment": meta.experiment,
            "model": meta.model,
            "model_type": meta.model_type,
            "language": meta.language,
            "timestamp": meta.timestamp,
            "file": meta.path.name,
        }
        metrics.update(_flatten_metrics(raw_metrics))

        # Override sample_size from metadata if present
        md = raw.get("metadata", {})
        metrics["sample_size"] = md.get("sample_size", metrics.get("total_samples"))

        # Samples
        results = raw.get("results", [])
        if not results:
            print(f"[loader] WARNING: no results in {meta.path.name}")
            return metrics, pd.DataFrame()

        rows = []
        for entry in results:
            try:
                rows.append(_row_from_result(entry, meta))
            except Exception:
                pass  # skip malformed entries silently

        df = pd.DataFrame(rows)
        return metrics, df

    except Exception:
        print(f"[loader] ERROR processing {meta.path.name}:\n{traceback.format_exc()}")
        return None, None


# ── registry ──────────────────────────────────────────────────────────────────

@dataclass
class Registry:
    """
    Central store for all loaded data.

    keys: (experiment, model, language)  → (metrics_dict, samples_df)
    """
    _data: dict[tuple[int, str, str], tuple[dict, pd.DataFrame]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def add(self, meta: ResultFileMeta, metrics: dict, df: pd.DataFrame) -> None:
        key = (meta.experiment, meta.model, meta.language)
        if key in self._data:
            # Keep the one with more samples (latest / most complete run)
            existing_df = self._data[key][1]
            if len(df) > len(existing_df):
                self._data[key] = (metrics, df)
                print(f"[loader] Replaced duplicate ({meta.experiment}, {meta.model}, {meta.language}) — kept larger file")
            else:
                print(f"[loader] Skipped duplicate ({meta.experiment}, {meta.model}, {meta.language}) — kept existing")
        else:
            self._data[key] = (metrics, df)

    def get(self, exp: int, model: str, lang: str) -> tuple[dict, pd.DataFrame] | None:
        return self._data.get((exp, model, lang))

    def get_df(self, exp: int, model: str, lang: str) -> pd.DataFrame | None:
        entry = self._data.get((exp, model, lang))
        return entry[1] if entry else None

    def get_metrics(self, exp: int, model: str, lang: str) -> dict | None:
        entry = self._data.get((exp, model, lang))
        return entry[0] if entry else None

    def keys(self):
        return self._data.keys()

    def all_models(self) -> list[str]:
        return sorted({k[1] for k in self._data})

    def all_languages(self) -> list[str]:
        return sorted({k[2] for k in self._data})

    def all_experiments(self) -> list[int]:
        return sorted({k[0] for k in self._data})

    def models_for(self, exp: int, lang: str | None = None) -> list[str]:
        return sorted({
            k[1] for k in self._data
            if k[0] == exp and (lang is None or k[2] == lang)
        })

    def languages_for(self, exp: int, model: str | None = None) -> list[str]:
        return sorted({
            k[2] for k in self._data
            if k[0] == exp and (model is None or k[1] == model)
        })

    # ── DataFrame accessors ──────────────────────────────────────────────────

    def all_samples(self, experiment: int | None = None) -> pd.DataFrame:
        """Concatenate all sample DataFrames, optionally filtering by experiment."""
        frames = []
        for (exp, model, lang), (_, df) in self._data.items():
            if experiment is not None and exp != experiment:
                continue
            if df is not None and not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def master_summary(self) -> pd.DataFrame:
        """One row per (exp, model, lang) with all metric columns."""
        rows = []
        for (exp, model, lang), (metrics, df) in self._data.items():
            row = dict(metrics)
            row["n_samples_loaded"] = len(df) if df is not None else 0
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Reorder columns nicely
        front = ["experiment", "model", "model_type", "language",
                 "accuracy", "f1_score", "precision", "recall",
                 "tp", "fp", "tn", "fn",
                 "agreement_rate", "total_samples", "total_classified",
                 "unclear_responses", "error_responses", "n_samples_loaded",
                 "timestamp", "file"]
        existing = [c for c in front if c in df.columns]
        rest = [c for c in df.columns if c not in existing]
        return df[existing + rest].sort_values(["experiment", "model", "language"])


def build_registry(files: list[ResultFileMeta]) -> Registry:
    """Load all discovered files into a Registry. Skips failures gracefully."""
    registry = Registry()
    total = len(files)
    for i, meta in enumerate(files, 1):
        print(f"[loader] Loading {i}/{total}: {meta.path.name} ...", end=" ")
        metrics, df = load_file(meta)
        if metrics is None or df is None:
            print("FAILED")
            registry.skipped.append(str(meta.path))
        else:
            registry.add(meta, metrics, df)
            n = len(df) if df is not None else 0
            print(f"OK ({n} rows)")

    if registry.skipped:
        print(f"\n[loader] {len(registry.skipped)} file(s) failed to load:")
        for p in registry.skipped:
            print(f"  ✗ {p}")

    return registry
