"""
main.py — Orchestration script for the multilingual mental health analysis pipeline.

Usage:
    python data_visualization/main.py
    python data_visualization/main.py --results ../results
    python data_visualization/main.py --no-plots
    python data_visualization/main.py --output ./my_output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root OR from inside data_visualization/
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))  # so imports from this package work

from discovery import discover, print_discovery_report
from loader import build_registry
from merger import (
    build_all_merges,
    keyword_summary,
    top_keywords_table,
    exp3_agreement_summary,
)
from plots import generate_all_plots

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data_visualization" / "outputs"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    if df is None or df.empty:
        print(f"[tables] Skipped '{label}': empty DataFrame")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lists can't be saved to CSV cleanly — convert to pipe-joined strings
    for col in df.columns:
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, list):
                    df = df.copy()
                    df[col] = df[col].apply(
                        lambda x: " | ".join(str(i) for i in x) if isinstance(x, list) else x
                    )
            except Exception:
                pass
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[tables] Saved: {path.relative_to(_PROJECT_ROOT)}")


def save_all_tables(
    registry,
    summary: pd.DataFrame,
    merges: dict[str, pd.DataFrame],
    kw_sum: pd.DataFrame,
    top_kws: pd.DataFrame,
    agreement_sum: pd.DataFrame,
    out_dir: Path,
) -> None:
    tables_dir = out_dir / "tables"
    print("\n[tables] Saving CSVs...")

    _save_csv(summary, tables_dir / "master_summary.csv", "master_summary")

    for exp in [1, 2, 3]:
        df = registry.all_samples(experiment=exp)
        _save_csv(df, tables_dir / f"samples_exp{exp}.csv", f"samples_exp{exp}")

    for name, df in merges.items():
        _save_csv(df, tables_dir / f"merged_{name}.csv", f"merged_{name}")

    _save_csv(kw_sum, tables_dir / "keyword_summary.csv", "keyword_summary")
    _save_csv(top_kws, tables_dir / "top_keywords.csv", "top_keywords")
    _save_csv(agreement_sum, tables_dir / "exp3_agreement_summary.csv", "exp3_agreement_summary")

    # Ranking table: models ranked by F1 per language
    exp1 = summary[summary["experiment"] == 1].copy()
    if not exp1.empty and "f1_score" in exp1.columns:
        ranking = (
            exp1.groupby(["language", "model"])[["f1_score", "accuracy", "precision", "recall"]]
            .mean()
            .reset_index()
            .sort_values(["language", "f1_score"], ascending=[True, False])
        )
        _save_csv(ranking, tables_dir / "exp1_model_ranking_by_language.csv", "ranking")

    # Per-sample case-level merged (exp1+2+3 where all exist, most useful for drill-down)
    all3 = merges.get("all_three", pd.DataFrame())
    _save_csv(all3, tables_dir / "case_level_merged_all_three.csv", "case_level_all_three")


# ── terminal report ───────────────────────────────────────────────────────────

def print_final_report(
    registry,
    summary: pd.DataFrame,
    merges: dict[str, pd.DataFrame],
    agreement_sum: pd.DataFrame,
    n_plots: int,
    elapsed: float,
    out_dir: Path,
) -> None:
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY REPORT")
    print("=" * 65)

    # What was loaded
    print(f"\n  Files loaded: {len(list(registry.keys()))}")
    print(f"  Experiments  : {registry.all_experiments()}")
    print(f"  Models       : {registry.all_models()}")
    print(f"  Languages    : {registry.all_languages()}")

    # Exp 1 performance highlights
    exp1 = summary[summary["experiment"] == 1].copy() if not summary.empty else pd.DataFrame()
    if not exp1.empty and "f1_score" in exp1.columns:
        best = exp1.loc[exp1["f1_score"].idxmax()]
        worst = exp1.loc[exp1["f1_score"].idxmin()]
        avg_f1 = exp1["f1_score"].mean()
        print(f"\n  Experiment 1 highlights:")
        print(f"    Avg F1 across all models/languages : {avg_f1:.3f}")
        print(f"    Best  : {best['model']:10} on {best['language']:8} (F1={best['f1_score']:.3f})")
        print(f"    Worst : {worst['model']:10} on {worst['language']:8} (F1={worst['f1_score']:.3f})")

    # Exp 3 highlights
    if not agreement_sum.empty and "agreement_rate" in agreement_sum.columns:
        ar = agreement_sum["agreement_rate"].dropna()
        if not ar.empty:
            print(f"\n  Experiment 3 highlights:")
            print(f"    Avg agreement rate : {ar.mean():.3f}")
            best_ag = agreement_sum.loc[agreement_sum["agreement_rate"].idxmax()]
            print(f"    Best  : {best_ag['model']} on {best_ag['language']} "
                  f"({best_ag['agreement_rate']:.3f})")

    # Merges
    print("\n  Merged tables produced:")
    for name, df in merges.items():
        status = f"{len(df):,} rows" if not df.empty else "EMPTY (no overlap)"
        print(f"    {name:20}: {status}")

    # Skipped files
    if registry.skipped:
        print(f"\n  Failed to load: {len(registry.skipped)} file(s)")
        for p in registry.skipped:
            print(f"    ✗ {p}")

    print(f"\n  Plots saved  : {n_plots}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print("=" * 65 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multilingual mental health NLP — analysis and visualization pipeline"
    )
    p.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS_DIR,
        help=f"Path to the results directory (default: {DEFAULT_RESULTS_DIR})"
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Path to the output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (only produce CSV tables)"
    )
    p.add_argument(
        "--no-tables", action="store_true",
        help="Skip CSV table saving"
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  Multilingual Mental Health — Analysis Pipeline")
    print("=" * 65)
    print(f"  Results dir : {args.results}")
    print(f"  Output dir  : {args.output}")

    # ── 1. Discover ──────────────────────────────────────────────────────────
    print("\n[step 1/5] Discovering result files...")
    files = discover(args.results)
    print_discovery_report(files)

    if not files:
        print("No result files found. Check the --results path.")
        sys.exit(1)

    # ── 2. Load ──────────────────────────────────────────────────────────────
    print("[step 2/5] Loading and normalising files...")
    registry = build_registry(files)

    # ── 3. Build summary + merges ─────────────────────────────────────────────
    print("\n[step 3/5] Building summary tables and merges...")
    summary = registry.master_summary()
    merges = build_all_merges(registry)
    kw_sum = keyword_summary(registry)
    top_kws = top_keywords_table(registry, n=50)
    agreement_sum = exp3_agreement_summary(registry)

    # ── 4. Save tables ───────────────────────────────────────────────────────
    if not args.no_tables:
        print("\n[step 4/5] Saving CSV tables...")
        save_all_tables(
            registry, summary, merges, kw_sum, top_kws, agreement_sum,
            args.output
        )
    else:
        print("\n[step 4/5] Skipping tables (--no-tables).")

    # ── 5. Generate plots ────────────────────────────────────────────────────
    plots_out = args.output / "plots"
    if not args.no_plots:
        print("\n[step 5/5] Generating plots...")
        saved_plots = generate_all_plots(
            registry, summary, merges, kw_sum, top_kws, agreement_sum,
            plots_out,
        )
        n_plots = len(saved_plots)
    else:
        print("\n[step 5/5] Skipping plots (--no-plots).")
        n_plots = 0

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_final_report(registry, summary, merges, agreement_sum, n_plots, elapsed, args.output)


if __name__ == "__main__":
    main()
