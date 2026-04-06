"""
plots.py — All visualization functions for the multilingual mental health pipeline.

Each function:
  • Accepts a DataFrame or Registry and an output Path
  • Saves a PNG and returns the saved path (or None if skipped)
  • Prints a log line when skipped, explaining why

Call generate_all_plots(registry, merges, out_dir) to run everything at once.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from loader import Registry

# ── style ─────────────────────────────────────────────────────────────────────

_PALETTE = {
    "gemini": "#4285F4",
    "openai": "#10A37F",
    "claude": "#D97706",
    "llama": "#7C3AED",
    "deepseek": "#DB2777",
    "gemma": "#059669",
}
_LANG_COLORS = {
    "arabic": "#EF4444",
    "urdu": "#F59E0B",
    "chinese": "#3B82F6",
}
_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
}
_FIGSIZE_DEFAULT = (12, 6)
_FIGSIZE_WIDE = (16, 7)
_FIGSIZE_TALL = (10, 10)

def _model_color(model: str) -> str:
    return _PALETTE.get(model, "#6B7280")

def _lang_color(lang: str) -> str:
    return _LANG_COLORS.get(lang, "#6B7280")

def _save(fig: plt.Figure, path: Path, title: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {path.relative_to(path.parent.parent.parent)}")
    return path

def _skip(reason: str, name: str) -> None:
    print(f"  [plot] Skipped '{name}': {reason}")


# ── Experiment 1 plots ────────────────────────────────────────────────────────

def plot_exp1_metrics_by_model(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Bar chart: all metrics for each model, grouped by metric."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty:
        _skip("no Exp1 data", "exp1_metrics_by_model"); return None

    metrics = ["accuracy", "f1_score", "precision", "recall"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        _skip("no metric columns", "exp1_metrics_by_model"); return None

    # Average across languages per model
    agg = df.groupby("model")[available].mean().reset_index()
    models = agg["model"].tolist()
    x = np.arange(len(available))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)
    for i, row in agg.iterrows():
        offset = (list(agg["model"]).index(row["model"]) - len(models) / 2 + 0.5) * width
        values = [row.get(m, 0) for m in available]
        bars = ax.bar(x + offset, values, width * 0.9,
                      label=row["model"],
                      color=_model_color(row["model"]))

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in available], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1 — Average metrics per model (across all languages)")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp1" / "metrics_by_model.png")


def plot_exp1_metrics_by_language(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Bar chart: all metrics for each language, grouped by metric."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty:
        _skip("no Exp1 data", "exp1_metrics_by_language"); return None

    metrics = ["accuracy", "f1_score", "precision", "recall"]
    available = [m for m in metrics if m in df.columns]
    agg = df.groupby("language")[available].mean().reset_index()
    langs = agg["language"].tolist()
    x = np.arange(len(available))
    width = 0.8 / max(len(langs), 1)

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)
    for _, row in agg.iterrows():
        lang = row["language"]
        offset = (langs.index(lang) - len(langs) / 2 + 0.5) * width
        values = [row.get(m, 0) for m in available]
        ax.bar(x + offset, values, width * 0.9,
               label=lang, color=_lang_color(lang))

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in available], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1 — Average metrics per language (across all models)")
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp1" / "metrics_by_language.png")


def plot_exp1_online_vs_local(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Side-by-side comparison of online vs local model performance."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty or "model_type" not in df.columns:
        _skip("no Exp1 data", "exp1_online_vs_local"); return None
    if df["model_type"].nunique() < 2:
        _skip("only one model type present", "exp1_online_vs_local"); return None

    metrics = ["accuracy", "f1_score", "precision", "recall"]
    available = [m for m in metrics if m in df.columns]
    agg = df.groupby(["model_type", "language"])[available].mean().reset_index()

    langs = sorted(agg["language"].unique())
    types = ["online", "local"]
    x = np.arange(len(langs))
    width = 0.35

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5), sharey=True)
    if len(available) == 1:
        axes = [axes]

    type_colors = {"online": "#3B82F6", "local": "#F59E0B"}

    for ax, metric in zip(axes, available):
        for j, mtype in enumerate(types):
            sub = agg[agg["model_type"] == mtype].set_index("language")
            vals = [sub.loc[l, metric] if l in sub.index else 0 for l in langs]
            ax.bar(x + j * width, vals, width * 0.9,
                   label=mtype.capitalize(), color=type_colors.get(mtype, "gray"))
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([l.capitalize() for l in langs], fontsize=10)
        ax.set_title(_METRIC_LABELS.get(metric, metric))
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Experiment 1 — Online vs Local models by language", fontsize=13)
    return _save(fig, out_dir / "exp1" / "online_vs_local.png")


def plot_exp1_f1_heatmap(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Heatmap: F1 score — models × languages."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty or "f1_score" not in df.columns:
        _skip("no f1_score data", "exp1_f1_heatmap"); return None
    if df["model"].nunique() < 2 or df["language"].nunique() < 2:
        _skip("need ≥2 models and ≥2 languages", "exp1_f1_heatmap"); return None

    pivot = df.pivot_table(index="model", columns="language", values="f1_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.9 + 1)))
    cmap = "YlOrRd" if not HAS_SEABORN else "YlOrRd"
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1 Score")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.capitalize() for m in pivot.index], fontsize=11)
    ax.set_title("Experiment 1 — F1 Score heatmap (model × language)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 0.8 else "white")
    return _save(fig, out_dir / "exp1" / "f1_heatmap.png")


def plot_exp1_accuracy_heatmap(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Heatmap: Accuracy — models × languages."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty or "accuracy" not in df.columns:
        _skip("no accuracy data", "exp1_accuracy_heatmap"); return None

    pivot = df.pivot_table(index="model", columns="language", values="accuracy", aggfunc="mean")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        _skip("insufficient data", "exp1_accuracy_heatmap"); return None

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.9 + 1)))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.capitalize() for m in pivot.index], fontsize=11)
    ax.set_title("Experiment 1 — Accuracy heatmap (model × language)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 0.8 else "white")
    return _save(fig, out_dir / "exp1" / "accuracy_heatmap.png")


def plot_exp1_model_ranking(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Horizontal bar chart: models ranked by avg F1 across languages."""
    df = summary[summary["experiment"] == 1].copy()
    if df.empty or "f1_score" not in df.columns:
        _skip("no data", "exp1_model_ranking"); return None

    ranked = df.groupby("model")["f1_score"].mean().sort_values(ascending=True)
    if len(ranked) < 2:
        _skip("need ≥2 models", "exp1_model_ranking"); return None

    fig, ax = plt.subplots(figsize=(9, max(4, len(ranked) * 0.7 + 1)))
    colors = [_model_color(m) for m in ranked.index]
    bars = ax.barh(ranked.index, ranked.values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Average F1 Score (across all languages)")
    ax.set_title("Experiment 1 — Model ranking by F1 Score")
    for bar, val in zip(bars, ranked.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_dir / "exp1" / "model_ranking.png")


def plot_exp1_per_language_comparison(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One grouped bar chart per language showing all models."""
    df = summary[summary["experiment"] == 1].copy()
    saved = []
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        if sub.empty:
            continue
        metrics = ["accuracy", "f1_score", "precision", "recall"]
        available = [m for m in metrics if m in sub.columns]
        models = sorted(sub["model"].unique())
        x = np.arange(len(available))
        width = 0.8 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)
        for _, row in sub.iterrows():
            model = row["model"]
            offset = (models.index(model) - len(models) / 2 + 0.5) * width
            values = [row.get(m, 0) for m in available]
            ax.bar(x + offset, values, width * 0.9,
                   label=model, color=_model_color(model))

        ax.set_xticks(x)
        ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in available], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("Score")
        ax.set_title(f"Experiment 1 — All models on {lang.capitalize()}")
        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        p = _save(fig, out_dir / "exp1" / f"per_language_{lang}.png")
        if p:
            saved.append(p)
    return saved


def plot_exp1_confusion_summary(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Stacked bar: TP/FP/TN/FN proportions per model (averaged across languages)."""
    df = summary[summary["experiment"] == 1].copy()
    needed = ["tp", "fp", "tn", "fn"]
    if df.empty or not all(c in df.columns for c in needed):
        _skip("missing confusion matrix columns", "exp1_confusion_summary"); return None

    agg = df.groupby("model")[needed].mean().reset_index()
    agg["total"] = agg[needed].sum(axis=1)
    for c in needed:
        agg[c + "_pct"] = agg[c] / agg["total"]

    models = agg["model"].tolist()
    x = np.arange(len(models))
    colors = {"tp": "#22C55E", "fp": "#F97316", "tn": "#3B82F6", "fn": "#EF4444"}
    labels = {"tp": "True Positives", "fp": "False Positives",
               "tn": "True Negatives", "fn": "False Negatives"}

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(models))
    for c in needed:
        vals = agg[c + "_pct"].values
        ax.bar(x, vals, bottom=bottoms, label=labels[c], color=colors[c], edgecolor="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in models], fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Proportion")
    ax.set_title("Experiment 1 — Confusion matrix breakdown per model (avg across languages)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp1" / "confusion_summary.png")


# ── Experiment 2 plots ────────────────────────────────────────────────────────

def plot_exp2_keyword_count_dist(registry: Registry, out_dir: Path) -> Path | None:
    """Box plot of keyword count distribution per model and language."""
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keyword_count" not in df2.columns:
        _skip("no Exp2 keyword data", "exp2_keyword_count_dist"); return None

    combos = sorted(df2.groupby(["model", "language"]).groups.keys())
    if len(combos) < 2:
        _skip("need ≥2 combos", "exp2_keyword_count_dist"); return None

    data = [df2[(df2["model"] == m) & (df2["language"] == l)]["keyword_count"].values
            for m, l in combos]
    labels = [f"{m}\n{l}" for m, l in combos]

    fig, ax = plt.subplots(figsize=(max(12, len(combos) * 1.5), 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, (m, _) in zip(bp["boxes"], combos):
        patch.set_facecolor(_model_color(m))
        patch.set_alpha(0.7)

    ax.set_ylabel("Keywords extracted per entry")
    ax.set_title("Experiment 2 — Keyword count distribution by model & language")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp2" / "keyword_count_distribution.png")


def plot_exp2_keyword_stats_heatmap(kw_summary: pd.DataFrame, out_dir: Path) -> Path | None:
    """Heatmap: avg keyword count per entry — models × languages."""
    if kw_summary.empty or "avg_keyword_count" not in kw_summary.columns:
        _skip("no keyword summary", "exp2_keyword_stats_heatmap"); return None

    pivot = kw_summary.pivot_table(
        index="model", columns="language", values="avg_keyword_count"
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        _skip("insufficient data", "exp2_keyword_stats_heatmap"); return None

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.9 + 1)))
    im = ax.imshow(pivot.values, cmap="Greens", aspect="auto")
    plt.colorbar(im, ax=ax, label="Avg keywords per entry")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.capitalize() for m in pivot.index], fontsize=11)
    ax.set_title("Experiment 2 — Average keywords per entry (model × language)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10)
    return _save(fig, out_dir / "exp2" / "keyword_stats_heatmap.png")


def plot_exp2_top_keywords(top_kws: pd.DataFrame, out_dir: Path) -> Path | None:
    """Horizontal bar chart of most frequent translated keywords globally."""
    if top_kws.empty or "keyword" not in top_kws.columns:
        _skip("no top keywords data", "exp2_top_keywords"); return None

    top = top_kws.head(25).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4 + 1)))
    ax.barh(top["keyword"], top["count"], color="#10B981")
    ax.set_xlabel("Frequency")
    ax.set_title("Experiment 2 — Top 25 translated keywords (all models & languages)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_dir / "exp2" / "top_keywords_global.png")


def plot_exp2_keywords_by_class(registry: Registry, out_dir: Path) -> Path | None:
    """
    Side-by-side bar charts: top keywords for depressed vs not-depressed entries.
    """
    from merger import top_keywords_table
    top_kws = top_keywords_table(registry, n=40, by_class=True)
    if top_kws.empty or "ground_truth" not in top_kws.columns:
        _skip("no class-level keyword data", "exp2_keywords_by_class"); return None

    classes = top_kws["ground_truth"].unique()
    if len(classes) < 2:
        _skip("need both depressed/not-depressed classes", "exp2_keywords_by_class"); return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    class_colors = {"depressed": "#EF4444", "not depressed": "#3B82F6"}

    for ax, cls in zip(axes, ["depressed", "not depressed"]):
        sub = top_kws[top_kws["ground_truth"] == cls].head(20).sort_values("count", ascending=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.barh(sub["keyword"], sub["count"],
                color=class_colors.get(cls, "gray"), alpha=0.85)
        ax.set_title(f"Top keywords — {cls.title()}")
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Experiment 2 — Top translated keywords by ground truth class")
    return _save(fig, out_dir / "exp2" / "top_keywords_by_class.png")


def plot_exp2_keywords_correct_vs_incorrect(
    merged_12: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Compare keyword counts for correct vs incorrect predictions."""
    if merged_12.empty:
        _skip("no Exp1+2 merged data", "exp2_keywords_correct_vs_incorrect"); return None

    # Find keyword_count column (may be suffixed)
    kc_col = next(
        (c for c in merged_12.columns if "keyword_count" in c and "exp2" in c),
        next((c for c in merged_12.columns if "keyword_count" in c), None)
    )
    correct_col = next(
        (c for c in merged_12.columns if c.startswith("correct")), None
    )
    if kc_col is None or correct_col is None:
        _skip("missing keyword_count or correct column", "exp2_keywords_correct_vs_incorrect"); return None

    correct_counts = merged_12[merged_12[correct_col] == True][kc_col].dropna()
    incorrect_counts = merged_12[merged_12[correct_col] == False][kc_col].dropna()
    if correct_counts.empty or incorrect_counts.empty:
        _skip("not enough data in both correct/incorrect", "exp2_keywords_correct_vs_incorrect"); return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([correct_counts.values, incorrect_counts.values],
                    labels=["Correct (Exp1)", "Incorrect (Exp1)"],
                    patch_artist=True, showfliers=False)
    box_colors = ["#22C55E", "#EF4444"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Keyword count (Exp2)")
    ax.set_title("Experiment 2 — Keyword count: Correct vs Incorrect predictions")
    ax.grid(axis="y", alpha=0.3)

    # Add mean annotations
    for i, (label, data) in enumerate(
        [("Correct", correct_counts), ("Incorrect", incorrect_counts)], 1
    ):
        ax.text(i, data.mean() + 0.05, f"mean={data.mean():.2f}",
                ha="center", va="bottom", fontsize=9, color="gray")

    return _save(fig, out_dir / "exp2" / "keywords_correct_vs_incorrect.png")


# ── Experiment 3 plots ────────────────────────────────────────────────────────

def plot_exp3_agreement_rates(
    agreement_summary: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Grouped bar chart: agreement rate per model and language."""
    if agreement_summary.empty or "agreement_rate" not in agreement_summary.columns:
        _skip("no Exp3 agreement data", "exp3_agreement_rates"); return None

    df = agreement_summary.dropna(subset=["agreement_rate"])
    if df.empty:
        _skip("all agreement_rate values are null", "exp3_agreement_rates"); return None

    langs = sorted(df["language"].unique())
    models = sorted(df["model"].unique())
    x = np.arange(len(langs))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("language")
        vals = [sub.loc[l, "agreement_rate"] if l in sub.index else 0 for l in langs]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=model.capitalize(),
               color=_model_color(model))

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in langs], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Experiment 3 — Agreement rate (original vs translated classification)")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp3" / "agreement_rates.png")


def plot_exp3_agreement_breakdown(
    agreement_summary: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Stacked bar: agree / disagree / no_translation counts per model+language."""
    if agreement_summary.empty:
        _skip("no Exp3 agreement data", "exp3_agreement_breakdown"); return None

    needed = ["agree", "disagree", "no_translation"]
    if not all(c in agreement_summary.columns for c in needed):
        _skip("missing agreement breakdown columns", "exp3_agreement_breakdown"); return None

    df = agreement_summary.copy()
    df["label"] = df["model"] + "\n" + df["language"]
    df = df.sort_values(["model", "language"])
    x = np.arange(len(df))

    colors = {"agree": "#22C55E", "disagree": "#EF4444", "no_translation": "#9CA3AF"}
    labels = {"agree": "Agree", "disagree": "Disagree", "no_translation": "No translation"}

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.5), 6))
    bottoms = np.zeros(len(df))
    for col in needed:
        vals = df[col].fillna(0).values
        ax.bar(x, vals, bottom=bottoms, label=labels[col], color=colors[col])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), fontsize=9)
    ax.set_ylabel("Sample count")
    ax.set_title("Experiment 3 — Agreement breakdown per model & language")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "exp3" / "agreement_breakdown.png")


def plot_exp3_agreement_heatmap(
    agreement_summary: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Heatmap: agreement rate — models × languages."""
    if agreement_summary.empty or "agreement_rate" not in agreement_summary.columns:
        _skip("no data", "exp3_agreement_heatmap"); return None

    pivot = agreement_summary.pivot_table(
        index="model", columns="language", values="agreement_rate"
    )
    if pivot.shape[0] < 1 or pivot.shape[1] < 1:
        _skip("insufficient data", "exp3_agreement_heatmap"); return None

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.9 + 1)))
    im = ax.imshow(pivot.values, cmap="Greens", vmin=0.7, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Agreement Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.capitalize() for m in pivot.index], fontsize=11)
    ax.set_title("Experiment 3 — Agreement rate heatmap (model × language)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, color="black" if val < 0.9 else "white")
    return _save(fig, out_dir / "exp3" / "agreement_heatmap.png")


# ── Cross-experiment plots ────────────────────────────────────────────────────

def plot_cross_f1_vs_agreement(
    summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    """Scatter: Exp1 F1 vs Exp3 agreement rate for models that have both."""
    if summary.empty or agreement_summary.empty:
        _skip("missing Exp1 or Exp3 data", "cross_f1_vs_agreement"); return None

    exp1 = summary[summary["experiment"] == 1][["model", "language", "f1_score"]].copy()
    merged = exp1.merge(
        agreement_summary[["model", "language", "agreement_rate"]],
        on=["model", "language"], how="inner"
    ).dropna()

    if len(merged) < 3:
        _skip(f"only {len(merged)} overlapping points", "cross_f1_vs_agreement"); return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in merged.iterrows():
        ax.scatter(row["f1_score"], row["agreement_rate"],
                   color=_model_color(row["model"]),
                   s=120, zorder=3)
        ax.annotate(
            f"{row['model']}\n({row['language']})",
            (row["f1_score"], row["agreement_rate"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("Exp1 F1 Score")
    ax.set_ylabel("Exp3 Agreement Rate")
    ax.set_title("Cross-experiment — Exp1 F1 vs Exp3 Agreement Rate")
    ax.grid(alpha=0.3)

    # Correlation annotation
    corr = merged["f1_score"].corr(merged["agreement_rate"])
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Add legend for models
    seen = set()
    for _, row in merged.iterrows():
        if row["model"] not in seen:
            ax.scatter([], [], color=_model_color(row["model"]),
                       label=row["model"].capitalize(), s=80)
            seen.add(row["model"])
    ax.legend(title="Model", loc="lower right")

    return _save(fig, out_dir / "cross" / "f1_vs_agreement.png")


def plot_cross_exp1_vs_exp2_accuracy(
    summary: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Side-by-side: Exp1 vs Exp2 accuracy per model (should be identical but good to verify)."""
    df1 = summary[summary["experiment"] == 1][["model", "language", "accuracy"]].rename(
        columns={"accuracy": "accuracy_exp1"}
    )
    df2 = summary[summary["experiment"] == 2][["model", "language", "accuracy"]].rename(
        columns={"accuracy": "accuracy_exp2"}
    )
    if df1.empty or df2.empty:
        _skip("missing Exp1 or Exp2 data", "cross_exp1_vs_exp2_accuracy"); return None

    merged = df1.merge(df2, on=["model", "language"], how="inner").dropna()
    if merged.empty:
        _skip("no overlapping data", "cross_exp1_vs_exp2_accuracy"); return None

    merged["label"] = merged["model"] + " / " + merged["language"]
    merged = merged.sort_values("accuracy_exp1", ascending=False)
    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(merged) * 1.4), 5))
    ax.bar(x - width / 2, merged["accuracy_exp1"], width, label="Exp 1", color="#3B82F6", alpha=0.85)
    ax.bar(x + width / 2, merged["accuracy_exp2"], width, label="Exp 2", color="#10B981", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"].tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-experiment — Exp1 vs Exp2 accuracy (same base classification)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "cross" / "exp1_vs_exp2_accuracy.png")


def plot_cross_exp1_exp3_label_flip(
    merged_13: pd.DataFrame, out_dir: Path
) -> Path | None:
    """
    For entries where Exp1 is wrong, what fraction 'flip' in Exp3?
    Stacked bars per model-language.
    """
    if merged_13.empty:
        _skip("no Exp1+3 merged data", "cross_exp1_exp3_label_flip"); return None

    correct_col = next((c for c in merged_13.columns if c.startswith("correct_")), None)
    agreement_col = next((c for c in merged_13.columns if "agreement_exp3" in c or c == "agreement_exp3"), None)

    if correct_col is None or agreement_col is None:
        _skip("missing correct or agreement columns", "cross_exp1_exp3_label_flip"); return None

    rows = []
    for (model, lang), grp in merged_13.groupby(["model", "language"]):
        incorrect = grp[grp[correct_col] == False]
        if incorrect.empty:
            continue
        flipped = (incorrect[agreement_col] == "no").sum()
        stayed_wrong = (incorrect[agreement_col] == "yes").sum()
        rows.append({
            "label": f"{model}\n{lang}",
            "model": model, "language": lang,
            "flipped": int(flipped),
            "stayed_wrong": int(stayed_wrong),
            "total_incorrect": len(incorrect),
        })

    if not rows:
        _skip("no incorrect-prediction rows found", "cross_exp1_exp3_label_flip"); return None

    df = pd.DataFrame(rows)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.5), 5))
    ax.bar(x, df["flipped"], label="Flipped in Exp3 (disagree)", color="#22C55E")
    ax.bar(x, df["stayed_wrong"], bottom=df["flipped"],
           label="Still wrong in Exp3 (agree)", color="#EF4444", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), fontsize=9)
    ax.set_ylabel("Count (among Exp1 incorrect predictions)")
    ax.set_title("Cross-experiment — Exp1 wrong predictions: flip behavior in Exp3")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "cross" / "exp1_incorrect_flip_in_exp3.png")


def plot_cross_keyword_agreement_connection(
    merged_23: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Box plot: keyword count (Exp2) split by agreement label (Exp3)."""
    if merged_23.empty:
        _skip("no Exp2+3 merged data", "cross_keyword_agreement_connection"); return None

    kc_col = next((c for c in merged_23.columns if "keyword_count" in c and "exp2" in c), None)
    ag_col = next((c for c in merged_23.columns if "agreement_exp3" in c or c == "agreement_exp3"), None)

    if kc_col is None or ag_col is None:
        _skip("missing keyword_count or agreement columns", "cross_keyword_agreement_connection"); return None

    sub = merged_23[[kc_col, ag_col]].dropna()
    groups = {}
    for val in ["yes", "no"]:
        g = sub[sub[ag_col] == val][kc_col].values
        if len(g) > 0:
            groups[val] = g

    if len(groups) < 2:
        _skip("need both agree/disagree groups", "cross_keyword_agreement_connection"); return None

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(list(groups.values()), labels=["Agree", "Disagree"],
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#22C55E", "#EF4444"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Keyword count (Exp2)")
    ax.set_title("Cross-experiment — Keyword count vs Exp3 agreement")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "cross" / "keyword_count_vs_agreement.png")


# ── main entry point ──────────────────────────────────────────────────────────

def generate_all_plots(
    registry: Registry,
    summary: pd.DataFrame,
    merges: dict[str, pd.DataFrame],
    kw_summary: pd.DataFrame,
    top_kws: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """
    Run all plot functions. Returns list of successfully saved plot paths.
    Skips gracefully and logs when data is missing.
    """
    print("\n[plots] Generating visualizations...")
    saved: list[Path] = []

    def _try(fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, list):
                saved.extend([p for p in result if p])
            elif result:
                saved.append(result)
        except Exception as e:
            import traceback
            print(f"  [plot] ERROR in {fn.__name__}: {e}")
            traceback.print_exc()

    # Experiment 1
    _try(plot_exp1_metrics_by_model, summary, out_dir)
    _try(plot_exp1_metrics_by_language, summary, out_dir)
    _try(plot_exp1_online_vs_local, summary, out_dir)
    _try(plot_exp1_f1_heatmap, summary, out_dir)
    _try(plot_exp1_accuracy_heatmap, summary, out_dir)
    _try(plot_exp1_model_ranking, summary, out_dir)
    _try(plot_exp1_per_language_comparison, summary, out_dir)
    _try(plot_exp1_confusion_summary, summary, out_dir)

    # Experiment 2
    _try(plot_exp2_keyword_count_dist, registry, out_dir)
    _try(plot_exp2_keyword_stats_heatmap, kw_summary, out_dir)
    _try(plot_exp2_top_keywords, top_kws, out_dir)
    _try(plot_exp2_keywords_by_class, registry, out_dir)
    _try(plot_exp2_keywords_correct_vs_incorrect, merges.get("exp1_exp2", pd.DataFrame()), out_dir)

    # Experiment 3
    _try(plot_exp3_agreement_rates, agreement_summary, out_dir)
    _try(plot_exp3_agreement_breakdown, agreement_summary, out_dir)
    _try(plot_exp3_agreement_heatmap, agreement_summary, out_dir)

    # Cross-experiment
    _try(plot_cross_f1_vs_agreement, summary, agreement_summary, out_dir)
    _try(plot_cross_exp1_vs_exp2_accuracy, summary, out_dir)
    _try(plot_cross_exp1_exp3_label_flip, merges.get("exp1_exp3", pd.DataFrame()), out_dir)
    _try(plot_cross_keyword_agreement_connection, merges.get("exp2_exp3", pd.DataFrame()), out_dir)

    print(f"\n[plots] {len(saved)} plot(s) saved.\n")
    return saved
