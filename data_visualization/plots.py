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


# ── merged-DataFrame column name constants ────────────────────────────────────
# Centralised so all functions use the same strings and refactoring is safe.

_COL_CORRECT_EXP1      = "correct_exp1"
_COL_CORRECT_EXP2      = "correct_exp2"
_COL_AGREEMENT_EXP3    = "agreement_exp3"
_COL_KW_COUNT_EXP2     = "keyword_count_exp2"
_COL_KW_COUNT_EXP3     = "keyword_count_exp3_exp3"
_COL_TRANSLATIONS_EXP2 = "translations_exp2"
_COL_KEYWORDS_EXP3     = "keywords_exp3_exp3"

# ── shared micro-helpers ──────────────────────────────────────────────────────

def _exclude_gemma_chinese(df: pd.DataFrame) -> pd.DataFrame:
    """Drop gemma/chinese rows — that run returned constant 1 keyword (experiment error)."""
    return df[~((df["model"] == "gemma") & (df["language"] == "chinese"))].copy()


def _ensure_axes_list(axes) -> list:
    """Wrap a single Axes object in a list so subplot loops are always uniform."""
    return [axes] if not isinstance(axes, (list, np.ndarray)) else list(axes)


def _annotate_box_mean(ax, position: int, data, fmt: str = ".2f",
                       offset: float = 0.05, fontsize: int = 9) -> None:
    """Place a mean-value label above the box at `position` (1-indexed)."""
    if len(data):
        mean = np.mean(data)
        ax.text(position, mean + offset, f"{mean:{fmt}}",
                ha="center", va="bottom", fontsize=fontsize)


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
    """
    Box plot of keyword count distribution — one subplot per language so models
    are clearly visible within each language. Excludes gemma/chinese (experiment
    error: all entries returned exactly 1 keyword).
    """
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keyword_count" not in df2.columns:
        _skip("no Exp2 keyword data", "exp2_keyword_count_dist"); return None

    df2 = _exclude_gemma_chinese(df2)

    langs = sorted(df2["language"].unique())
    if not langs:
        _skip("no language data after filtering", "exp2_keyword_count_dist"); return None

    fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 6), sharey=False)
    axes = _ensure_axes_list(axes)

    for ax, lang in zip(axes, langs):
        sub = df2[df2["language"] == lang]
        models = sorted(sub["model"].unique())
        data = [sub[sub["model"] == m]["keyword_count"].values for m in models]

        bp = ax.boxplot(data, labels=[m.capitalize() for m in models],
                        patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(_model_color(m))
            patch.set_alpha(0.75)

        for i, (_, d) in enumerate(zip(models, data), 1):
            _annotate_box_mean(ax, i, d, fmt=".1f", fontsize=7)

        ax.set_title(f"{lang.capitalize()}", fontsize=12)
        ax.set_ylabel("Keywords per entry" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 2 — Keyword count distribution by model per language\n"
        "(gemma/chinese excluded: experiment error — all entries returned 1 keyword)",
        fontsize=11,
    )
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


def plot_exp2_keyword_count_by_predicted_class(
    registry: Registry, out_dir: Path
) -> Path | None:
    """
    Box plot comparing keyword counts extracted for depressed vs not-depressed
    predicted entries, per model. Shows whether models extract more/fewer keywords
    when predicting depression vs no depression.
    """
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keyword_count" not in df2.columns or "prediction" not in df2.columns:
        _skip("no Exp2 keyword/prediction data", "exp2_keyword_count_by_predicted_class"); return None

    df2 = _exclude_gemma_chinese(df2)
    df2 = df2[df2["prediction"].isin(["depressed", "not depressed"])].dropna(subset=["keyword_count"])

    models = sorted(df2["model"].unique())
    if not models:
        _skip("no data after filtering", "exp2_keyword_count_by_predicted_class"); return None

    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 5), sharey=True)
    axes = _ensure_axes_list(axes)

    classes = ["depressed", "not depressed"]
    colors = {"depressed": "#EF4444", "not depressed": "#3B82F6"}

    for ax, model in zip(axes, models):
        sub = df2[df2["model"] == model]
        data = [sub[sub["prediction"] == cls]["keyword_count"].values for cls in classes]
        bp = ax.boxplot(data, labels=["Dep.", "Not dep."],
                        patch_artist=True, showfliers=False)
        for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.75)
        for i, (_, d) in enumerate(zip(classes, data), 1):
            _annotate_box_mean(ax, i, d, fontsize=8)
        ax.set_title(model.capitalize(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Keywords per entry")
    fig.suptitle(
        "Experiment 2 — Keyword count by predicted class per model\n"
        "(Do models extract more/fewer keywords when predicting depression?)",
        fontsize=11,
    )
    return _save(fig, out_dir / "exp2" / "keyword_count_by_predicted_class.png")


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




def plot_cross_exp1_exp3_flip_analysis(
    merged_13: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """
    Detailed 4-category breakdown of how Exp3 agreement relates to Exp1 correctness.

    The 4 categories per sample:
      A) Exp1 correct   + Exp3 agrees     → "Correct & confirmed"       (best case)
      B) Exp1 correct   + Exp3 disagrees  → "Correct but overruled"     (harmful flip)
      C) Exp1 wrong     + Exp3 disagrees  → "Wrong, corrected by Exp3"  (beneficial flip)
      D) Exp1 wrong     + Exp3 agrees     → "Wrong & confirmed wrong"   (worst case)

    Produces two complementary plots:
      1. Stacked bar chart showing absolute counts of all 4 categories per model+language
      2. Rate comparison: beneficial-flip rate and harmful-flip rate per model
    """
    if merged_13.empty:
        _skip("no Exp1+3 merged data", "cross_exp1_exp3_flip_analysis"); return []

    correct_col = _COL_CORRECT_EXP1
    agreement_col = _COL_AGREEMENT_EXP3
    if correct_col not in merged_13.columns or agreement_col not in merged_13.columns:
        _skip("missing correct_exp1 or agreement_exp3 column", "cross_exp1_exp3_flip_analysis"); return []

    # ── build per-combo table ──────────────────────────────────────────────
    rows = []
    for (model, lang), grp in merged_13.groupby(["model", "language"]):
        classified = grp[grp[agreement_col].isin(["yes", "no"])]
        if classified.empty:
            continue

        correct_agree    = ((classified[correct_col] == True)  & (classified[agreement_col] == "yes")).sum()
        correct_disagree = ((classified[correct_col] == True)  & (classified[agreement_col] == "no")).sum()
        wrong_disagree   = ((classified[correct_col] == False) & (classified[agreement_col] == "no")).sum()
        wrong_agree      = ((classified[correct_col] == False) & (classified[agreement_col] == "yes")).sum()
        total = len(classified)

        rows.append({
            "label": f"{model.capitalize()}\n{lang.capitalize()}",
            "model": model, "language": lang,
            "correct_confirmed": int(correct_agree),
            "correct_overruled": int(correct_disagree),
            "wrong_corrected":   int(wrong_disagree),
            "wrong_confirmed":   int(wrong_agree),
            "total": total,
            "beneficial_flip_rate": wrong_disagree / total if total else 0,
            "harmful_flip_rate":    correct_disagree / total if total else 0,
            "agreement_on_correct": correct_agree / (correct_agree + correct_disagree) if (correct_agree + correct_disagree) else 0,
            "agreement_on_wrong":   wrong_agree  / (wrong_agree + wrong_disagree)   if (wrong_agree + wrong_disagree) else 0,
        })

    if not rows:
        _skip("could not build flip-analysis rows", "cross_exp1_exp3_flip_analysis"); return []

    df = pd.DataFrame(rows)
    saved = []

    # ── Plot 1: Stacked bar — all 4 categories ─────────────────────────────
    cat_cols = ["correct_confirmed", "correct_overruled", "wrong_corrected", "wrong_confirmed"]
    cat_labels = [
        "A: Correct & confirmed by Exp3",
        "B: Correct but overruled by Exp3 (harmful)",
        "C: Wrong, corrected by Exp3 (beneficial)",
        "D: Wrong & confirmed wrong by Exp3",
    ]
    cat_colors = ["#22C55E", "#F97316", "#3B82F6", "#EF4444"]

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(11, len(df) * 1.8), 6))
    bottoms = np.zeros(len(df))
    for col, label, color in zip(cat_cols, cat_labels, cat_colors):
        vals = df[col].values
        bars = ax.bar(x, vals, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
        # Label each segment if large enough
        for bar, bot, val in zip(bars, bottoms, vals):
            if val / df["total"].max() > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        str(val), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), fontsize=9)
    ax.set_ylabel("Sample count")
    ax.set_title(
        "Exp1 vs Exp3 — Flip analysis: did Exp3 agreement help or hurt?\n"
        "A=both correct  |  B=harmful flip  |  C=beneficial flip  |  D=both wrong"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    saved.append(_save(fig, out_dir / "cross" / "exp3_flip_analysis_stacked.png"))

    # ── Plot 2: Rate chart — beneficial vs harmful flip rates per model ────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: beneficial vs harmful flip rates
    ax_rates = axes2[0]
    x2 = np.arange(len(df))
    width = 0.35
    ax_rates.bar(x2 - width / 2, df["beneficial_flip_rate"] * 100,
                 width, label="Beneficial flip rate\n(wrong→corrected)",
                 color="#3B82F6", alpha=0.85)
    ax_rates.bar(x2 + width / 2, df["harmful_flip_rate"] * 100,
                 width, label="Harmful flip rate\n(correct→overruled)",
                 color="#F97316", alpha=0.85)
    ax_rates.set_xticks(x2)
    ax_rates.set_xticklabels(df["label"].tolist(), fontsize=8)
    ax_rates.set_ylabel("Rate (%)")
    ax_rates.set_title("Beneficial vs Harmful flip rate per model+language")
    ax_rates.legend(fontsize=8)
    ax_rates.grid(axis="y", alpha=0.3)

    # Right: agreement rate when Exp1 was correct vs when Exp1 was wrong
    ax_cond = axes2[1]
    ax_cond.bar(x2 - width / 2, df["agreement_on_correct"] * 100,
                width, label="Agreement rate\n(when Exp1 correct)",
                color="#22C55E", alpha=0.85)
    ax_cond.bar(x2 + width / 2, df["agreement_on_wrong"] * 100,
                width, label="Agreement rate\n(when Exp1 wrong)",
                color="#EF4444", alpha=0.85)
    ax_cond.set_xticks(x2)
    ax_cond.set_xticklabels(df["label"].tolist(), fontsize=8)
    ax_cond.set_ylabel("Rate (%)")
    ax_cond.set_title("Exp3 agreement rate conditioned on Exp1 correctness")
    ax_cond.legend(fontsize=8)
    ax_cond.grid(axis="y", alpha=0.3)

    fig2.suptitle("Exp3 flip behavior analysis — rates", fontsize=12)
    saved.append(_save(fig2, out_dir / "cross" / "exp3_flip_rates.png"))

    return saved


def plot_exp2_exp3_keyword_count_paired(
    merged_23: pd.DataFrame, out_dir: Path
) -> Path | None:
    """
    Side-by-side comparison of how many keywords each model extracts in Exp2
    (original-language → translated to English) vs Exp3 (direct from English
    translation). One subplot per model. Colored by agreement.
    """
    if merged_23.empty:
        _skip("no Exp2+3 merged data", "exp2_exp3_keyword_count_paired"); return None

    kc2 = _COL_KW_COUNT_EXP2
    kc3 = _COL_KW_COUNT_EXP3
    if kc2 not in merged_23.columns or kc3 not in merged_23.columns:
        _skip("missing keyword_count columns", "exp2_exp3_keyword_count_paired"); return None

    models = sorted(merged_23["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=False)
    axes = _ensure_axes_list(axes)

    for ax, model in zip(axes, models):
        sub = merged_23[merged_23["model"] == model]
        data_exp2 = sub[kc2].dropna().values
        data_exp3 = sub[kc3].dropna().values

        bp = ax.boxplot([data_exp2, data_exp3],
                        labels=["Exp2\n(translated)", "Exp3\n(direct)"],
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#F59E0B", "#3B82F6"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        for i, d in enumerate([data_exp2, data_exp3], 1):
            _annotate_box_mean(ax, i, d)

        ax.set_title(f"{model.capitalize()}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Keywords per entry")
    fig.suptitle(
        "Exp2 vs Exp3 — Keyword count comparison\n"
        "Exp2: original-language kws translated to English  |  Exp3: kws from English translation",
        fontsize=11,
    )
    return _save(fig, out_dir / "cross" / "exp2_exp3_keyword_count_paired.png")


def plot_exp2_exp3_keyword_overlap_and_top(
    merged_23: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """
    Two plots:
      1. Jaccard similarity distribution between Exp2 translated keywords and
         Exp3 English keywords, per model — shows conceptual overlap.
      2. Top-20 keyword comparison: Exp2 translated vs Exp3 direct, side by side
         (combined across all models/languages with data).
    """
    if merged_23.empty:
        _skip("no Exp2+3 merged data", "exp2_exp3_keyword_overlap"); return []

    kw2_col = _COL_TRANSLATIONS_EXP2
    kw3_col = _COL_KEYWORDS_EXP3
    if kw2_col not in merged_23.columns or kw3_col not in merged_23.columns:
        _skip("missing keyword list columns", "exp2_exp3_keyword_overlap"); return []

    def _jaccard(a: list, b: list) -> float | None:
        sa = {w.lower().strip() for w in a if w}
        sb = {w.lower().strip() for w in b if w}
        if not sa and not sb:
            return None
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0

    saved = []

    # Single pass: compute per-model Jaccard scores AND global keyword counters
    from collections import Counter
    models = sorted(merged_23["model"].unique())
    jaccard_data: dict[str, list[float]] = {m: [] for m in models}
    cnt2: Counter = Counter()
    cnt3: Counter = Counter()

    for model, kw2_val, kw3_val in zip(
        merged_23["model"], merged_23[kw2_col], merged_23[kw3_col]
    ):
        if isinstance(kw2_val, list):
            cnt2.update(w.lower().strip() for w in kw2_val if w and w.strip())
        if isinstance(kw3_val, list):
            cnt3.update(w.lower().strip() for w in kw3_val if w and w.strip())
        if isinstance(kw2_val, list) and isinstance(kw3_val, list):
            j = _jaccard(kw2_val, kw3_val)
            if j is not None:
                jaccard_data[model].append(j)

    jaccard_data = {m: scores for m, scores in jaccard_data.items() if scores}

    # ── Plot 1: Jaccard similarity box plot per model ─────────────────────
    if jaccard_data:
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(
            list(jaccard_data.values()),
            labels=[m.capitalize() for m in jaccard_data.keys()],
            patch_artist=True, showfliers=False,
        )
        for patch, model in zip(bp["boxes"], jaccard_data.keys()):
            patch.set_facecolor(_model_color(model))
            patch.set_alpha(0.75)
        for i, (_, scores) in enumerate(jaccard_data.items(), 1):
            _annotate_box_mean(ax, i, scores, fmt=".3f", offset=0.005)
        ax.set_ylabel("Jaccard similarity")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            "Exp2 vs Exp3 — Keyword overlap (Jaccard) per model\n"
            "How similar are Exp2 translated keywords to Exp3 direct English keywords?"
        )
        ax.grid(axis="y", alpha=0.3)
        saved.append(_save(fig, out_dir / "cross" / "exp2_exp3_keyword_jaccard.png"))

    # ── Plot 2: Top-20 keywords side-by-side ──────────────────────────────

    if cnt2 and cnt3:
        top2 = cnt2.most_common(20)
        top3 = cnt3.most_common(20)

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
        kws2, vals2 = zip(*top2)
        kws3, vals3 = zip(*top3)

        ax2.barh(list(kws2)[::-1], list(vals2)[::-1], color="#F59E0B", alpha=0.85)
        ax2.set_title("Exp2 — Top 20 translated keywords\n(original-language → English)")
        ax2.set_xlabel("Frequency")
        ax2.grid(axis="x", alpha=0.3)

        ax3.barh(list(kws3)[::-1], list(vals3)[::-1], color="#3B82F6", alpha=0.85)
        ax3.set_title("Exp3 — Top 20 keywords\n(direct from English translation)")
        ax3.set_xlabel("Frequency")
        ax3.grid(axis="x", alpha=0.3)

        fig2.suptitle(
            "Exp2 vs Exp3 — Top keyword comparison (all models & languages)\n"
            "Are the same concepts surfacing through translation vs direct extraction?",
            fontsize=11,
        )
        saved.append(_save(fig2, out_dir / "cross" / "exp2_exp3_top_keywords_comparison.png"))

    return saved


def plot_exp2_exp3_multidimensional(
    merged_23: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """
    Multi-dimensional analysis connecting Exp2 keywords, Exp3 keywords, agreement,
    and correctness. Produces two plots:

      1. Scatter: Exp2 keyword count vs Exp3 keyword count, colored by agreement
         (yes=green, no=red), faceted by language — reveals whether keyword volume
         patterns differ for agreed vs disagreed cases.

      2. Grouped bar: for each (agree/disagree) × (correct/wrong) quadrant,
         show the avg Exp2 keyword count and avg Exp3 keyword count — reveals
         whether keyword verbosity is linked to both agreement and correctness.
    """
    if merged_23.empty:
        _skip("no Exp2+3 merged data", "exp2_exp3_multidimensional"); return []

    kc2 = _COL_KW_COUNT_EXP2
    kc3 = _COL_KW_COUNT_EXP3
    ag  = _COL_AGREEMENT_EXP3
    cor = _COL_CORRECT_EXP2

    missing = [c for c in [kc2, kc3, ag] if c not in merged_23.columns]
    if missing:
        _skip(f"missing columns: {missing}", "exp2_exp3_multidimensional"); return []

    sub = merged_23[merged_23[ag].isin(["yes", "no"])].copy()
    sub[kc2] = pd.to_numeric(sub[kc2], errors="coerce")
    sub[kc3] = pd.to_numeric(sub[kc3], errors="coerce")
    sub = sub.dropna(subset=[kc2, kc3])
    if sub.empty:
        _skip("no valid rows after filtering", "exp2_exp3_multidimensional"); return []

    saved = []
    ag_colors = {"yes": "#22C55E", "no": "#EF4444"}
    cmaps    = {"yes": "Greens",   "no": "Reds"}

    # ── Plot 1: density (hexbin) per language — replaces transparent scatter ──
    # With ~10 K points per panel, hexbin reveals density far better than
    # alpha-blended individual points. Mean diamonds are overlaid for each group.
    langs = sorted(sub["language"].unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 5), sharey=True)
    axes = _ensure_axes_list(axes)

    for ax, lang in zip(axes, langs):
        grp = sub[sub["language"] == lang]
        for ag_val in ["yes", "no"]:
            pts = grp[grp[ag] == ag_val]
            if pts.empty:
                continue
            ax.hexbin(pts[kc2].values, pts[kc3].values,
                      gridsize=20, cmap=cmaps[ag_val],
                      alpha=0.55, mincnt=1, linewidths=0)
            ax.scatter(
                pts[kc2].mean(), pts[kc3].mean(),
                c=ag_colors[ag_val], s=160, marker="D",
                edgecolors="black", linewidths=1.2,
                label=f"Agree={ag_val} (mean)", zorder=5,
            )
        ax.set_xlabel("Exp2 keyword count")
        ax.set_title(f"{lang.capitalize()}", fontsize=11)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Exp3 keyword count")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Exp2 keyword count vs Exp3 keyword count by Exp3 agreement\n"
        "Hexbin = density; diamonds = group means",
        fontsize=11,
    )
    saved.append(_save(fig, out_dir / "cross" / "exp2_exp3_kw_count_scatter.png"))

    # ── Plot 2: quadrant bar chart ────────────────────────────────────────
    if cor not in sub.columns:
        return saved  # skip quadrant chart if no correctness column

    sub_c = sub.dropna(subset=[cor])
    quadrants = [
        ("Correct + Agree",    (sub_c[cor] == True)  & (sub_c[ag] == "yes")),
        ("Correct + Disagree", (sub_c[cor] == True)  & (sub_c[ag] == "no")),
        ("Wrong + Disagree",   (sub_c[cor] == False) & (sub_c[ag] == "no")),
        ("Wrong + Agree",      (sub_c[cor] == False) & (sub_c[ag] == "yes")),
    ]
    q_labels, q_mean2, q_mean3, q_counts = [], [], [], []
    for label, mask in quadrants:
        pts = sub_c[mask]
        if len(pts) >= 10:
            q_labels.append(f"{label}\n(n={len(pts):,})")
            q_mean2.append(pts[kc2].mean())
            q_mean3.append(pts[kc3].mean())
            q_counts.append(len(pts))

    if len(q_labels) >= 2:
        x = np.arange(len(q_labels))
        w = 0.35
        q_colors = ["#22C55E", "#F97316", "#3B82F6", "#EF4444"]

        fig2, ax2 = plt.subplots(figsize=(11, 5))
        bars2 = ax2.bar(x - w / 2, q_mean2, w, label="Avg Exp2 kw count", color="#F59E0B", alpha=0.85)
        bars3 = ax2.bar(x + w / 2, q_mean3, w, label="Avg Exp3 kw count", color="#3B82F6", alpha=0.85)
        for bar, v in zip(list(bars2) + list(bars3), q_mean2 + q_mean3):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(q_labels, fontsize=9)
        ax2.set_ylabel("Average keyword count")
        ax2.set_title(
            "Avg keyword count (Exp2 vs Exp3) across correctness × agreement quadrants\n"
            "Do disagreement cases extract more or fewer keywords?"
        )
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        saved.append(_save(fig2, out_dir / "cross" / "exp2_exp3_kw_quadrant_bars.png"))

    return saved


def plot_cross_keyword_agreement_connection(
    merged_23: pd.DataFrame, out_dir: Path
) -> Path | None:
    """Box plot: keyword count (Exp2) split by agreement label (Exp3)."""
    if merged_23.empty:
        _skip("no Exp2+3 merged data", "cross_keyword_agreement_connection"); return None

    kc_col = _COL_KW_COUNT_EXP2
    ag_col = _COL_AGREEMENT_EXP3

    if kc_col not in merged_23.columns or ag_col not in merged_23.columns:
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


# ── Experiment 2 — native-language keyword frequency per language ─────────────

def plot_exp2_native_keywords_per_language(
    registry: Registry, out_dir: Path
) -> list[Path]:
    """
    For each language, one figure with rows=models, 2 cols (depressed / not depressed).
    Top native-language keywords are shown in original script (Arabic / Chinese / Urdu).

    Arabic text is reshaped for correct letter-joining if arabic_reshaper and
    python-bidi are installed; otherwise raw Unicode is used (may look disconnected
    but remains readable).
    """
    import matplotlib.font_manager as fm
    from collections import Counter

    # ── font selection ────────────────────────────────────────────────────────
    # Arabic fonts on Windows store standard Unicode codepoints (U+0600-U+06FF).
    # arabic_reshaper converts to presentation forms (U+FExx) which these fonts
    # do NOT have — so we skip reshaping and render raw Arabic Unicode directly.
    # Matplotlib cannot do full Arabic shaping (letter-joining), but the glyphs
    # are correct and readable for a research visualisation.
    _LANG_FONT_CANDIDATES = {
        "arabic": [
            "Traditional Arabic", "Simplified Arabic", "Arabic Typesetting",
            "Noto Naskh Arabic", "Amiri", "Arial",
        ],
        "chinese": [
            "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
            "WenQuanYi Micro Hei", "DejaVu Sans",
        ],
        "urdu": ["DejaVu Sans", "Arial", "Noto Sans"],  # Roman Urdu — Latin script
    }

    _available_fonts = {f.name for f in fm.fontManager.ttflist}

    def _best_font(lang: str) -> fm.FontProperties | None:
        for name in _LANG_FONT_CANDIDATES.get(lang, []):
            if name in _available_fonts:
                return fm.FontProperties(family=name)
        return None

    # ── data ──────────────────────────────────────────────────────────────────
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keywords" not in df2.columns:
        _skip("no Exp2 keyword data", "exp2_native_keywords_per_language")
        return []

    saved: list[Path] = []
    languages = sorted(df2["language"].unique())
    classes = ["depressed", "not depressed"]
    class_colors = {"depressed": "#EF4444", "not depressed": "#3B82F6"}
    TOP_N = 15

    for lang in languages:
        lang_df = df2[df2["language"] == lang].copy()

        # Keep only models that have actual varied keyword data
        # (gemma/chinese is excluded because it erroneously returned 1 keyword per entry)
        if lang == "chinese":
            # Use only the larger gemma-chinese file; _exclude_gemma_chinese keeps it
            # but the existing pipeline notes it as problematic — skip it here too
            lang_df = lang_df[lang_df["model"] != "gemma"]

        models = sorted(lang_df["model"].unique())
        if not models:
            continue

        font_prop = _best_font(lang)
        n_models = len(models)
        fig, axes = plt.subplots(
            n_models, 2,
            figsize=(18, 4 * n_models),
            squeeze=False,
        )

        for row_idx, model in enumerate(models):
            model_df = lang_df[lang_df["model"] == model]

            for col_idx, cls in enumerate(classes):
                ax = axes[row_idx, col_idx]
                cls_df = model_df[model_df["ground_truth"] == cls]

                # Collect original-language keywords.
                # Filter out error/fallback strings: anything longer than 40 chars
                # or containing 4+ consecutive ASCII words (model error messages
                # accidentally stored as keywords in some Claude/Llama entries).
                def _is_valid_kw(kw: str) -> bool:
                    kw = kw.strip()
                    if not kw or len(kw) > 40:
                        return False
                    # reject if it looks like a full English sentence
                    ascii_words = [w for w in kw.split() if w.isascii() and w.isalpha()]
                    if len(ascii_words) >= 4:
                        return False
                    return True

                all_kws: list[str] = []
                for kws in cls_df["keywords"]:
                    if isinstance(kws, list):
                        all_kws.extend(k.strip() for k in kws if _is_valid_kw(k))

                if not all_kws:
                    ax.text(
                        0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=11, color="#9CA3AF",
                    )
                    ax.set_axis_off()
                    if col_idx == 0:
                        ax.set_ylabel(
                            model.capitalize(), fontsize=11, rotation=90, labelpad=14
                        )
                    continue

                counter = Counter(all_kws)
                top = counter.most_common(TOP_N)

                labels = [t[0] for t in reversed(top)]
                counts = [t[1] for t in reversed(top)]

                y_pos = range(len(labels))
                ax.barh(y_pos, counts, color=class_colors[cls], alpha=0.82)
                ax.set_yticks(list(y_pos))

                if font_prop:
                    ax.set_yticklabels(labels, fontproperties=font_prop, fontsize=9)
                else:
                    ax.set_yticklabels(labels, fontsize=9)

                # Annotate bar values
                for i, v in enumerate(counts):
                    ax.text(
                        v + max(counts) * 0.01, i, str(v),
                        va="center", fontsize=7, color="#374151",
                    )

                ax.set_xlabel("Frequency", fontsize=9)
                ax.grid(axis="x", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Column headers on the top row only
                if row_idx == 0:
                    ax.set_title(
                        cls.title(),
                        fontsize=13,
                        color=class_colors[cls],
                        fontweight="bold",
                        pad=10,
                    )

                # Model name on the left column only
                if col_idx == 0:
                    ax.set_ylabel(
                        model.capitalize(), fontsize=11, rotation=90, labelpad=14
                    )

        lang_display = lang.title()
        fig.suptitle(
            f"Experiment 2 — Top {TOP_N} keywords by class  ·  {lang_display}\n"
            "(keywords in original language script)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        path = out_dir / "exp2" / f"native_keywords_{lang}.png"
        saved.append(_save(fig, path))

    return [p for p in saved if p]


# ── Experiment 2 — combined native keywords (all models pooled) per language ──

def plot_exp2_native_keywords_combined(
    registry: Registry, out_dir: Path
) -> list[Path]:
    """
    For each language, one figure with 2 panels: Depressed (left) and Not Depressed (right).
    Keywords from ALL models are pooled and ranked by total frequency — shows the
    language-level signal regardless of which model extracted it.
    """
    import matplotlib.font_manager as fm
    from collections import Counter

    _LANG_FONT_CANDIDATES = {
        "arabic": [
            "Traditional Arabic", "Simplified Arabic", "Arabic Typesetting",
            "Noto Naskh Arabic", "Amiri", "Arial",
        ],
        "chinese": [
            "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
            "WenQuanYi Micro Hei", "DejaVu Sans",
        ],
        "urdu": ["DejaVu Sans", "Arial", "Noto Sans"],
    }
    _available_fonts = {f.name for f in fm.fontManager.ttflist}

    def _best_font(lang: str) -> fm.FontProperties | None:
        for name in _LANG_FONT_CANDIDATES.get(lang, []):
            if name in _available_fonts:
                return fm.FontProperties(family=name)
        return None

    def _is_valid_kw(kw: str) -> bool:
        kw = kw.strip()
        if not kw or len(kw) > 40:
            return False
        ascii_words = [w for w in kw.split() if w.isascii() and w.isalpha()]
        return len(ascii_words) < 4

    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keywords" not in df2.columns:
        _skip("no Exp2 keyword data", "exp2_native_keywords_combined")
        return []

    saved: list[Path] = []
    classes = ["depressed", "not depressed"]
    class_colors = {"depressed": "#EF4444", "not depressed": "#3B82F6"}
    TOP_N = 20

    for lang in sorted(df2["language"].unique()):
        lang_df = df2[df2["language"] == lang].copy()
        if lang == "chinese":
            lang_df = lang_df[lang_df["model"] != "gemma"]

        models_used = sorted(lang_df["model"].unique())
        font_prop = _best_font(lang)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        for ax, cls in zip(axes, classes):
            cls_df = lang_df[lang_df["ground_truth"] == cls]

            all_kws: list[str] = []
            for kws in cls_df["keywords"]:
                if isinstance(kws, list):
                    all_kws.extend(k.strip() for k in kws if _is_valid_kw(k))

            if not all_kws:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=13, color="#9CA3AF")
                ax.set_axis_off()
                continue

            counter = Counter(all_kws)
            top = counter.most_common(TOP_N)
            labels = [t[0] for t in reversed(top)]
            counts = [t[1] for t in reversed(top)]

            y_pos = list(range(len(labels)))
            ax.barh(y_pos, counts, color=class_colors[cls], alpha=0.82)
            ax.set_yticks(y_pos)
            if font_prop:
                ax.set_yticklabels(labels, fontproperties=font_prop, fontsize=11)
            else:
                ax.set_yticklabels(labels, fontsize=11)

            max_count = max(counts)
            for i, v in enumerate(counts):
                ax.text(v + max_count * 0.005, i, str(v),
                        va="center", fontsize=8, color="#374151")

            ax.set_title(cls.title(), fontsize=15, color=class_colors[cls],
                         fontweight="bold", pad=12)
            ax.set_xlabel("Total frequency (all models pooled)", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        models_label = ", ".join(m.capitalize() for m in models_used)
        fig.suptitle(
            f"Experiment 2 — Top {TOP_N} keywords by class  ·  {lang.title()}  (all models combined)\n"
            f"Models pooled: {models_label}  ·  keywords in original language script",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        path = out_dir / "exp2" / f"native_keywords_{lang}_combined.png"
        saved.append(_save(fig, path))

    return [p for p in saved if p]


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
    _try(plot_exp2_keyword_count_by_predicted_class, registry, out_dir)
    _try(plot_exp2_native_keywords_per_language, registry, out_dir)
    _try(plot_exp2_native_keywords_combined, registry, out_dir)

    # Experiment 3
    _try(plot_exp3_agreement_rates, agreement_summary, out_dir)
    _try(plot_exp3_agreement_breakdown, agreement_summary, out_dir)
    _try(plot_exp3_agreement_heatmap, agreement_summary, out_dir)

    # Cross-experiment
    _try(plot_cross_f1_vs_agreement, summary, agreement_summary, out_dir)
    _try(plot_cross_exp1_exp3_flip_analysis, merges.get("exp1_exp3", pd.DataFrame()), out_dir)
    _try(plot_exp2_exp3_keyword_count_paired, merges.get("exp2_exp3", pd.DataFrame()), out_dir)
    _try(plot_exp2_exp3_keyword_overlap_and_top, merges.get("exp2_exp3", pd.DataFrame()), out_dir)
    _try(plot_exp2_exp3_multidimensional, merges.get("exp2_exp3", pd.DataFrame()), out_dir)
    _try(plot_cross_keyword_agreement_connection, merges.get("exp2_exp3", pd.DataFrame()), out_dir)

    print(f"\n[plots] {len(saved)} plot(s) saved.\n")
    return saved
