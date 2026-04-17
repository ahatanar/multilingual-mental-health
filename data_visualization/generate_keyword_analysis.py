"""
Quantitative keyword analysis from Experiment 2 data.

Produces:
  1. Top-10 keywords table per (language, class)
  2. Shannon entropy of keyword distributions (concentration metric)
  3. Keyword discriminability (exclusivity of top-K to their class)
  4. Bar chart of entropy + discriminability by language

Usage:
    python data_visualization/generate_keyword_analysis.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment2"
OUT_DIR = PROJECT_ROOT / "data_visualization" / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

LANGUAGES = ["arabic", "urdu", "chinese"]
CLASSES = ["depressed", "not depressed"]
LANG_DISPLAY = {"arabic": "Arabic", "urdu": "Urdu", "chinese": "Chinese"}

TOP_K = 20  # for discriminability analysis


# ── collect keywords (same as wordcloud script) ─────────────────────────────

def collect_keywords() -> dict[tuple[str, str], Counter]:
    freqs: dict[tuple[str, str], Counter] = {
        (lang, cls): Counter() for lang in LANGUAGES for cls in CLASSES
    }
    for path in RESULTS_DIR.rglob("*.json"):
        if path.name.endswith(".partial.json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta = data.get("metadata", {})
        lang = meta.get("language", "")
        if lang not in LANGUAGES:
            continue
        for entry in data.get("results", []):
            gt = entry.get("ground_truth", "").lower().strip()
            if gt not in CLASSES:
                continue
            trans = entry.get("translations", [])
            if isinstance(trans, str):
                trans = [k.strip() for k in trans.split(",") if k.strip()]
            if not isinstance(trans, list):
                continue
            for kw in trans:
                kw_clean = str(kw).lower().strip()
                if kw_clean and len(kw_clean) > 1:
                    freqs[(lang, gt)][kw_clean] += 1
    return freqs


# ── metrics ──────────────────────────────────────────────────────────────────

def shannon_entropy(counter: Counter) -> float:
    """Normalized Shannon entropy (0 = all mass on one word, 1 = uniform)."""
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 0.0
    probs = [c / total for c in counter.values()]
    raw = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counter))
    return raw / max_entropy if max_entropy > 0 else 0.0


def discriminability(dep_counter: Counter, notdep_counter: Counter, k: int) -> tuple[float, float]:
    """
    What fraction of a class's top-K keywords are exclusive to that class
    (don't appear in the other class's top-K)?
    Returns (dep_exclusivity, notdep_exclusivity).
    """
    dep_top = set(w for w, _ in dep_counter.most_common(k))
    notdep_top = set(w for w, _ in notdep_counter.most_common(k))

    dep_exclusive = dep_top - notdep_top
    notdep_exclusive = notdep_top - dep_top

    dep_excl_rate = len(dep_exclusive) / len(dep_top) if dep_top else 0
    notdep_excl_rate = len(notdep_exclusive) / len(notdep_top) if notdep_top else 0

    return dep_excl_rate, notdep_excl_rate


def keyword_concentration(counter: Counter, k: int = 10) -> float:
    """Fraction of total keyword mass captured by top-K keywords."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top_k_mass = sum(c for _, c in counter.most_common(k))
    return top_k_mass / total


# ── generate outputs ─────────────────────────────────────────────────────────

def print_top_keywords_table(freqs: dict[tuple[str, str], Counter], k: int = 10):
    """Print top-K keywords per (language, class) in a format ready for the paper."""
    print(f"\n{'='*80}")
    print(f"  Top-{k} Keywords by Language and Class (Experiment 2)")
    print(f"{'='*80}")
    for lang in LANGUAGES:
        for cls in CLASSES:
            counter = freqs[(lang, cls)]
            top = counter.most_common(k)
            total = sum(counter.values())
            label = f"{LANG_DISPLAY[lang]} — {cls.title()}"
            kw_str = ", ".join(f"{w} ({c})" for w, c in top)
            pct = sum(c for _, c in top) / total * 100 if total else 0
            print(f"\n  {label}  (top-{k} cover {pct:.1f}% of all keywords)")
            print(f"    {kw_str}")


def save_metrics_csv(freqs: dict[tuple[str, str], Counter]):
    """Save a CSV with entropy, concentration, and discriminability per language."""
    rows = []
    for lang in LANGUAGES:
        dep = freqs[(lang, "depressed")]
        notdep = freqs[(lang, "not depressed")]
        dep_excl, notdep_excl = discriminability(dep, notdep, TOP_K)

        rows.append({
            "language": LANG_DISPLAY[lang],
            "dep_unique_keywords": len(dep),
            "dep_total_occurrences": sum(dep.values()),
            "dep_entropy": round(shannon_entropy(dep), 4),
            "dep_top10_concentration": round(keyword_concentration(dep, 10), 4),
            "notdep_unique_keywords": len(notdep),
            "notdep_total_occurrences": sum(notdep.values()),
            "notdep_entropy": round(shannon_entropy(notdep), 4),
            "notdep_top10_concentration": round(keyword_concentration(notdep, 10), 4),
            f"dep_top{TOP_K}_exclusivity": round(dep_excl, 4),
            f"notdep_top{TOP_K}_exclusivity": round(notdep_excl, 4),
        })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "keyword_distribution_metrics.csv"

    # Write manually to avoid pandas dependency for this small table
    headers = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"\nSaved: {csv_path.relative_to(PROJECT_ROOT)}")
    return rows


def generate_metrics_plot(freqs: dict[tuple[str, str], Counter]) -> Path:
    """
    2-panel bar chart:
      Left:  Normalized entropy by language and class
      Right: Top-K keyword exclusivity (discriminability)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(LANGUAGES))
    width = 0.32

    # --- Panel 1: Entropy ---
    dep_entropy = [shannon_entropy(freqs[(l, "depressed")]) for l in LANGUAGES]
    notdep_entropy = [shannon_entropy(freqs[(l, "not depressed")]) for l in LANGUAGES]

    bars1 = ax1.bar(x - width/2, dep_entropy, width, label="Depressed",
                    color="#DC2626", alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, notdep_entropy, width, label="Not Depressed",
                    color="#2563EB", alpha=0.85, edgecolor="white", linewidth=0.8)

    ax1.set_ylabel("Normalized Shannon Entropy", fontsize=12)
    ax1.set_title("Keyword Distribution Entropy", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LANG_DISPLAY[l] for l in LANGUAGES], fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=9, color="#333")

    # Add annotation for the key finding
    ax1.annotate("Higher entropy =\nmore diffuse vocabulary",
                 xy=(0.98, 0.98), xycoords="axes fraction",
                 ha="right", va="top", fontsize=8.5, color="#666",
                 fontstyle="italic")

    # --- Panel 2: Discriminability ---
    dep_excl = []
    notdep_excl = []
    for l in LANGUAGES:
        de, ne = discriminability(freqs[(l, "depressed")], freqs[(l, "not depressed")], TOP_K)
        dep_excl.append(de)
        notdep_excl.append(ne)

    bars3 = ax2.bar(x - width/2, dep_excl, width, label="Depressed",
                    color="#DC2626", alpha=0.85, edgecolor="white", linewidth=0.8)
    bars4 = ax2.bar(x + width/2, notdep_excl, width, label="Not Depressed",
                    color="#2563EB", alpha=0.85, edgecolor="white", linewidth=0.8)

    ax2.set_ylabel(f"Top-{TOP_K} Exclusivity Rate", fontsize=12)
    ax2.set_title(f"Keyword Discriminability (Top-{TOP_K})", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LANG_DISPLAY[l] for l in LANGUAGES], fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=9, color="#333")

    ax2.annotate("Higher exclusivity =\ncleaner class separation",
                 xy=(0.98, 0.98), xycoords="axes fraction",
                 ha="right", va="top", fontsize=8.5, color="#666",
                 fontstyle="italic")

    # --- Panel 3 (bottom): Top-10 concentration ---
    # Actually let's add it as a small table annotation instead

    plt.tight_layout(w_pad=3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "exp2_keyword_metrics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting Exp2 keywords...")
    freqs = collect_keywords()

    # Print summary
    for lang in LANGUAGES:
        for cls in CLASSES:
            c = freqs[(lang, cls)]
            e = shannon_entropy(c)
            conc = keyword_concentration(c, 10)
            print(f"  {LANG_DISPLAY[lang]:8} / {cls:14}: "
                  f"{len(c):>5} unique, entropy={e:.4f}, top-10 concentration={conc:.3f}")

    # Top keywords table
    print_top_keywords_table(freqs, k=10)

    # Discriminability
    print(f"\n  Discriminability (top-{TOP_K} exclusivity):")
    for lang in LANGUAGES:
        de, ne = discriminability(freqs[(lang, "depressed")],
                                  freqs[(lang, "not depressed")], TOP_K)
        print(f"    {LANG_DISPLAY[lang]:8}: depressed={de:.3f}, not_depressed={ne:.3f}")

    # Save CSV
    save_metrics_csv(freqs)

    # Generate plot
    print("\nGenerating metrics plot...")
    out = generate_metrics_plot(freqs)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
