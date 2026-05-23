"""
Generate a 2×3 word cloud grid from Experiment 2 keyword data.

Rows:    Depressed, Not Depressed
Columns: Arabic, Urdu, Chinese

Usage:
    python data_visualization/generate_wordclouds.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment2"
OUT_DIR = PROJECT_ROOT / "data_visualization" / "outputs" / "plots"

LANGUAGES = ["arabic", "urdu", "chinese"]
CLASSES = ["depressed", "not depressed"]

LANG_DISPLAY = {"arabic": "Arabic", "urdu": "Urdu", "chinese": "Chinese"}
CLASS_DISPLAY = {"depressed": "Depressed", "not depressed": "Not Depressed"}

# ── collect keywords ─────────────────────────────────────────────────────────

def collect_keywords() -> dict[tuple[str, str], Counter]:
    """
    Scan all exp2 result JSONs.
    Returns {(language, ground_truth): Counter({keyword: freq})}.
    """
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

            # translations = English keywords from Exp2
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


# ── generate plot ────────────────────────────────────────────────────────────

def generate_wordcloud_grid(freqs: dict[tuple[str, str], Counter]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    dep_cmap = "Reds"
    notdep_cmap = "Blues"

    for col_idx, lang in enumerate(LANGUAGES):
        for row_idx, cls in enumerate(CLASSES):
            ax = axes[row_idx, col_idx]
            counts = freqs.get((lang, cls), Counter())

            if not counts:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=14, color="#999")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                cmap = dep_cmap if cls == "depressed" else notdep_cmap
                wc = WordCloud(
                    width=800, height=500,
                    background_color="white",
                    colormap=cmap,
                    max_words=80,
                    prefer_horizontal=0.7,
                    min_font_size=8,
                    max_font_size=90,
                    relative_scaling=0.5,
                )
                wc.generate_from_frequencies(counts)
                ax.imshow(wc, interpolation="bilinear")

            ax.axis("off")

            # Column titles (top row only)
            if row_idx == 0:
                ax.set_title(LANG_DISPLAY[lang], fontsize=16, fontweight="bold",
                             pad=10, color="#333")

    # Row labels on the left
    for row_idx, cls in enumerate(CLASSES):
        color = "#DC2626" if cls == "depressed" else "#2563EB"
        fig.text(0.02, 0.72 - row_idx * 0.48, CLASS_DISPLAY[cls],
                 fontsize=15, fontweight="bold", color=color,
                 rotation=90, va="center", ha="center")

    fig.suptitle("Experiment 2 Keyword Clouds by Language and Class",
                 fontsize=18, fontweight="bold", y=0.98, color="#222")

    plt.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.05,
                        wspace=0.05, hspace=0.08)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "exp2_keyword_wordclouds.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting Exp2 keywords by (language, ground_truth)...")
    freqs = collect_keywords()

    for lang in LANGUAGES:
        for cls in CLASSES:
            c = freqs[(lang, cls)]
            print(f"  {lang:8} / {cls:14}: {sum(c.values()):>6} total, "
                  f"{len(c):>4} unique keywords")

    print("\nGenerating word cloud grid...")
    out = generate_wordcloud_grid(freqs)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
