"""
Scientific word cloud: 2×3 grid with quantitative annotations + discriminability panel.

Usage:
    python data_visualization/generate_wordclouds_scientific.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from wordcloud import WordCloud

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2" / "experiment2"
OUT_DIR = PROJECT_ROOT / "data_visualization" / "outputs" / "plots"

LANGUAGES = ["arabic", "urdu", "chinese"]
CLASSES = ["depressed", "not depressed"]
LANG_DISPLAY = {"arabic": "Arabic", "urdu": "Urdu", "chinese": "Chinese"}
CLASS_DISPLAY = {"depressed": "Depressed", "not depressed": "Not Depressed"}

TOP_K = 20


# ── data collection ──────────────────────────────────────────────────────────

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
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 0.0
    probs = [c / total for c in counter.values()]
    raw = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counter))
    return raw / max_entropy if max_entropy > 0 else 0.0


def top_k_concentration(counter: Counter, k: int = 10) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(c for _, c in counter.most_common(k)) / total


def discriminability(dep: Counter, notdep: Counter, k: int) -> float:
    dep_top = set(w for w, _ in dep.most_common(k))
    notdep_top = set(w for w, _ in notdep.most_common(k))
    overlap = dep_top & notdep_top
    union = dep_top | notdep_top
    return 1.0 - (len(overlap) / len(union)) if union else 0.0


# ── plot ─────────────────────────────────────────────────────────────────────

def generate(freqs: dict[tuple[str, str], Counter]) -> Path:
    fig = plt.figure(figsize=(22, 10))

    # GridSpec: 2 rows × 4 cols — 3 word cloud columns + 1 bar column on the right
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           width_ratios=[1, 1, 1, 0.38],
                           hspace=0.18, wspace=0.12,
                           left=0.05, right=0.97, top=0.91, bottom=0.05)

    dep_cmap = "Reds"
    notdep_cmap = "Blues"

    for col_idx, lang in enumerate(LANGUAGES):
        for row_idx, cls in enumerate(CLASSES):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            counts = freqs.get((lang, cls), Counter())

            if not counts:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=14, color="#999")
            else:
                cmap = dep_cmap if cls == "depressed" else notdep_cmap
                wc = WordCloud(
                    width=800, height=500,
                    background_color="white",
                    colormap=cmap,
                    max_words=80,
                    prefer_horizontal=0.7,
                    min_font_size=8,
                    max_font_size=85,
                    relative_scaling=0.5,
                )
                wc.generate_from_frequencies(counts)
                ax.imshow(wc, interpolation="bilinear")

                # Metric annotations — bottom-left box
                entropy = shannon_entropy(counts)
                conc = top_k_concentration(counts, 10)
                n_unique = len(counts)
                annotation = (f"H = {entropy:.3f}  |  "
                              f"Top-10: {conc:.1%}  |  "
                              f"n = {n_unique:,}")
                ax.text(0.5, -0.02, annotation, transform=ax.transAxes,
                        ha="center", va="top", fontsize=9,
                        color="#555", fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="#DDD", alpha=0.9))

            ax.axis("off")

            # Column titles
            if row_idx == 0:
                ax.set_title(LANG_DISPLAY[lang], fontsize=16,
                             fontweight="bold", pad=12, color="#333")

    # Row labels
    for row_idx, cls in enumerate(CLASSES):
        color = "#DC2626" if cls == "depressed" else "#2563EB"
        fig.text(0.015, 0.73 - row_idx * 0.42, CLASS_DISPLAY[cls],
                 fontsize=15, fontweight="bold", color=color,
                 rotation=90, va="center", ha="center")

    # --- Right column: discriminability bars (spans both rows) ---
    ax_bar = fig.add_subplot(gs[:, 3])

    disc_scores = []
    for lang in LANGUAGES:
        d = discriminability(freqs[(lang, "depressed")],
                             freqs[(lang, "not depressed")], TOP_K)
        disc_scores.append(d)

    y = np.arange(len(LANGUAGES))
    colors = ["#EF4444", "#F59E0B", "#3B82F6"]
    bars = ax_bar.barh(y, disc_scores, height=0.55, color=colors,
                       alpha=0.85, edgecolor="white", linewidth=1.5)

    for bar, score in zip(bars, disc_scores):
        w = bar.get_width()
        ax_bar.text(w - 0.03, bar.get_y() + bar.get_height()/2,
                    f"{score:.2f}", va="center", ha="right",
                    fontsize=13, fontweight="bold", color="white")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([LANG_DISPLAY[l] for l in LANGUAGES], fontsize=12)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Discriminability", fontsize=11, color="#555")
    ax_bar.set_title(f"Class Separability\n(Top-{TOP_K} Keyword Overlap)",
                     fontsize=12, fontweight="bold", pad=10, color="#333")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.invert_yaxis()

    # Small legend below the bar
    ax_bar.text(0.5, -0.08, "1.0 = fully distinct\n0.0 = identical sets",
                transform=ax_bar.transAxes, ha="center", va="top",
                fontsize=9, color="#777", fontstyle="italic")

    # Suptitle
    fig.suptitle("Experiment 2: Keyword Attribution Analysis by Language and Class",
                 fontsize=18, fontweight="bold", y=0.97, color="#222")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "exp2_keyword_scientific.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting keywords...")
    freqs = collect_keywords()

    for lang in LANGUAGES:
        for cls in CLASSES:
            c = freqs[(lang, cls)]
            print(f"  {LANG_DISPLAY[lang]:8} / {cls:14}: {len(c):>5} unique, "
                  f"entropy={shannon_entropy(c):.3f}")

    print(f"\nDiscriminability (top-{TOP_K}):")
    for lang in LANGUAGES:
        d = discriminability(freqs[(lang, "depressed")],
                             freqs[(lang, "not depressed")], TOP_K)
        print(f"  {LANG_DISPLAY[lang]:8}: {d:.3f}")

    print("\nGenerating plot...")
    out = generate(freqs)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
