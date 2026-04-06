"""
merger.py — Join and merge DataFrames across experiments.

All joins are performed by (model, language, index) using inner joins so that
only rows present in all joined experiments are included. Missing-experiment
combos are skipped and logged.

Provides:
  merge_exp1_exp2(registry)    → DataFrame
  merge_exp2_exp3(registry)    → DataFrame
  merge_exp1_exp3(registry)    → DataFrame
  merge_all_three(registry)    → DataFrame
  build_all_merges(registry)   → dict[str, DataFrame]
  keyword_summary(registry)    → DataFrame  (per-model-lang keyword stats)
  top_keywords_table(registry) → DataFrame  (most frequent keywords)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from loader import Registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_suffixes(df: pd.DataFrame, suffix: str,
                  skip_cols: set[str]) -> pd.DataFrame:
    """Rename all columns except skip_cols by appending suffix."""
    return df.rename(columns={
        c: f"{c}_{suffix}" for c in df.columns if c not in skip_cols
    })


_JOIN_KEYS = {"model", "language", "index"}


def _merge_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_suffix: str,
    right_suffix: str,
    shared_passthrough: list[str] | None = None,
) -> pd.DataFrame:
    """
    Inner-join two sample DataFrames on (model, language, index).

    Columns that are identical in both (ground_truth, word_count, etc.) are
    kept once from the left.  Experiment-specific columns get suffixes.
    """
    if left.empty or right.empty:
        return pd.DataFrame()

    # Columns to keep as-is from left (no suffix)
    passthrough = list(_JOIN_KEYS) + (shared_passthrough or [])
    passthrough_present = [c for c in passthrough if c in left.columns]

    # Shared columns that are truly identical between experiments
    _shared = {"ground_truth", "word_count", "post_preview", "post_full",
               "model_type", "experiment"}
    shared_from_left = [
        c for c in _shared if c in left.columns and c in right.columns
    ]
    keep_from_left = set(passthrough_present + shared_from_left)

    # Suffix columns that need disambiguation
    left_cols_to_suffix = [c for c in left.columns if c not in keep_from_left]
    right_cols_to_suffix = [c for c in right.columns if c not in _JOIN_KEYS]

    left_renamed = left[list(keep_from_left) + left_cols_to_suffix].rename(
        columns={c: f"{c}_{left_suffix}" for c in left_cols_to_suffix}
    )
    right_renamed = right[list(_JOIN_KEYS) + right_cols_to_suffix].rename(
        columns={c: f"{c}_{right_suffix}" for c in right_cols_to_suffix}
    )

    merged = left_renamed.merge(right_renamed, on=list(_JOIN_KEYS), how="inner")
    return merged


# ── per-combination merges ────────────────────────────────────────────────────

def _iter_common_combos(
    registry: Registry,
    exp_a: int,
    exp_b: int,
) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame]]:
    """Yield (model, lang, df_a, df_b) for all combos present in both experiments."""
    results = []
    for model in registry.all_models():
        for lang in registry.all_languages():
            df_a = registry.get_df(exp_a, model, lang)
            df_b = registry.get_df(exp_b, model, lang)
            if df_a is not None and not df_a.empty and df_b is not None and not df_b.empty:
                results.append((model, lang, df_a, df_b))
            elif df_a is not None and df_b is None:
                pass  # silently skip — normal if exp2/3 not yet run
    return results


def merge_exp1_exp2(registry: Registry) -> pd.DataFrame:
    """Join Exp 1 classification data with Exp 2 keyword data."""
    combos = _iter_common_combos(registry, 1, 2)
    if not combos:
        print("[merger] No common (Exp1, Exp2) combos found.")
        return pd.DataFrame()
    frames = []
    for model, lang, df1, df2 in combos:
        m = _merge_two(df1, df2, "exp1", "exp2")
        if not m.empty:
            frames.append(m)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    print(f"[merger] Exp1+Exp2: {len(result)} rows across {len(combos)} model-lang combos")
    return result


def merge_exp2_exp3(registry: Registry) -> pd.DataFrame:
    """Join Exp 2 keyword data with Exp 3 translation re-eval data."""
    combos = _iter_common_combos(registry, 2, 3)
    if not combos:
        print("[merger] No common (Exp2, Exp3) combos found.")
        return pd.DataFrame()
    frames = []
    for model, lang, df2, df3 in combos:
        m = _merge_two(df2, df3, "exp2", "exp3")
        if not m.empty:
            frames.append(m)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    print(f"[merger] Exp2+Exp3: {len(result)} rows across {len(combos)} model-lang combos")
    return result


def merge_exp1_exp3(registry: Registry) -> pd.DataFrame:
    """Join Exp 1 classification data with Exp 3 translation re-eval data."""
    combos = _iter_common_combos(registry, 1, 3)
    if not combos:
        print("[merger] No common (Exp1, Exp3) combos found.")
        return pd.DataFrame()
    frames = []
    for model, lang, df1, df3 in combos:
        m = _merge_two(df1, df3, "exp1", "exp3")
        if not m.empty:
            frames.append(m)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    print(f"[merger] Exp1+Exp3: {len(result)} rows across {len(combos)} model-lang combos")
    return result


def merge_all_three(registry: Registry) -> pd.DataFrame:
    """Join all three experiments. Only includes rows present in all three."""
    frames = []
    skipped = []
    for model in registry.all_models():
        for lang in registry.all_languages():
            df1 = registry.get_df(1, model, lang)
            df2 = registry.get_df(2, model, lang)
            df3 = registry.get_df(3, model, lang)
            if df1 is None or df2 is None or df3 is None:
                if df1 is not None:  # has exp1, missing something
                    skipped.append(f"{model}/{lang} (missing exp{2 if df2 is None else 3})")
                continue
            if df1.empty or df2.empty or df3.empty:
                continue

            # Merge 1+2 first, then +3
            m12 = _merge_two(df1, df2, "exp1", "exp2")
            if m12.empty:
                continue
            m123 = _merge_two(m12, df3, "e12", "exp3",
                               shared_passthrough=list(_JOIN_KEYS))
            if not m123.empty:
                frames.append(m123)

    if skipped:
        print(f"[merger] Exp1+2+3 skipped (incomplete): {', '.join(skipped)}")
    if not frames:
        print("[merger] No complete Exp1+2+3 combos found.")
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    print(f"[merger] Exp1+2+3: {len(result)} rows")
    return result


# ── keyword aggregation ───────────────────────────────────────────────────────

def keyword_summary(registry: Registry) -> pd.DataFrame:
    """
    Per (model, language) keyword statistics from Exp 2.
    Returns a DataFrame with columns:
      model, language, total_entries, entries_with_keywords,
      avg_keyword_count, total_unique_keywords, ...
    """
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "keyword_count" not in df2.columns:
        return pd.DataFrame()

    rows = []
    for (model, lang), group in df2.groupby(["model", "language"]):
        kw_counts = group["keyword_count"]
        # Explode keyword lists to count unique tokens
        all_kws: list[str] = []
        for kws in group.get("keywords", pd.Series(dtype=object)):
            if isinstance(kws, list):
                all_kws.extend([k.lower().strip() for k in kws if k])

        has_kws = (kw_counts > 0).sum()
        rows.append({
            "model": model,
            "language": lang,
            "total_entries": len(group),
            "entries_with_keywords": int(has_kws),
            "pct_with_keywords": round(has_kws / len(group) * 100, 1),
            "avg_keyword_count": round(kw_counts.mean(), 2),
            "median_keyword_count": kw_counts.median(),
            "total_keyword_occurrences": int(kw_counts.sum()),
            "total_unique_keywords": len(set(all_kws)),
        })

    return pd.DataFrame(rows).sort_values(["model", "language"])


def top_keywords_table(
    registry: Registry,
    n: int = 30,
    by_class: bool = False,
) -> pd.DataFrame:
    """
    Most frequent translated keywords from Exp 2.

    If by_class=True, returns separate counts for depressed vs not-depressed.
    """
    df2 = registry.all_samples(experiment=2)
    if df2.empty or "translations" not in df2.columns:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in df2.iterrows():
        trans = row.get("translations")
        if not isinstance(trans, list):
            continue
        for kw in trans:
            kw_clean = str(kw).lower().strip()
            if not kw_clean:
                continue
            rec: dict[str, Any] = {
                "keyword": kw_clean,
                "model": row["model"],
                "language": row["language"],
            }
            if by_class:
                rec["ground_truth"] = row.get("ground_truth", "")
                rec["correct"] = row.get("correct")
            records.append(rec)

    if not records:
        return pd.DataFrame()

    kdf = pd.DataFrame(records)
    if by_class:
        result = (
            kdf.groupby(["keyword", "ground_truth"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(n * 2)
        )
    else:
        result = (
            kdf.groupby("keyword")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(n)
        )
    return result


def exp3_agreement_summary(registry: Registry) -> pd.DataFrame:
    """
    Per (model, language) agreement statistics from Exp 3.
    """
    df3 = registry.all_samples(experiment=3)
    if df3.empty or "agreement" not in df3.columns:
        return pd.DataFrame()

    rows = []
    for (model, lang), group in df3.groupby(["model", "language"]):
        total = len(group)
        agree = (group["agreement"] == "yes").sum()
        disagree = (group["agreement"] == "no").sum()
        no_trans = (group["agreement"] == "no_translation").sum()
        other = total - agree - disagree - no_trans

        # Agreement rate on classified entries only
        classified = agree + disagree
        rate = agree / classified if classified > 0 else None

        # Among disagreements, check label flips
        flip_mask = (
            (group["agreement"] == "no") &
            group.get("prediction", pd.Series(dtype=str)).notna()
        )
        flips = flip_mask.sum()

        rows.append({
            "model": model,
            "language": lang,
            "total": total,
            "agree": int(agree),
            "disagree": int(disagree),
            "no_translation": int(no_trans),
            "other": int(other),
            "agreement_rate": round(float(rate), 4) if rate is not None else None,
            "flip_count": int(flips),
            "flip_rate": round(flips / total, 4) if total > 0 else None,
        })

    return pd.DataFrame(rows).sort_values(["model", "language"])


# ── build all at once ─────────────────────────────────────────────────────────

def build_all_merges(registry: Registry) -> dict[str, pd.DataFrame]:
    """Run all merges and return a named dict."""
    print("\n[merger] Building merged tables...")
    return {
        "exp1_exp2": merge_exp1_exp2(registry),
        "exp2_exp3": merge_exp2_exp3(registry),
        "exp1_exp3": merge_exp1_exp3(registry),
        "all_three": merge_all_three(registry),
    }
