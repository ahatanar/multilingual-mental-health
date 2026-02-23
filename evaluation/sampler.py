"""Stratified dataset sampler for evaluation."""

import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Maximum word count per post — posts exceeding this are excluded
MAX_WORD_COUNT = 240
# For CJK languages, use character count instead
MAX_CHAR_COUNT = 240

# Languages that should use character-level length counting
CJK_LANGUAGES = {"chinese"}


class DatasetSampler:
    """
    Loads parsed data and performs stratified sampling.

    Filters out posts exceeding the length limit, then samples equally
    from each label class so the evaluation set is balanced.

    For CJK languages (Chinese), length is measured in characters
    instead of words.
    """

    def __init__(self, data: List[Dict[str, str]], language: str = "english",
                 max_words: int = MAX_WORD_COUNT, max_chars: int = MAX_CHAR_COUNT):
        """
        Args:
            data: List of {"post": str, "label": "depression"|"normal"} dicts
                  (output of a DatasetParser).
            language: Language identifier — used to pick word vs char counting.
            max_words: Maximum word count (for non-CJK languages).
            max_chars: Maximum character count (for CJK languages).
        """
        self.language = language.lower()
        self.is_cjk = self.language in CJK_LANGUAGES
        self.max_length = max_chars if self.is_cjk else max_words
        self.data = self._filter(data)

    def _measure_length(self, text: str) -> int:
        """Return the length of *text* — chars for CJK, words otherwise."""
        if self.is_cjk:
            return len(text)
        return len(text.split())

    def _filter(self, data: List[Dict[str, str]]) -> List[Dict]:
        """Filter out posts exceeding the length limit."""
        filtered = []
        for row in data:
            post = row.get("post", "").strip()
            label = row.get("label", "").strip().lower()
            length = self._measure_length(post)

            if not post or not label:
                continue
            if length > self.max_length:
                continue

            entry = {
                "post": post,
                "label": label,  # ground truth: "depression" or "normal"
                "word_count": length,
            }
            # Preserve extra fields (e.g. severity for Urdu)
            for key in row:
                if key not in ("post", "label", "word_count"):
                    entry[key] = row[key]
            filtered.append(entry)

        unit = "chars" if self.is_cjk else "words"
        logger.info(
            f"[{self.language}] Filtered to {len(filtered)} posts "
            f"(max {self.max_length} {unit}, from {len(data)} total)"
        )
        return filtered

    def sample(self, n: int = 500, seed: int = 42) -> List[Dict]:
        """
        Stratified sampling: n/2 from each label class.

        Args:
            n: Total number of samples (will be split equally across labels).
            seed: Random seed for reproducibility.

        Returns:
            List of sampled data dicts, shuffled.
        """
        random.seed(seed)

        # Group by label
        by_label: Dict[str, List[Dict]] = {}
        for row in self.data:
            by_label.setdefault(row["label"], []).append(row)

        labels = sorted(by_label.keys())
        per_label = n // len(labels)
        remainder = n % len(labels)

        logger.info(f"Sampling {n} total: {per_label} per label from {labels}")

        sampled = []
        for i, label in enumerate(labels):
            pool = by_label[label]
            count = per_label + (1 if i < remainder else 0)
            if count > len(pool):
                logger.warning(f"Requested {count} samples of '{label}' but only {len(pool)} available. Using all.")
                count = len(pool)
            sampled.extend(random.sample(pool, count))

        random.shuffle(sampled)

        # Normalize labels for consistency in evaluation
        for row in sampled:
            row["ground_truth"] = "depressed" if row["label"] == "depression" else "not depressed"

        logger.info(f"Sampled {len(sampled)} posts (seed={seed})")
        return sampled

    def get_stats(self) -> Dict:
        """Return basic stats about the loaded dataset."""
        from collections import Counter
        label_counts = Counter(row["label"] for row in self.data)
        lengths = [row["word_count"] for row in self.data]
        unit = "char_count" if self.is_cjk else "word_count"
        return {
            "total_posts": len(self.data),
            "label_distribution": dict(label_counts),
            f"avg_{unit}": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            f"max_{unit}": max(lengths) if lengths else 0,
            f"min_{unit}": min(lengths) if lengths else 0,
        }
