"""Stratified dataset sampler for evaluation."""

import csv
import random
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum word count per post — posts exceeding this are excluded
MAX_WORD_COUNT = 240


class DatasetSampler:
    """
    Loads a CSV dataset and performs stratified sampling.
    
    Filters out posts exceeding MAX_WORD_COUNT, then samples equally
    from each label class so the evaluation set is balanced.
    """

    def __init__(self, csv_path: str, text_col: str = "post", label_col: str = "label",
                 max_words: int = MAX_WORD_COUNT):
        self.csv_path = csv_path
        self.text_col = text_col
        self.label_col = label_col
        self.max_words = max_words
        self.data = self._load_and_filter()

    def _load_and_filter(self) -> List[Dict]:
        """Load CSV and filter out posts exceeding the word limit."""
        all_rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                post = row.get(self.text_col, "").strip()
                label = row.get(self.label_col, "").strip().lower()
                word_count = len(post.split())

                if not post or not label:
                    continue
                if word_count > self.max_words:
                    continue

                all_rows.append({
                    "post": post,
                    "label": label,  # ground truth: "depression" or "normal"
                    "word_count": word_count,
                    **{k: v for k, v in row.items() if k not in (self.text_col, self.label_col)},
                })

        logger.info(f"Loaded {len(all_rows)} posts from {self.csv_path} (after filtering >{self.max_words} words)")
        return all_rows

    def sample(self, n: int = 50, seed: int = 42) -> List[Dict]:
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
        word_counts = [row["word_count"] for row in self.data]
        return {
            "total_posts": len(self.data),
            "label_distribution": dict(label_counts),
            "avg_word_count": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0,
            "min_word_count": min(word_counts) if word_counts else 0,
        }
