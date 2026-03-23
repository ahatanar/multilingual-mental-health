"""Checkpoint manager for labeling pipeline progress."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase1", "labeler_progress")


class CheckpointManager:
    """
    Saves and resumes labeling progress to avoid re-processing posts.

    Checkpoint file: data/labeler_progress/<language>_checkpoint.json
    """

    def __init__(self, language: str):
        self.language = language
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._path = os.path.join(CHECKPOINT_DIR, f"{language}_checkpoint.json")
        self.processed_indices: Set[int] = set()
        self.accepted: List[Dict] = []
        self.depressed_count: int = 0
        self.not_depressed_count: int = 0
        self.total_processed: int = 0
        self.disagreements: int = 0

    def load(self) -> bool:
        """Load checkpoint from disk. Returns True if checkpoint was found."""
        if not os.path.isfile(self._path):
            return False
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.processed_indices = set(data.get("processed_indices", []))
            self.accepted = data.get("accepted", [])
            self.depressed_count = data.get("depressed_count", 0)
            self.not_depressed_count = data.get("not_depressed_count", 0)
            self.total_processed = data.get("total_processed", 0)
            self.disagreements = data.get("disagreements", 0)
            logger.info(
                f"[{self.language}] Resumed checkpoint: "
                f"{self.depressed_count} depressed, "
                f"{self.not_depressed_count} not depressed, "
                f"{self.total_processed} processed"
            )
            return True
        except Exception as e:
            logger.warning(f"[{self.language}] Failed to load checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save current progress to disk."""
        data = {
            "language": self.language,
            "processed_indices": sorted(self.processed_indices),
            "accepted": self.accepted,
            "depressed_count": self.depressed_count,
            "not_depressed_count": self.not_depressed_count,
            "total_processed": self.total_processed,
            "disagreements": self.disagreements,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_result(self, index: int, accepted: bool,
                   entry: Optional[dict] = None, label: Optional[str] = None) -> None:
        """
        Record a processed post.

        Args:
            index: The index of the post in the shuffled source list.
            accepted: Whether both classifiers agreed.
            entry: The full entry dict to store (only if accepted).
            label: The consensus label ("depressed" or "not depressed").
        """
        self.processed_indices.add(index)
        self.total_processed += 1

        if accepted and entry and label:
            self.accepted.append(entry)
            if label == "depressed":
                self.depressed_count += 1
            else:
                self.not_depressed_count += 1
        elif not accepted:
            self.disagreements += 1

    def is_done(self, target: int = 250) -> bool:
        """Check if we have enough posts of each class."""
        return self.depressed_count >= target and self.not_depressed_count >= target

    def is_processed(self, index: int) -> bool:
        """Check if a post index has already been processed."""
        return index in self.processed_indices
