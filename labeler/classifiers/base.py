"""Abstract base class for labeler classifiers."""

from abc import ABC, abstractmethod


class LabelerClassifier(ABC):
    """
    Interface for classifiers used in the silver-label consensus pipeline.

    Each classifier takes a text string and returns a binary depression label.
    """

    @abstractmethod
    def classify(self, text: str) -> str:
        """
        Classify a single post for signs of depression.

        Args:
            text: The post text (may be original or translated to English,
                  depending on the classifier and language).

        Returns:
            "depressed" or "not depressed"
        """
        pass
