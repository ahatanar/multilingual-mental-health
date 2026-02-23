"""Grok API classifier using xAI's OpenAI-compatible endpoint."""

import logging
import time

from openai import OpenAI

from .base import LabelerClassifier

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are evaluating a social media post for signs of depression.

Read the following post carefully and classify whether the author is expressing signs of depression or not.

Respond with ONLY one of these two labels, nothing else:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


class GrokClassifier(LabelerClassifier):
    """
    Depression classifier using xAI Grok API.

    Uses the same prompt structure and response parsing as the evaluation
    pipeline (models/base.py:_parse_response).
    """

    def __init__(self, api_key: str, model: str = "grok-3-mini",
                 max_retries: int = 3, retry_delay: float = 2.0,
                 request_delay: float = 1.0):
        self._client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._request_delay = request_delay

    def classify(self, text: str) -> str:
        prompt = CLASSIFICATION_PROMPT.format(post_text=text)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system",
                         "content": "You are a mental health text classifier. Respond concisely."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                raw = response.choices[0].message.content
                label = self._parse_response(raw)

                # Rate limiting
                time.sleep(self._request_delay)

                if label == "unclear":
                    logger.warning(f"[Grok] Unclear response: {raw!r}")
                    raise ValueError(f"Unclear Grok response: {raw!r}")

                return label

            except Exception as e:
                logger.warning(f"[Grok] Attempt {attempt}/{self._max_retries} failed: {e}")
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * attempt)
                else:
                    raise

    @staticmethod
    def _parse_response(raw_response: str) -> str:
        """Parse Grok response — mirrors ModelProvider._parse_response()."""
        text = raw_response.strip().lower()

        not_depressed_signals = [
            "not depressed", "not_depressed", "no depression",
            "normal", "non-depressed",
        ]
        for signal in not_depressed_signals:
            if signal in text:
                return "not depressed"

        depressed_signals = ["depressed", "depression", "yes"]
        for signal in depressed_signals:
            if signal in text:
                return "depressed"

        return "unclear"
