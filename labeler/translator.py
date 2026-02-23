"""Google Translate wrapper for Chinese/Spanish -> English translation."""

import asyncio
import logging
import time
from typing import Dict, List

from googletrans import Translator

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


class CachedTranslator:
    """
    Translates text using Google Translate with caching and retry.

    Used for both Chinese -> English and Spanish -> English.
    XLM-RoBERTa handles Spanish natively, but Grok needs English for both.
    """

    def __init__(self):
        self._translator = Translator()
        self._cache: Dict[str, str] = {}

    def translate(self, text: str, src: str = "zh-cn", dest: str = "en") -> str:
        """
        Translate text with caching and retry on failure.

        Args:
            text: Source text to translate.
            src: Source language code.
            dest: Destination language code.

        Returns:
            Translated text string.
        """
        cache_key = f"{src}:{dest}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                coro = self._translator.translate(text, src=src, dest=dest)
                result = asyncio.get_event_loop().run_until_complete(coro)
                translated = result.text
                self._cache[cache_key] = translated
                return translated
            except Exception as e:
                logger.warning(f"[Translator] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    # Re-create translator instance (googletrans can get stale)
                    self._translator = Translator()
                else:
                    raise

    def load_cache(self, entries: List[Dict], src: str = "zh-cn") -> None:
        """Restore translation cache from checkpoint accepted entries."""
        for entry in entries:
            translated = entry.get("translated")
            post = entry.get("post")
            if translated and post and translated != post:
                cache_key = f"{src}:en:{post}"
                self._cache[cache_key] = translated
