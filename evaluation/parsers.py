"""
Language-specific dataset parsers.

Each parser reads raw data files and normalises them into a unified format:
    [{"post": str, "label": "depression" | "normal"}, ...]

Parsed results can be cached to ``data/cleaned/<language>.json`` so that
expensive parsing (Chinese / Spanish) only runs once.
"""

import csv
import json
import logging
import os
import re
import tarfile
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CLEANED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned")


# ── Base class ───────────────────────────────────────────────────────────────

class DatasetParser(ABC):
    """Abstract base for all language parsers."""

    language: str = ""

    @abstractmethod
    def parse(self) -> List[Dict[str, str]]:
        """Return list of {"post": ..., "label": "depression"|"normal"}."""

    # ── Caching helpers ──────────────────────────────────────────────────

    def cache_path(self) -> str:
        return os.path.join(CLEANED_DIR, f"{self.language}.json")

    def load_cache(self) -> Optional[List[Dict[str, str]]]:
        path = self.cache_path()
        if os.path.isfile(path):
            logger.info(f"Loading cached data from {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_cache(self, data: List[Dict[str, str]]) -> str:
        os.makedirs(CLEANED_DIR, exist_ok=True)
        path = self.cache_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached {len(data)} entries to {path}")
        return path

    def get_data(self, reparse: bool = False) -> List[Dict[str, str]]:
        """Return parsed data, using disk cache unless *reparse* is True."""
        if not reparse:
            cached = self.load_cache()
            if cached is not None:
                return cached
        data = self.parse()
        self.save_cache(data)
        return data


# ── English ──────────────────────────────────────────────────────────────────

class EnglishParser(DatasetParser):
    """
    Reads ``data/english/sentiment_tweets3.csv``.

    Columns: ``message to examine``, ``label (depression result)``
    Labels:  1 → depression, 0 → normal
    """

    language = "english"

    def __init__(self, data_dir: str):
        self.csv_path = os.path.join(data_dir, "english", "sentiment_tweets3.csv")

    def parse(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                post = row.get("message to examine", "").strip()
                raw_label = row.get("label (depression result)", "").strip()
                if not post or raw_label not in ("0", "1"):
                    continue
                label = "depression" if raw_label == "1" else "normal"
                rows.append({"post": post, "label": label})
        logger.info(f"[English] Parsed {len(rows)} posts from {self.csv_path}")
        return rows


# ── Arabic ───────────────────────────────────────────────────────────────────

class ArabicParser(DatasetParser):
    """
    Reads ``data/arabic/CairoDep_Datasets.csv``.

    Columns: ``post``, ``label`` (depression / normal)
    """

    language = "arabic"

    def __init__(self, data_dir: str):
        self.csv_path = os.path.join(data_dir, "arabic", "CairoDep_Datasets.csv")

    def parse(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                post = row.get("post", "").strip()
                label = row.get("label", "").strip().lower()
                if not post or label not in ("depression", "normal"):
                    continue
                rows.append({"post": post, "label": label})
        logger.info(f"[Arabic] Parsed {len(rows)} posts from {self.csv_path}")
        return rows


# ── Chinese ──────────────────────────────────────────────────────────────────

class ChineseParser(DatasetParser):
    """
    Reads ``data/chinese/depressed.jsonl`` and ``data/chinese/control.jsonl``.

    Each line is a user object with a ``tweets`` array.  We flatten to
    individual posts, keeping only *original* tweets (``is_origin=True``)
    and stripping HTML tags.
    """

    language = "chinese"

    _HTML_RE = re.compile(r"<[^>]+>")

    def __init__(self, data_dir: str):
        self.depressed_path = os.path.join(data_dir, "chinese", "depressed.jsonl")
        self.control_path = os.path.join(data_dir, "chinese", "control.jsonl")

    def _parse_file(self, path: str, label: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user = json.loads(line)
                for tweet in user.get("tweets", []):
                    if not tweet.get("is_origin", False):
                        continue
                    text = tweet.get("text", "").strip()
                    text = self._HTML_RE.sub("", text).strip()
                    if not text:
                        continue
                    rows.append({"post": text, "label": label})
        return rows

    def parse(self) -> List[Dict[str, str]]:
        dep = self._parse_file(self.depressed_path, "depression")
        logger.info(f"[Chinese] Parsed {len(dep)} depressed posts")
        ctrl = self._parse_file(self.control_path, "normal")
        logger.info(f"[Chinese] Parsed {len(ctrl)} control posts")
        return dep + ctrl


# ── Spanish ──────────────────────────────────────────────────────────────────

class SpanishParser(DatasetParser):
    """
    Reads Spanish tweet data:
      - ``data/spanish/tweets_Español_depresivos.json`` — **tar archive**
        containing JSONL inside.
      - ``data/spanish/tweets_Español_no-depresivos.json`` — plain JSONL.

    Each line is a user object with a ``tweets`` array.  Retweets
    (``RT @…``) are filtered out.
    """

    language = "spanish"

    _RT_RE = re.compile(r"^RT @")

    def __init__(self, data_dir: str):
        self.depressed_path = os.path.join(
            data_dir, "spanish", "tweets_Español_depresivos.json"
        )
        self.non_depressed_path = os.path.join(
            data_dir, "spanish", "tweets_Español_no-depresivos.json"
        )

    def _parse_tar_jsonl(self, tar_path: str, label: str) -> List[Dict[str, str]]:
        """Extract JSONL from a tar archive and parse it."""
        rows: List[Dict[str, str]] = []
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            if f is None:
                return rows
            raw = f.read().decode("utf-8")
            f.close()
            for line in raw.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                user = json.loads(line)
                for tweet in user.get("tweets", []):
                    text = tweet.get("text", "").strip()
                    if not text or self._RT_RE.match(text):
                        continue
                    rows.append({"post": text, "label": label})
        return rows

    def _parse_jsonl(self, path: str, label: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user = json.loads(line)
                for tweet in user.get("tweets", []):
                    text = tweet.get("text", "").strip()
                    if not text or self._RT_RE.match(text):
                        continue
                    rows.append({"post": text, "label": label})
        return rows

    def parse(self) -> List[Dict[str, str]]:
        dep = self._parse_tar_jsonl(self.depressed_path, "depression")
        logger.info(f"[Spanish] Parsed {len(dep)} depressed posts (from tar)")
        non_dep = self._parse_jsonl(self.non_depressed_path, "normal")
        logger.info(f"[Spanish] Parsed {len(non_dep)} non-depressed posts")
        return dep + non_dep


# ── Registry ─────────────────────────────────────────────────────────────────

PARSERS = {
    "english": EnglishParser,
    "arabic": ArabicParser,
    "chinese": ChineseParser,
    "spanish": SpanishParser,
}


def get_parser(language: str, data_dir: str) -> DatasetParser:
    """Return the appropriate parser for *language*."""
    cls = PARSERS.get(language.lower())
    if cls is None:
        raise ValueError(
            f"Unknown language: {language}. "
            f"Supported: {list(PARSERS.keys())}"
        )
    return cls(data_dir)
