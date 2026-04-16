"""
discovery.py — Crawl results/phase2/ and build a registry of all result files.

Each file is represented as a ResultFileMeta dataclass containing everything
needed to load and identify it without reading the file itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ── canonical names ──────────────────────────────────────────────────────────

_MODEL_FOLDER_ALIASES: dict[str, str] = {
    "deepseek-r1-0528-qwen3-8b": "deepseek",
    "meta-llama-3.1-8b-instruct": "llama",
    "gemma": "gemma",
    "gemini": "gemini",
    "openai": "openai",
    "claude": "claude",
}

_MODEL_PREFIX_ALIASES: dict[str, str] = {
    "deepseek-local": "deepseek",
    "deepseek": "deepseek",
    "llama": "llama",
    "gemma": "gemma",
    "gemini": "gemini",
    "openai": "openai",
    "claude": "claude",
}

_ONLINE_MODELS = {"gemini", "openai", "claude"}
_LOCAL_MODELS = {"llama", "deepseek", "gemma"}

# ── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ResultFileMeta:
    path: Path
    experiment: int          # 1, 2, or 3
    model: str               # canonical: gemini / openai / claude / llama / deepseek / gemma
    model_type: str          # "online" or "local"
    language: str            # arabic / urdu / chinese
    timestamp: str           # e.g. 20260330_022308
    model_folder: str        # raw folder name, for reference


def _normalise_model(raw: str) -> str:
    """Return canonical model name from a raw folder or file-prefix string."""
    raw_lower = raw.lower()
    # Try exact match first
    if raw_lower in _MODEL_PREFIX_ALIASES:
        return _MODEL_PREFIX_ALIASES[raw_lower]
    # Try folder alias
    if raw_lower in _MODEL_FOLDER_ALIASES:
        return _MODEL_FOLDER_ALIASES[raw_lower]
    # Fuzzy: check if any key is a substring
    for alias, canonical in {**_MODEL_FOLDER_ALIASES, **_MODEL_PREFIX_ALIASES}.items():
        if alias in raw_lower:
            return canonical
    return raw_lower  # fallback: use as-is


def _model_type(canonical: str) -> str:
    if canonical in _ONLINE_MODELS:
        return "online"
    if canonical in _LOCAL_MODELS:
        return "local"
    return "unknown"


# ── filename parsing ─────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-zA-Z0-9_\-]+?)_(?P<lang>arabic|urdu|chinese)_(?P<ts>\d{8}_\d{6})\.json$",
    re.IGNORECASE,
)


def _parse_filename(stem_with_ext: str) -> tuple[str, str, str] | None:
    """Return (raw_prefix, language, timestamp) or None."""
    m = _FILENAME_RE.match(stem_with_ext)
    if not m:
        return None
    return m.group("prefix"), m.group("lang").lower(), m.group("ts")


# ── directory traversal ──────────────────────────────────────────────────────

def _iter_exp1_exp2(exp_dir: Path, experiment: int) -> Iterator[ResultFileMeta]:
    """Walk experiment1/ or experiment2/ which have Online/Local Models sub-dirs."""
    for type_dir in exp_dir.iterdir():
        if not type_dir.is_dir():
            continue
        type_lower = type_dir.name.lower()
        if "online" in type_lower:
            declared_type = "online"
        elif "local" in type_lower:
            declared_type = "local"
        else:
            declared_type = "unknown"

        for model_dir in type_dir.iterdir():
            if not model_dir.is_dir():
                continue
            folder_canonical = _normalise_model(model_dir.name)

            for json_file in model_dir.glob("*.json"):
                parsed = _parse_filename(json_file.name)
                if parsed is None:
                    continue
                raw_prefix, lang, ts = parsed
                prefix_canonical = _normalise_model(raw_prefix)
                # Prefer folder-derived name; fall back to prefix
                canonical = folder_canonical if folder_canonical != raw_prefix.lower() else prefix_canonical

                yield ResultFileMeta(
                    path=json_file,
                    experiment=experiment,
                    model=canonical,
                    model_type=declared_type if declared_type != "unknown" else _model_type(canonical),
                    language=lang,
                    timestamp=ts,
                    model_folder=model_dir.name,
                )


def _iter_exp3(exp_dir: Path) -> Iterator[ResultFileMeta]:
    """Walk experiment3/ which is flat (no model sub-dirs)."""
    for json_file in exp_dir.glob("*.json"):
        parsed = _parse_filename(json_file.name)
        if parsed is None:
            continue
        raw_prefix, lang, ts = parsed
        canonical = _normalise_model(raw_prefix)

        yield ResultFileMeta(
            path=json_file,
            experiment=3,
            model=canonical,
            model_type=_model_type(canonical),
            language=lang,
            timestamp=ts,
            model_folder="",
        )


# ── public API ───────────────────────────────────────────────────────────────

def discover(results_root: Path) -> list[ResultFileMeta]:
    """
    Recursively discover all result JSON files under `results_root`.

    Returns a sorted list of ResultFileMeta objects.
    Skips files whose names don't match the expected pattern and logs them.
    """
    phase2_root = results_root / "phase2"
    if not phase2_root.exists():
        raise FileNotFoundError(f"Expected results dir not found: {phase2_root}")

    found: list[ResultFileMeta] = []

    for exp_dir in sorted(phase2_root.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment"):
            continue
        try:
            exp_num = int(exp_dir.name.replace("experiment", ""))
        except ValueError:
            print(f"[discovery] Skipping unknown dir: {exp_dir.name}")
            continue

        if exp_num in (1, 2, 3):
            found.extend(_iter_exp1_exp2(exp_dir, exp_num))
        else:
            print(f"[discovery] Unknown experiment number {exp_num}, skipping.")

    found.sort(key=lambda m: (m.experiment, m.model, m.language))
    return found


def print_discovery_report(files: list[ResultFileMeta]) -> None:
    """Print a human-readable summary of what was discovered."""
    print("\n" + "=" * 60)
    print("  DISCOVERY REPORT")
    print("=" * 60)
    print(f"  Total files found: {len(files)}")

    by_exp: dict[int, list[ResultFileMeta]] = {}
    for f in files:
        by_exp.setdefault(f.experiment, []).append(f)

    for exp in sorted(by_exp):
        entries = by_exp[exp]
        models = sorted({e.model for e in entries})
        langs = sorted({e.language for e in entries})
        print(f"\n  Experiment {exp}: {len(entries)} file(s)")
        print(f"    Models   : {', '.join(models)}")
        print(f"    Languages: {', '.join(langs)}")
        for e in entries:
            print(f"    • [{e.model_type:6}] {e.model:10} | {e.language:8} | {e.timestamp}")

    print("=" * 60 + "\n")
