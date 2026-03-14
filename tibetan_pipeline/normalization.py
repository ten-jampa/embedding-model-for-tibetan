"""Input normalization utilities for Tibetan text."""

from __future__ import annotations

import re
import unicodedata

import pyewts

_MULTISPACE_RE = re.compile(r"\s+")
_converter = pyewts.pyewts()


def normalize_text(text: str, source_format: str = "unicode") -> str:
    """Normalize incoming text to Unicode Tibetan."""
    if text is None:
        return ""

    source_format = source_format.lower()
    cleaned = unicodedata.normalize("NFC", text).strip()
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)

    if source_format == "unicode":
        return cleaned
    if source_format == "wylie":
        return unicodedata.normalize("NFC", _converter.toUnicode(cleaned)).strip()

    raise ValueError(f"Unsupported source format: {source_format}")
