"""Adapters for Intellexus segmentation engines."""

from __future__ import annotations

from pathlib import Path

from intellexus_engine_code.botok_engine import BotokSegmenter as IntellexusBotokCore
from intellexus_engine_code.regex_engine import RegexSegmenter as IntellexusRegexCore

from .base import BaseSegmenter, Segment


class IntellexusBotokAdapter(BaseSegmenter):
    """Adapter from Intellexus Botok engine to our BaseSegmenter contract."""

    engine_name = "botok_intellexus"

    def __init__(
        self,
        min_syllables: int = 4,
        dialect_pack_dir: str | Path | None = None,
    ) -> None:
        if dialect_pack_dir is not None:
            # Intellexus engine constructs WordTokenizer() directly. Patch Botok's
            # default cache path so it stays inside this repo for reproducibility.
            import botok.config as botok_config

            botok_config.DEFAULT_BASE_PATH = Path(dialect_pack_dir)
        self._core = IntellexusBotokCore(min_syllables=min_syllables)

    def segment(self, text: str) -> list[Segment]:
        return [
            Segment(text=segment_text, start=start_idx, end=end_idx)
            for segment_text, start_idx, end_idx in self._core.segment_with_indices(text)
        ]


class IntellexusRegexAdapter(BaseSegmenter):
    """Adapter from Intellexus regex engine to our BaseSegmenter contract."""

    engine_name = "regex_intellexus"

    def __init__(self, min_syllables: int = 4) -> None:
        self._core = IntellexusRegexCore(min_syllables=min_syllables)

    def segment(self, text: str) -> list[Segment]:
        return [
            Segment(text=segment_text, start=start_idx, end=end_idx)
            for segment_text, start_idx, end_idx in self._core.segment_with_indices(text)
        ]
