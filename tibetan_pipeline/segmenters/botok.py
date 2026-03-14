"""Botok-backed sentence segmentation."""

from __future__ import annotations

from pathlib import Path

from botok import Config, WordTokenizer

from .base import (
    CONTINUATORS,
    TERMINATORS,
    TIBETAN_DOUBLE_SHAD,
    TIBETAN_SHAD,
    TER_TSHEG,
    TSHEG,
    BaseSegmenter,
    Segment,
)


class BotokSegmenter(BaseSegmenter):
    """Sentence segmentation using Botok tokenization plus shad heuristics."""

    engine_name = "botok_ours"

    def __init__(
        self,
        min_syllables: int = 4,
        dialect_pack_dir: str | Path | None = None,
        tokenizer: WordTokenizer | None = None,
    ) -> None:
        self.min_syllables = min_syllables
        self.dialect_pack_dir = Path(dialect_pack_dir or ".cache/botok/dialect_packs")
        self._tokenizer = tokenizer

    def segment(self, text: str) -> list[Segment]:
        if not text.strip():
            return []

        tokenizer = self._tokenizer or self._build_tokenizer()
        tokens = tokenizer.tokenize(text)
        segments: list[Segment] = []
        current_parts: list[str] = []
        current_start = 0
        cursor = 0
        last_meaningful = ""

        for token in tokens:
            token_text = token.text
            current_parts.append(token_text)
            delimiter = self._is_delimiter(token_text)
            if not delimiter and not token_text.isspace():
                last_meaningful = token_text

            cursor += len(token_text)
            if delimiter and self._should_split("".join(current_parts), token_text, last_meaningful):
                segment_text = "".join(current_parts).strip()
                if segment_text:
                    segments.append(Segment(segment_text, current_start, cursor))
                current_parts = []
                current_start = cursor

        tail = "".join(current_parts).strip()
        if tail:
            segments.append(Segment(tail, current_start, len(text)))
        return segments

    def _build_tokenizer(self) -> WordTokenizer:
        if self._tokenizer is None:
            config = Config(base_path=self.dialect_pack_dir)
            self._tokenizer = WordTokenizer(config=config)
        return self._tokenizer

    @staticmethod
    def _is_delimiter(token_text: str) -> bool:
        return any(mark in token_text for mark in (TIBETAN_SHAD, TIBETAN_DOUBLE_SHAD, TER_TSHEG))

    def _should_split(self, current_text: str, token_text: str, previous_word: str) -> bool:
        if TIBETAN_DOUBLE_SHAD in token_text or token_text.count(TER_TSHEG) >= 2:
            return True

        previous_word = previous_word.strip().rstrip(TSHEG).rstrip(TIBETAN_SHAD).rstrip(TIBETAN_DOUBLE_SHAD)
        if previous_word in CONTINUATORS:
            return False
        if previous_word in TERMINATORS:
            return True
        return self.count_syllables(current_text) >= self.min_syllables
