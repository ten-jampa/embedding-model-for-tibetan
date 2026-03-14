"""Shared sentence segmenter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

TIBETAN_SHAD = "\u0F0D"
TIBETAN_DOUBLE_SHAD = "\u0F0E"
TER_TSHEG = "\u0F14"
TSHEG = "\u0F0B"

TERMINATORS = {
    "གོ",
    "ངོ",
    "དོ",
    "ནོ",
    "བོ",
    "མོ",
    "འོ",
    "རོ",
    "ལོ",
    "སོ",
    "ཏོ",
    "ཐོ",
}

CONTINUATORS = {
    "དང",
    "ནས",
    "ཏེ",
    "སྟེ",
    "ཀྱང",
    "ཡང",
    "འང",
    "ཞིང",
    "ཤིང",
    "ཅིང",
    "བཞིན",
}


@dataclass(slots=True)
class Segment:
    """A segmented sentence span."""

    text: str
    start: int
    end: int


class BaseSegmenter(ABC):
    """Base contract for sentence segmentation engines."""

    engine_name = "base"

    @abstractmethod
    def segment(self, text: str) -> list[Segment]:
        """Return segmented spans for the provided text."""

    @staticmethod
    def count_syllables(text: str) -> int:
        return text.count(TSHEG)
