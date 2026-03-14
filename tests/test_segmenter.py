"""Segmenter contract tests."""

from __future__ import annotations

import unittest

from tibetan_pipeline.segmenters.botok import BotokSegmenter


class Token:
    def __init__(self, text: str) -> None:
        self.text = text


class Tokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def tokenize(self, text: str) -> list[Token]:
        return [Token(piece) for piece in self._pieces]


class SegmenterTests(unittest.TestCase):
    def test_empty_text_returns_no_segments(self) -> None:
        segmenter = BotokSegmenter(tokenizer=Tokenizer([]))
        self.assertEqual(segmenter.segment(""), [])

    def test_double_shad_creates_sentence_boundary(self) -> None:
        tokenizer = Tokenizer(["བོད་", "ཡིག་", "༎", "གཉིས་", "ཀ་", "།"])
        segmenter = BotokSegmenter(min_syllables=1, tokenizer=tokenizer)
        segments = segmenter.segment("unused")

        self.assertEqual([segment.text for segment in segments], ["བོད་ཡིག་༎", "གཉིས་ཀ་།"])

    def test_continuator_blocks_weak_split(self) -> None:
        tokenizer = Tokenizer(["བོད་", "དང", "།", "ཡིག་", "།"])
        segmenter = BotokSegmenter(min_syllables=1, tokenizer=tokenizer)
        segments = segmenter.segment("unused")

        self.assertEqual(len(segments), 1)
