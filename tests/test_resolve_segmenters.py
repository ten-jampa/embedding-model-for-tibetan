"""Tests for segmenter resolver engine mapping."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tibetan_pipeline.pipeline import resolve_segmenter


class ResolveSegmentersTests(unittest.TestCase):
    def test_botok_alias_maps_to_ours(self) -> None:
        with patch("tibetan_pipeline.pipeline.BotokSegmenter", return_value="ours") as mock_cls:
            resolved = resolve_segmenter("botok", min_syllables=2, dialect_pack_dir=".cache/x")
        self.assertEqual(resolved, "ours")
        mock_cls.assert_called_once_with(min_syllables=2, dialect_pack_dir=".cache/x")

    def test_intellexus_botok_maps_to_adapter(self) -> None:
        with patch("tibetan_pipeline.pipeline.IntellexusBotokAdapter", return_value="intellexus-botok") as mock_cls:
            resolved = resolve_segmenter("botok_intellexus", min_syllables=3, dialect_pack_dir=".cache/y")
        self.assertEqual(resolved, "intellexus-botok")
        mock_cls.assert_called_once_with(min_syllables=3, dialect_pack_dir=".cache/y")

    def test_intellexus_regex_maps_to_adapter(self) -> None:
        with patch("tibetan_pipeline.pipeline.IntellexusRegexAdapter", return_value="intellexus-regex") as mock_cls:
            resolved = resolve_segmenter("regex_intellexus", min_syllables=5, dialect_pack_dir=".cache/z")
        self.assertEqual(resolved, "intellexus-regex")
        mock_cls.assert_called_once_with(min_syllables=5)
