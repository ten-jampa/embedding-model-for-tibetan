"""Normalization and I/O tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tibetan_pipeline.io import load_records
from tibetan_pipeline.normalization import normalize_text


class NormalizationTests(unittest.TestCase):
    def test_unicode_passthrough(self) -> None:
        self.assertEqual(normalize_text(" བོད་ཀྱི  "), "བོད་ཀྱི")

    def test_wylie_to_unicode(self) -> None:
        self.assertEqual(normalize_text("bod kyi", source_format="wylie"), "བོད་ཀྱི")

    def test_load_records_uses_first_column_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.csv"
            path.write_text("text\nབོད་ཡིག\n", encoding="utf-8")
            records = load_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "བོད་ཡིག")
