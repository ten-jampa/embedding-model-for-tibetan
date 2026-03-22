from __future__ import annotations

from pathlib import Path
import unittest

from rkts_ingestion.parser import parse_collection_page, parse_info_page, parse_text_line, summarize_text


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class ParserTests(unittest.TestCase):
    def test_parse_collection_page_extracts_volumes_and_source_origin(self) -> None:
        html = (FIXTURES_DIR / "repository_sample.html").read_text(encoding="utf-8")
        entries = parse_collection_page(html)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].collection_code, "KanjurDerge")
        self.assertEqual(entries[0].source_origin, "e-texts based on the Esukhia Input")
        self.assertEqual(entries[0].volume_number, "001")
        self.assertEqual(entries[0].volume_title, "'dul ba ka")
        self.assertTrue(entries[0].text_url.endswith("/repository/KanjurDerge/001.txt"))
        self.assertTrue(entries[0].info_url.endswith("/info.php?fich=KanjurDerge001"))

    def test_parse_info_page_extracts_history(self) -> None:
        html = (FIXTURES_DIR / "info_sample.html").read_text(encoding="utf-8")
        entry = parse_info_page(html)

        self.assertEqual(entry.identifier, "KanjurDerge volume 001")
        self.assertEqual(entry.history_note, "2025.01: E-texts based on the Esukhia input.")

    def test_parse_text_line_splits_addressed_lines(self) -> None:
        parsed = parse_text_line(
            "1b1(rnying rgyud): @@// rgya gang saM skr-i ta'i skad du / ti la kA pra"
        )

        self.assertEqual(parsed.line_ref, "1b1(rnying rgyud)")
        self.assertEqual(parsed.raw_text_body, "@@// rgya gang saM skr-i ta'i skad du / ti la kA pra")
        self.assertEqual(parsed.cleaned_text_body, "rgya gang saM skr-i ta'i skad du / ti la kA pra")
        self.assertFalse(parsed.is_placeholder)
        self.assertFalse(parsed.is_malformed)

    def test_parse_text_line_marks_placeholder_and_malformed_lines(self) -> None:
        placeholder = parse_text_line("3b1(rnying rgyud): xxx")
        malformed = parse_text_line("mkhyen pa'i sku gsung thugs la phyag 'tshal lo//")

        self.assertTrue(placeholder.is_placeholder)
        self.assertFalse(placeholder.is_malformed)
        self.assertTrue(malformed.is_malformed)
        self.assertIsNone(malformed.line_ref)

    def test_summarize_text_detects_line_addressed_shape(self) -> None:
        text = "\n".join(
            [
                "1b1(rnying rgyud): @@// rgya gang saM skr-i ta'i skad du",
                "1b2(rnying rgyud): mkhyen pa'i sku gsung thugs la phyag 'tshal lo//",
                "3b1(rnying rgyud): xxx",
            ]
        )

        stats = summarize_text(text)

        self.assertEqual(stats.bytes_count, len(text.encode("utf-8")))
        self.assertEqual(stats.line_count, 3)
        self.assertEqual(stats.placeholder_line_count, 1)
        self.assertEqual(stats.content_shape, "line_addressed_wylie")


if __name__ == "__main__":
    unittest.main()
