from __future__ import annotations

from csv import DictReader
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rkts_ingestion.conversion import (
    convert_conjoined_texts,
    convert_unicode_to_wylie,
    convert_wylie_to_unicode,
)


class ConversionTests(unittest.TestCase):
    def test_basic_wylie_to_unicode(self) -> None:
        unicode_text = convert_wylie_to_unicode("bod kyi")
        self.assertEqual(unicode_text, "བོད་ཀྱི")

    def test_roundtrip_from_unicode(self) -> None:
        wylie_text = convert_unicode_to_wylie("བོད་ཀྱི")
        self.assertEqual(wylie_text, "bod kyi")

    def test_convert_conjoined_texts_writes_outputs_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "input"
            output_root = root / "output"

            file_a = input_root / "BonBkz" / "001.txt"
            file_a.parent.mkdir(parents=True, exist_ok=True)
            file_a.write_text("bod kyi chos", encoding="utf-8")

            file_b = input_root / "NGBGdg" / "013.txt"
            file_b.parent.mkdir(parents=True, exist_ok=True)
            file_b.write_text("bka' bstan", encoding="utf-8")

            manifest_path = convert_conjoined_texts(input_root=input_root, output_root=output_root, overwrite=True)

            unicode_a = output_root / "unicode" / "BonBkz" / "001.txt"
            roundtrip_b = output_root / "wylie_roundtrip" / "NGBGdg" / "013.txt"
            self.assertTrue(unicode_a.exists())
            self.assertTrue(roundtrip_b.exists())
            self.assertIn("བོད", unicode_a.read_text(encoding="utf-8"))
            self.assertIn("bka'", roundtrip_b.read_text(encoding="utf-8"))

            with manifest_path.open("r", encoding="utf-8") as handle:
                rows = list(DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["collection_code"], "BonBkz")
            self.assertIn(rows[0]["conversion_status"], {"ok", "warnings"})


if __name__ == "__main__":
    unittest.main()
