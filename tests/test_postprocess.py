from __future__ import annotations

from csv import DictReader
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rkts_ingestion.postprocess import postprocess_conjoined_texts, process_volume_text


class PostprocessTests(unittest.TestCase):
    def test_process_volume_text_strips_prefixes_and_joins(self) -> None:
        raw_text = "\n".join(
            [
                "2a1(ka): zhang zhung gi skad du/ se sto lig zhi",
                "2a2(ka): yu sa/ bod skad du/",
            ]
        )
        conjoined, stats = process_volume_text(raw_text)
        self.assertNotIn("2a1(ka):", conjoined)
        self.assertEqual(
            conjoined,
            "zhang zhung gi skad du/ se sto lig zhi yu sa/ bod skad du/",
        )
        self.assertEqual(stats["retained_line_count"], 2)

    def test_process_volume_text_drops_placeholders_and_empty(self) -> None:
        raw_text = "\n".join(
            [
                "1b1(ka): ",
                "1b2(ka): xxx",
                "1b3(ka): @@// ",
                "2a1(ka): real content",
            ]
        )
        conjoined, stats = process_volume_text(raw_text)
        self.assertEqual(conjoined, "real content")
        self.assertEqual(stats["dropped_placeholder_count"], 1)
        self.assertEqual(stats["dropped_empty_count"], 2)

    def test_postprocess_conjoined_texts_writes_outputs_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "input"
            output_root = root / "output"

            file_a = input_root / "rkts" / "BonBkz" / "001.txt"
            file_a.parent.mkdir(parents=True, exist_ok=True)
            file_a.write_text("1a1(ka): alpha\n1a2(ka): beta\n", encoding="utf-8")

            file_b = input_root / "rkts" / "NGBGdg" / "013.txt"
            file_b.parent.mkdir(parents=True, exist_ok=True)
            file_b.write_text("1a1(pa): xxx\n1a2(pa): gamma\n", encoding="utf-8")

            manifest_path = postprocess_conjoined_texts(
                input_root=input_root,
                output_root=output_root,
                overwrite=True,
            )

            out_a = output_root / "rkts_conjoined" / "BonBkz" / "001.txt"
            out_b = output_root / "rkts_conjoined" / "NGBGdg" / "013.txt"
            self.assertEqual(out_a.read_text(encoding="utf-8"), "alpha beta")
            self.assertEqual(out_b.read_text(encoding="utf-8"), "gamma")
            self.assertTrue(manifest_path.exists())

            with manifest_path.open("r", encoding="utf-8") as handle:
                rows = list(DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["collection_code"], "BonBkz")
            self.assertEqual(rows[1]["collection_code"], "NGBGdg")


if __name__ == "__main__":
    unittest.main()
