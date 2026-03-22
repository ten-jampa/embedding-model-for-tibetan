from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rkts_ingestion.scraper import scrape_pilot


COLLECTION_PAGES = {
    "http://www.rkts.org/etexts/repository.php?col=BonBkz": """
        <td>e-texts produced with BDRC's OCR application</td>
        <a href="repository/BonBkz/001.txt">vol. 001</a> (1) <sub>[rev. 28 Apr 2025]</sub><sup><a href="info.php?fich=BonBkz001"></a></sup>
        <a href="repository/BonBkz/050.txt">vol. 050</a> (50) <sub>[rev. 25 Mar 2025]</sub><sup><a href="info.php?fich=BonBkz050"></a></sup>
    """,
    "http://www.rkts.org/etexts/repository.php?col=KanjurDerge": """
        <td>e-texts based on the Esukhia Input</td>
        <a href="repository/KanjurDerge/001.txt">vol. 001</a> ('dul ba ka) <sub>[rev. 18 Feb 2025]</sub><sup><a href="info.php?fich=KanjurDerge001"></a></sup>
        <a href="repository/KanjurDerge/045.txt">vol. 045</a> (mdo sde ka) <sub>[rev. 04 Feb 2025]</sub><sup><a href="info.php?fich=KanjurDerge045"></a></sup>
    """,
    "http://www.rkts.org/etexts/repository.php?col=NGBGdg": """
        <td>e-texts produced with BDRC's OCR application</td>
        <a href="repository/NGBGdg/001.txt">vol. 001</a> (ka) <sub>[rev. 11 Mar 2025]</sub><sup><a href="info.php?fich=NGBGdg001"></a></sup>
        <a href="repository/NGBGdg/013.txt">vol. 013</a> (pa) <sub>[rev. 11 Mar 2025]</sub><sup><a href="info.php?fich=NGBGdg013"></a></sup>
    """,
    "http://www.rkts.org/etexts/repository.php?col=TanjurDerge": """
        <td>e-texts based on the ACIP/ALL Input</td>
        <a href="repository/TanjurDerge/001.txt">vol. 001</a> (bstod tshogs (ka)) <sub>[rev. 05 Feb 2025]</sub><sup><a href="info.php?fich=TanjurDerge001"></a></sup>
        <a href="repository/TanjurDerge/173.txt">vol. 173</a> (mdo 'grel (nge)) <sub>[rev. 05 Feb 2025]</sub><sup><a href="info.php?fich=TanjurDerge173"></a></sup>
    """,
}


def fake_fetch(url: str, timeout: int = 30) -> str:
    if url in COLLECTION_PAGES:
        return COLLECTION_PAGES[url]
    if "info.php?fich=" in url:
        identifier = url.rsplit("=", 1)[-1]
        return f"<h1>History of the e-texts {identifier}</h1>2025.01: sample history.<br>"
    if url.endswith(".txt"):
        return "1b1(rnying rgyud): @@// sample line\n1b2(rnying rgyud): xxx\n"
    raise AssertionError(f"Unexpected URL {url}")


class ScraperTests(unittest.TestCase):
    def test_scrape_pilot_writes_manifest_and_raw_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            with patch("rkts_ingestion.scraper.fetch_text", side_effect=fake_fetch):
                manifest_path = scrape_pilot(output_dir=output_dir, skip_existing=False)

            self.assertTrue(manifest_path.exists())
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn("collection_code,collection_label,source_family,source_origin", manifest_text)
            self.assertIn("KanjurDerge", manifest_text)
            self.assertIn("BonBkz", manifest_text)
            raw_file = output_dir / "data" / "raw" / "rkts" / "KanjurDerge" / "001.txt"
            self.assertTrue(raw_file.exists())
            self.assertIn("sample line", raw_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
