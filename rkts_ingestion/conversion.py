"""Bidirectional pyewts conversion for conjoined RKTS transliteration files."""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path
import re

import pyewts


CONVERSION_MANIFEST_HEADERS = [
    "collection_code",
    "volume_number",
    "input_path",
    "unicode_output_path",
    "roundtrip_output_path",
    "input_char_count",
    "unicode_char_count",
    "roundtrip_char_count",
    "warning_count",
    "warnings_sample",
    "conversion_status",
]

SUSPECT_UNICODE_PATTERN = re.compile(r"[A-Za-z0-9@#+]")


def convert_wylie_to_unicode(text: str, converter: pyewts.pyewts | None = None) -> str:
    conv = converter or pyewts.pyewts()
    return conv.toUnicode(text)


def convert_unicode_to_wylie(text: str, converter: pyewts.pyewts | None = None) -> str:
    conv = converter or pyewts.pyewts()
    return conv.toWylie(text)


def convert_conjoined_texts(input_root: Path, output_root: Path, overwrite: bool = False) -> Path:
    converter = pyewts.pyewts()
    input_files = sorted(input_root.glob("*/*.txt"))

    unicode_base = output_root / "unicode"
    roundtrip_base = output_root / "wylie_roundtrip"
    manifest_rows: list[dict[str, str | int]] = []

    for input_file in input_files:
        collection_code = input_file.parent.name
        volume_number = input_file.stem
        unicode_output_file = unicode_base / collection_code / f"{volume_number}.txt"
        roundtrip_output_file = roundtrip_base / collection_code / f"{volume_number}.txt"
        unicode_output_file.parent.mkdir(parents=True, exist_ok=True)
        roundtrip_output_file.parent.mkdir(parents=True, exist_ok=True)

        wylie_text = input_file.read_text(encoding="utf-8")
        warnings: list[str] = []
        unicode_text = ""
        roundtrip_text = ""

        try:
            unicode_text = convert_wylie_to_unicode(wylie_text, converter=converter)
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"wylie_to_unicode_error:{exc.__class__.__name__}")

        if unicode_text:
            if SUSPECT_UNICODE_PATTERN.search(unicode_text):
                warnings.append("unicode_contains_ascii_tokens")
            try:
                roundtrip_text = convert_unicode_to_wylie(unicode_text, converter=converter)
            except Exception as exc:  # pragma: no cover - defensive path
                warnings.append(f"unicode_to_wylie_error:{exc.__class__.__name__}")

        if overwrite or not unicode_output_file.exists():
            unicode_output_file.write_text(unicode_text, encoding="utf-8")
        if overwrite or not roundtrip_output_file.exists():
            roundtrip_output_file.write_text(roundtrip_text, encoding="utf-8")

        warning_count = len(warnings)
        manifest_rows.append(
            {
                "collection_code": collection_code,
                "volume_number": volume_number,
                "input_path": str(input_file),
                "unicode_output_path": str(unicode_output_file),
                "roundtrip_output_path": str(roundtrip_output_file),
                "input_char_count": len(wylie_text),
                "unicode_char_count": len(unicode_text),
                "roundtrip_char_count": len(roundtrip_text),
                "warning_count": warning_count,
                "warnings_sample": " | ".join(warnings[:3]),
                "conversion_status": "warnings" if warning_count else "ok",
            }
        )

    manifest_path = output_root / "metadata" / "rkts" / "rkts_conversion_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=CONVERSION_MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path
