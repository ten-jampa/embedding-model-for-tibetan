"""Post-processing for line-addressed RKTS raw text files."""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path

from rkts_ingestion.parser import parse_text_line


POSTPROCESS_MANIFEST_HEADERS = [
    "collection_code",
    "volume_number",
    "input_path",
    "output_path",
    "raw_line_count",
    "retained_line_count",
    "dropped_placeholder_count",
    "dropped_empty_count",
    "output_char_count",
    "output_token_estimate",
]


def process_volume_text(raw_text: str) -> tuple[str, dict[str, int]]:
    retained_chunks: list[str] = []
    raw_line_count = 0
    dropped_placeholder_count = 0
    dropped_empty_count = 0

    for line in raw_text.splitlines():
        raw_line_count += 1
        parsed = parse_text_line(line)
        cleaned = parsed.cleaned_text_body.strip()

        if cleaned == "":
            dropped_empty_count += 1
            continue

        if cleaned.lower() == "xxx":
            dropped_placeholder_count += 1
            continue

        retained_chunks.append(cleaned)

    conjoined_text = " ".join(retained_chunks).strip()
    stats = {
        "raw_line_count": raw_line_count,
        "retained_line_count": len(retained_chunks),
        "dropped_placeholder_count": dropped_placeholder_count,
        "dropped_empty_count": dropped_empty_count,
        "output_char_count": len(conjoined_text),
        "output_token_estimate": len(conjoined_text.split()) if conjoined_text else 0,
    }
    return conjoined_text, stats


def postprocess_conjoined_texts(input_root: Path, output_root: Path, overwrite: bool = False) -> Path:
    raw_base = input_root / "rkts"
    processed_base = output_root / "rkts_conjoined"
    manifest_rows: list[dict[str, str | int]] = []

    raw_files = sorted(raw_base.glob("*/*.txt"))
    for raw_file in raw_files:
        collection_code = raw_file.parent.name
        volume_number = raw_file.stem
        output_file = processed_base / collection_code / f"{volume_number}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        raw_text = raw_file.read_text(encoding="utf-8")
        conjoined_text, stats = process_volume_text(raw_text)

        if overwrite or not output_file.exists():
            output_file.write_text(conjoined_text, encoding="utf-8")

        manifest_rows.append(
            {
                "collection_code": collection_code,
                "volume_number": volume_number,
                "input_path": str(raw_file),
                "output_path": str(output_file),
                **stats,
            }
        )

    manifest_path = output_root / "metadata" / "rkts" / "rkts_conjoined_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=POSTPROCESS_MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path
