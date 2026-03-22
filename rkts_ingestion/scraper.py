"""RKTS pilot scraper implementation."""

from __future__ import annotations

from csv import DictWriter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from rkts_ingestion.parser import BASE_URL, parse_collection_page, parse_info_page, summarize_text
from rkts_ingestion.targets import PILOT_TARGETS, PilotTarget


COLLECTION_URL_TEMPLATE = BASE_URL + "repository.php?col={collection_code}"
MANIFEST_HEADERS = [
    "collection_code",
    "collection_label",
    "source_family",
    "source_origin",
    "volume_number",
    "volume_label",
    "volume_title",
    "revision_date",
    "text_url",
    "info_url",
    "local_path",
    "bytes",
    "line_count",
    "nonempty_line_count",
    "placeholder_line_count",
    "malformed_line_count",
    "content_shape",
    "history_note",
]


class FetchError(RuntimeError):
    """Raised when an RKTS resource cannot be fetched."""


def fetch_text(url: str, timeout: int = 30) -> str:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError) as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc


def scrape_pilot(output_dir: Path, skip_existing: bool = True) -> Path:
    collection_pages = _load_collection_entries()
    manifest_rows: list[dict[str, str | int]] = []

    for target in PILOT_TARGETS:
        entry = collection_pages[target.collection_code][target.volume_number]
        destination = output_dir / "data" / "raw" / "rkts" / target.collection_code / f"{target.volume_number}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and skip_existing:
            text = destination.read_text(encoding="utf-8")
        else:
            text = fetch_text(entry.text_url)
            destination.write_text(text, encoding="utf-8")

        stats = summarize_text(text)
        history_note = parse_info_page(fetch_text(entry.info_url)).history_note
        manifest_rows.append(
            {
                "collection_code": entry.collection_code,
                "collection_label": entry.collection_label,
                "source_family": entry.source_family,
                "source_origin": entry.source_origin,
                "volume_number": entry.volume_number,
                "volume_label": entry.volume_label,
                "volume_title": entry.volume_title,
                "revision_date": entry.revision_date,
                "text_url": entry.text_url,
                "info_url": entry.info_url,
                "local_path": str(destination.relative_to(output_dir)),
                "bytes": stats.bytes_count,
                "line_count": stats.line_count,
                "nonempty_line_count": stats.nonempty_line_count,
                "placeholder_line_count": stats.placeholder_line_count,
                "malformed_line_count": stats.malformed_line_count,
                "content_shape": stats.content_shape,
                "history_note": history_note,
            }
        )

    manifest_path = output_dir / "data" / "metadata" / "rkts" / "rkts_pilot_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def _load_collection_entries():
    collection_codes = sorted({target.collection_code for target in PILOT_TARGETS})
    collection_entries: dict[str, dict[str, object]] = {}
    for collection_code in collection_codes:
        html = fetch_text(COLLECTION_URL_TEMPLATE.format(collection_code=collection_code))
        entries = parse_collection_page(html)
        collection_entries[collection_code] = {entry.volume_number: entry for entry in entries}
    return collection_entries
