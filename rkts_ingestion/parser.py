"""Parsers and statistics helpers for RKTS repository pages and text files."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import urljoin


BASE_URL = "http://www.rkts.org/etexts/"

SOURCE_FAMILY_BY_COLLECTION = {
    "KanjurDerge": "Kanjur",
    "TanjurDerge": "Tanjur",
    "NGBGdg": "Old Tantra",
    "BonBkz": "Bon Canon",
}

COLLECTION_LABELS = {
    "KanjurDerge": "D Derge Kanjur",
    "TanjurDerge": "D Derge Tanjur",
    "NGBGdg": "Gdg Derge rnying rgyud",
    "BonBkz": "Bkz Bon Katen",
}

COLLECTION_SOURCE_PATTERNS = (
    re.compile(r"<td>\s*(e-texts .*?)</td>", re.IGNORECASE | re.DOTALL),
    re.compile(r"(e-texts .*?)(?:<td width=50>|Go to the collection)", re.IGNORECASE | re.DOTALL),
)
VOLUME_PATTERN = re.compile(
    r'<a href="(?P<href>repository/(?P<collection>[^/]+)/(?P<volume>\d+)\.txt)"[^>]*>'
    r"(?P<label>[^<]+)</a>\s*"
    r"\((?P<title>.*?)\)\s*"
    r"<sub>\[rev\. (?P<revision>[^\]]+)\]</sub>\s*"
    r'<sup><a href="(?P<info_href>info\.php\?fich=[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
HISTORY_PATTERN = re.compile(
    r"<h1>History of the e-texts (?P<identifier>[^<]+)</h1>(?P<history>.*?)<br>",
    re.IGNORECASE | re.DOTALL,
)
LINE_PATTERN = re.compile(r"^(?P<line_ref>[^:]+):\s*(?P<body>.*)$")
LEADING_WRAPPER_PATTERN = re.compile(r"^(?:[@#/ ]+)+")


@dataclass(frozen=True)
class VolumeEntry:
    collection_code: str
    collection_label: str
    source_family: str
    source_origin: str
    volume_number: str
    volume_label: str
    volume_title: str
    revision_date: str
    text_url: str
    info_url: str


@dataclass(frozen=True)
class InfoEntry:
    identifier: str
    history_note: str


@dataclass(frozen=True)
class ParsedTextLine:
    original_line: str
    line_ref: str | None
    raw_text_body: str
    cleaned_text_body: str
    is_placeholder: bool
    is_malformed: bool


@dataclass(frozen=True)
class TextStats:
    bytes_count: int
    line_count: int
    nonempty_line_count: int
    placeholder_line_count: int
    malformed_line_count: int
    content_shape: str


def strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return " ".join(unescape(text).split())


def parse_collection_page(html: str, base_url: str = BASE_URL) -> list[VolumeEntry]:
    source_origin = "unknown"
    for pattern in COLLECTION_SOURCE_PATTERNS:
        source_origin_match = pattern.search(html)
        if source_origin_match:
            source_origin = strip_tags(source_origin_match.group(1))
            break

    entries: list[VolumeEntry] = []
    for match in VOLUME_PATTERN.finditer(html):
        collection_code = match.group("collection")
        entries.append(
            VolumeEntry(
                collection_code=collection_code,
                collection_label=COLLECTION_LABELS.get(collection_code, collection_code),
                source_family=SOURCE_FAMILY_BY_COLLECTION.get(collection_code, "unknown"),
                source_origin=source_origin,
                volume_number=match.group("volume"),
                volume_label=strip_tags(match.group("label")),
                volume_title=strip_tags(match.group("title")),
                revision_date=strip_tags(match.group("revision")),
                text_url=urljoin(base_url, match.group("href")),
                info_url=urljoin(base_url, match.group("info_href")),
            )
        )
    return entries


def parse_info_page(html: str) -> InfoEntry:
    match = HISTORY_PATTERN.search(html)
    if not match:
        raise ValueError("Unable to parse RKTS info page")
    return InfoEntry(
        identifier=strip_tags(match.group("identifier")),
        history_note=strip_tags(match.group("history")),
    )


def parse_text_line(line: str) -> ParsedTextLine:
    stripped_line = line.rstrip("\n")
    if not stripped_line.strip():
        return ParsedTextLine(
            original_line=stripped_line,
            line_ref=None,
            raw_text_body="",
            cleaned_text_body="",
            is_placeholder=False,
            is_malformed=False,
        )

    match = LINE_PATTERN.match(stripped_line)
    if not match:
        placeholder = stripped_line.strip().lower() == "xxx"
        return ParsedTextLine(
            original_line=stripped_line,
            line_ref=None,
            raw_text_body=stripped_line.strip(),
            cleaned_text_body=_clean_text_body(stripped_line),
            is_placeholder=placeholder,
            is_malformed=not placeholder,
        )

    raw_text_body = match.group("body").strip()
    cleaned_text_body = _clean_text_body(raw_text_body)
    is_placeholder = cleaned_text_body.lower() == "xxx" or cleaned_text_body == ""
    return ParsedTextLine(
        original_line=stripped_line,
        line_ref=match.group("line_ref").strip(),
        raw_text_body=raw_text_body,
        cleaned_text_body=cleaned_text_body,
        is_placeholder=is_placeholder,
        is_malformed=False,
    )


def summarize_text(text: str) -> TextStats:
    encoded = text.encode("utf-8")
    parsed_lines = [parse_text_line(line) for line in text.splitlines()]
    line_count = len(parsed_lines)
    nonempty_line_count = sum(1 for item in parsed_lines if item.original_line.strip())
    placeholder_line_count = sum(1 for item in parsed_lines if item.is_placeholder)
    malformed_line_count = sum(1 for item in parsed_lines if item.is_malformed)
    addressed_line_count = sum(1 for item in parsed_lines if item.line_ref is not None)
    content_shape = (
        "line_addressed_wylie" if line_count and addressed_line_count / max(line_count, 1) >= 0.7 else "unknown"
    )
    return TextStats(
        bytes_count=len(encoded),
        line_count=line_count,
        nonempty_line_count=nonempty_line_count,
        placeholder_line_count=placeholder_line_count,
        malformed_line_count=malformed_line_count,
        content_shape=content_shape,
    )


def _clean_text_body(raw_text_body: str) -> str:
    cleaned = LEADING_WRAPPER_PATTERN.sub("", raw_text_body.strip())
    return cleaned.strip()
