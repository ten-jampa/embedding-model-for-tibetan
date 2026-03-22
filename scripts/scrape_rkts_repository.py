"""CLI entrypoint for the RKTS pilot scraper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rkts_ingestion.scraper import scrape_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape a small pilot set from the RKTS e-text repository.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where raw downloads and the manifest will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = scrape_pilot(output_dir=args.output_dir.resolve(), skip_existing=not args.overwrite)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
