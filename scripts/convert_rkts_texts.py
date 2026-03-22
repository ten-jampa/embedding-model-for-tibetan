"""CLI entrypoint for pyewts conversion of conjoined RKTS files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rkts_ingestion.conversion import convert_conjoined_texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert conjoined RKTS transliteration files to Unicode and round-trip transliteration."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("smoke_output/data/processed/rkts_conjoined"),
        help="Directory with conjoined transliteration files under <collection>/<volume>.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smoke_output/data/converted"),
        help="Directory where Unicode, round-trip files, and conversion manifest will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing conversion outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = convert_conjoined_texts(
        input_root=args.input_dir.resolve(),
        output_root=args.output_dir.resolve(),
        overwrite=args.overwrite,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
