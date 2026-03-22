"""CLI entrypoint for conjoining RKTS line-addressed transliteration files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rkts_ingestion.postprocess import postprocess_conjoined_texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process RKTS raw files into conjoined transliteration text.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("smoke_output/data/raw"),
        help="Directory containing raw RKTS files under rkts/<collection>/<volume>.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smoke_output/data/processed"),
        help="Directory where conjoined files and metadata will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = postprocess_conjoined_texts(
        input_root=args.input_dir.resolve(),
        output_root=args.output_dir.resolve(),
        overwrite=args.overwrite,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
