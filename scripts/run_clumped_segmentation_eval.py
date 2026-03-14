#!/usr/bin/env python3
"""Build passage clumps from sentence rows and run pseudo-evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tibetan_pipeline.clumping import build_clumped_records
from tibetan_pipeline.io import load_records
from tibetan_pipeline.pipeline import TibetanPipeline, resolve_segmenter
from tibetan_pipeline.pseudo_eval import compare_clump_to_prediction, summarize_pseudo_eval, write_pseudo_eval_csv
from tibetan_pipeline.review import write_review_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pseudo-evaluation on clumped Tibetan passages.")
    parser.add_argument("--input", required=True, help="Input file of upstream segmented sentence rows.")
    parser.add_argument("--output-dir", required=True, help="Output directory for pseudo-eval artifacts.")
    parser.add_argument("--text-column", default="input_text", help="Column containing Tibetan text.")
    parser.add_argument("--input-format", default="unicode", choices=["unicode", "wylie"])
    parser.add_argument(
        "--engine",
        default="botok_ours",
        choices=["botok", "botok_ours", "botok_intellexus", "regex_intellexus"],
    )
    parser.add_argument("--clump-size", type=int, default=5, help="Number of neighboring sentences per synthetic passage.")
    parser.add_argument("--stride", type=int, default=None, help="Stride between clumps; defaults to clump size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit before clumping.")
    parser.add_argument("--botok-cache-dir", default=".cache/botok/dialect_packs")
    parser.add_argument("--min-syllables", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input, text_column=args.text_column, limit=args.limit)
    clumps = build_clumped_records(records, clump_size=args.clump_size, stride=args.stride)

    segmenter = resolve_segmenter(
        args.engine,
        dialect_pack_dir=args.botok_cache_dir,
        min_syllables=args.min_syllables,
    )
    pipeline = TibetanPipeline(segmenter)
    results = pipeline.run_segmentation(clumps, source_format=args.input_format)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = write_review_artifact(results, output_dir / "clumped_segmentation_review.csv")
    pseudo_rows = [
        compare_clump_to_prediction(clump, result)
        for clump, result in zip(clumps, results, strict=False)
    ]
    pseudo_eval_path = write_pseudo_eval_csv(pseudo_rows, output_dir / "clumped_pseudo_eval.csv")
    summary = summarize_pseudo_eval(pseudo_rows)
    (output_dir / "clumped_pseudo_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"review_csv={review_path}")
    print(f"pseudo_eval_csv={pseudo_eval_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
