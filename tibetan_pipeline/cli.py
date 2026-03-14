"""CLI entrypoint for the Tibetan pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embeddings import DEFAULT_MODEL_ID
from .io import load_records
from .pipeline import PipelineArtifacts, TibetanPipeline, resolve_segmenter
from .review import write_review_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Tibetan segmentation and embedding pipeline.")
    parser.add_argument("--input", required=True, help="Input file path (.csv, .tsv, .jsonl, or .txt).")
    parser.add_argument("--output-dir", required=True, help="Output directory for pipeline artifacts.")
    parser.add_argument(
        "--input-format",
        default="unicode",
        choices=["unicode", "wylie"],
        help="Format of the incoming Tibetan text.",
    )
    parser.add_argument(
        "--engine",
        default="botok_ours",
        choices=["botok", "botok_ours", "botok_intellexus", "regex_intellexus"],
        help="Sentence segmentation backend.",
    )
    parser.add_argument(
        "--text-column",
        default="input_text",
        help="Column name containing the source text for CSV/TSV/JSONL inputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick inspection.")
    parser.add_argument(
        "--botok-cache-dir",
        default=".cache/botok/dialect_packs",
        help="Directory where Botok dialect packs should be stored.",
    )
    parser.add_argument("--min-syllables", type=int, default=4, help="Minimum syllables before splitting on weak shad boundaries.")
    parser.add_argument("--embed", action="store_true", help="Run the embedding stage after segmentation.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id for embeddings.")
    return parser


def run(args: argparse.Namespace) -> PipelineArtifacts:
    records = load_records(args.input, text_column=args.text_column, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = resolve_segmenter(
        args.engine,
        dialect_pack_dir=args.botok_cache_dir,
        min_syllables=args.min_syllables,
    )
    pipeline = TibetanPipeline(segmenter)
    results = pipeline.run_segmentation(records, source_format=args.input_format)

    review_path = write_review_artifact(results, output_dir / "segmentation_review.csv")
    artifacts = PipelineArtifacts(review_csv=review_path)
    if args.embed:
        embeddings_npy, metadata_json = pipeline.write_embeddings(
            results,
            output_dir=output_dir,
            model_id=args.model_id,
        )
        artifacts.embeddings_npy = embeddings_npy
        artifacts.embeddings_metadata_json = metadata_json
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
