"""CLI tests for the Tibetan pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tibetan_pipeline.cli import run
from tibetan_pipeline.embeddings import EmbeddingResult
from tibetan_pipeline.io import InputRecord
from tibetan_pipeline.pipeline import PipelineArtifacts
from tibetan_pipeline.segmenters.base import BaseSegmenter, Segment


class FakeSegmenter(BaseSegmenter):
    engine_name = "botok"

    def segment(self, text: str) -> list[Segment]:
        midpoint = max(1, len(text) // 2)
        return [
            Segment(text[:midpoint].strip(), 0, midpoint),
            Segment(text[midpoint:].strip(), midpoint, len(text)),
        ]


class CLITests(unittest.TestCase):
    def test_segmentation_only_writes_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nབོད་ཡིག་། ཚིག་གཉིས་།\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="unicode",
                engine="botok",
                text_column="input_text",
                limit=None,
                botok_cache_dir=str(Path(temp_dir) / "cache"),
                min_syllables=1,
                embed=False,
                model_id="unused",
            )

            with patch("tibetan_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                artifacts = run(args)

            self.assertIsInstance(artifacts, PipelineArtifacts)
            self.assertTrue(artifacts.review_csv.exists())
            with artifacts.review_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["engine"], "botok")

    def test_embedding_stage_writes_numpy_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nབོད་ཡིག་། ཚིག་གཉིས་།\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="unicode",
                engine="botok",
                text_column="input_text",
                limit=None,
                botok_cache_dir=str(Path(temp_dir) / "cache"),
                min_syllables=1,
                embed=True,
                model_id="fake/model",
            )

            with patch("tibetan_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                with patch("tibetan_pipeline.pipeline.TextEmbedder.encode", return_value=EmbeddingResult("fake/model", np.ones((2, 3), dtype=np.float32))):
                    artifacts = run(args)

            self.assertTrue(artifacts.embeddings_npy.exists())
            self.assertTrue(artifacts.embeddings_metadata_json.exists())
