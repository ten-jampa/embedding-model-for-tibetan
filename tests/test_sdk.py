"""Tests for the notebook-friendly research SDK."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tibetan_pipeline.embeddings import EmbeddingResult
from tibetan_pipeline.segmenters.base import BaseSegmenter, Segment
from tibetan_pipeline.sdk import TibetanResearchSDK


class FakeSegmenter(BaseSegmenter):
    engine_name = "botok_ours"

    def segment(self, text: str) -> list[Segment]:
        return [
            Segment("ཀ་།", 0, 2),
            Segment("ཁ་།", 3, 5),
        ]


class SDKTests(unittest.TestCase):
    def test_segment_text_returns_view_and_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(engine="botok_ours")
            view = sdk.segment_text("ཀ་། ཁ་།")

        self.assertEqual(view.engine_name, "botok_ours")
        self.assertEqual(view.segments, ["ཀ་།", "ཁ་།"])
        df = view.to_dataframe()
        self.assertEqual(list(df.columns), ["segment_index", "start", "end", "segment_text"])
        self.assertEqual(len(df), 2)

    def test_embed_sentences_passes_device_and_returns_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=2)
        with patch("tibetan_pipeline.sdk.TextEmbedder.encode", return_value=EmbeddingResult("fake/model", np.ones((2, 3), dtype=np.float32))):
            view = sdk.embed_sentences(["a", "b"])

        self.assertEqual(view.model_id, "fake/model")
        self.assertEqual(view.device, "cpu")
        self.assertEqual(view.embeddings.shape, (2, 3))
        df = view.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("vector_norm", df.columns)

    def test_pairwise_from_sentences_returns_ranked_dataframe(self) -> None:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            sdk = TibetanResearchSDK(device="cpu", model_id="fake/model", batch_size=1)
        with patch(
            "tibetan_pipeline.sdk.TextEmbedder.encode",
            side_effect=[
                EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
                EmbeddingResult("fake/model", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
            ],
        ):
            view = sdk.pairwise_from_sentences(["a0", "a1"], ["b0", "b1"], top_k=2)

        self.assertEqual(view.similarity_matrix.shape, (2, 2))
        topk_df = view.topk_dataframe()
        self.assertEqual(len(topk_df), 2)
        self.assertEqual(topk_df.iloc[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
