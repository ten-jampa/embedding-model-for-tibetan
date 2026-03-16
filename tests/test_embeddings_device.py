"""Device selection tests for embedding backend."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tibetan_pipeline.embeddings import DEFAULT_QUERY_INSTRUCTION, TextEmbedder, _resolve_torch_device


class EmbeddingDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(_resolve_torch_device("auto"), "cuda")

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(_resolve_torch_device("auto"), "mps")

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(_resolve_torch_device("auto"), "cpu")

    def test_explicit_cpu_always_valid(self) -> None:
        self.assertEqual(_resolve_torch_device("cpu"), "cpu")

    def test_explicit_unavailable_device_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_torch_device("cuda")

        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_torch_device("mps")

    def test_sentence_transformer_receives_selected_device(self) -> None:
        with patch("tibetan_pipeline.embeddings.SentenceTransformer") as mock_st:
            embedder = TextEmbedder(model_id="fake/model", device="cpu")
            embedder._ensure_backend()
            mock_st.assert_called_once_with("fake/model", device="cpu")

    def test_query_format_uses_model_card_template(self) -> None:
        embedder = TextEmbedder(model_id="fake/model", device="cpu")
        formatted = embedder._format_query("བོད་ཡིག")
        self.assertEqual(
            formatted,
            f"<instruct>{DEFAULT_QUERY_INSTRUCTION}\n<query>བོད་ཡིག",
        )


if __name__ == "__main__":
    unittest.main()
