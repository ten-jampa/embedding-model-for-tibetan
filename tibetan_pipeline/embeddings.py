"""Embedding backends for Tibetan sentence lists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_ID = "buddhist-nlp/gemma-2-mitra-e"


@dataclass(slots=True)
class EmbeddingResult:
    """Embedding output with model metadata."""

    model_id: str
    embeddings: np.ndarray


class TextEmbedder:
    """A flexible text embedder supporting sentence-transformers and fallback pooling."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._backend = None
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode a list of texts into dense vectors."""
        if not texts:
            return EmbeddingResult(self.model_id, np.empty((0, 0), dtype=np.float32))

        self._ensure_backend()
        if self._backend == "sentence-transformers":
            embeddings = self._sentence_transformer.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
            return EmbeddingResult(self.model_id, embeddings)

        embeddings = self._transformers_encode(texts)
        return EmbeddingResult(self.model_id, embeddings)

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        try:
            self._sentence_transformer = SentenceTransformer(self.model_id)
            self._backend = "sentence-transformers"
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self._device)
            self._model.eval()
            self._backend = "transformers"

    def _transformers_encode(self, texts: list[str]) -> np.ndarray:
        output_batches: list[np.ndarray] = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            output_batches.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(output_batches)
