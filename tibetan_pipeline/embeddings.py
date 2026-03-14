"""Embedding backends for Tibetan sentence lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
    ) -> None:
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.device = device
        self._backend = None
        self._tokenizer = None
        self._model = None
        self._device = _resolve_torch_device(device)

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
            self._sentence_transformer = SentenceTransformer(self.model_id, device=self._device)
            self._backend = "sentence-transformers"
        except Exception as exc:
            if _is_mps_oom(exc):
                raise RuntimeError(
                    "MPS out of memory while loading embedding model. "
                    "Rerun with device='cpu' (CLI: --device cpu)."
                ) from exc
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


def _resolve_torch_device(preferred: Literal["auto", "cpu", "mps", "cuda"] = "auto") -> str:
    """Pick the best available PyTorch device for inference."""
    if preferred not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {preferred}")

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_mps_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "mps backend out of memory" in message or ("mps" in message and "out of memory" in message)
