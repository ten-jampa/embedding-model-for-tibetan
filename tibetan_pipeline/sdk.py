"""Jupyter-friendly SDK surface for modular Tibetan pipeline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .embeddings import DEFAULT_MODEL_ID, TextEmbedder, TorchDTypeName
from .normalization import normalize_text
from .pairwise import PairMatch, cosine_similarity_matrix, global_top_k_matches
from .pipeline import resolve_segmenter


@dataclass(slots=True)
class SegmentationView:
    """Notebook-friendly segmentation view for one source text."""

    original_text: str
    normalized_text: str
    source_format: str
    engine_name: str
    segments: list[str]
    spans: list[tuple[int, int]]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "segment_text": text,
            }
            for index, (text, (start, end)) in enumerate(zip(self.segments, self.spans))
        ]
        return pd.DataFrame(rows)


@dataclass(slots=True)
class EmbeddingView:
    """Notebook-friendly embedding view for sentence lists."""

    model_id: str
    device: str
    sentences: list[str]
    embeddings: np.ndarray

    def to_dataframe(self, include_vectors: bool = False) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, sentence in enumerate(self.sentences):
            row: dict[str, object] = {
                "sentence_index": index,
                "sentence_text": sentence,
            }
            if self.embeddings.size:
                row["vector_norm"] = float(np.linalg.norm(self.embeddings[index]))
                if include_vectors:
                    row["embedding"] = self.embeddings[index].tolist()
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PairwiseView:
    """Notebook-friendly pairwise similarity outputs."""

    model_id: str
    device: str
    segments_a: list[str]
    segments_b: list[str]
    similarity_matrix: np.ndarray
    matches: list[PairMatch]

    def topk_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(match) for match in self.matches])


class TibetanResearchSDK:
    """High-level SDK for segmentation, embeddings, and pairwise notebook workflows."""

    def __init__(
        self,
        *,
        engine: str = "botok_ours",
        source_format: str = "unicode",
        botok_cache_dir: str | Path | None = ".cache/botok/dialect_packs",
        min_syllables: int = 4,
        model_id: str = DEFAULT_MODEL_ID,
        batch_size: int = 8,
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
        embedding_progress: Literal["off", "batch", "sentence"] = "off",
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool = False,
    ) -> None:
        self.engine = engine
        self.source_format = source_format
        self.botok_cache_dir = botok_cache_dir
        self.min_syllables = min_syllables
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.embedding_progress = embedding_progress
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self._segmenter = resolve_segmenter(
            engine=engine,
            dialect_pack_dir=botok_cache_dir,
            min_syllables=min_syllables,
        )
        self._embedders: dict[tuple[object, ...], TextEmbedder] = {}

    def segment_text(self, text: str, *, source_format: str | None = None) -> SegmentationView:
        source_format = source_format or self.source_format
        normalized = normalize_text(text, source_format=source_format)
        segmented = self._segmenter.segment(normalized)
        segments = [segment.text for segment in segmented]
        spans = [(segment.start, segment.end) for segment in segmented]
        return SegmentationView(
            original_text=text,
            normalized_text=normalized,
            source_format=source_format,
            engine_name=self._segmenter.engine_name,
            segments=segments,
            spans=spans,
        )

    def embed_sentences(
        self,
        sentences: list[str],
        *,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        is_query: bool = False,
    ) -> EmbeddingView:
        model_id = model_id or self.model_id
        batch_size = batch_size if batch_size is not None else self.batch_size
        device = device or self.device
        embedding_progress = embedding_progress or self.embedding_progress
        torch_dtype = torch_dtype if torch_dtype is not None else self.torch_dtype
        device_map = device_map if device_map is not None else self.device_map
        load_in_8bit = load_in_8bit if load_in_8bit is not None else self.load_in_8bit
        embedder = self._get_embedder(
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
        )
        encoded = embedder.encode_queries(sentences) if is_query else embedder.encode_corpus(sentences)
        return EmbeddingView(
            model_id=encoded.model_id,
            device=device,
            sentences=sentences,
            embeddings=encoded.embeddings,
        )

    def pairwise(
        self,
        text_a: str,
        text_b: str,
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
    ) -> PairwiseView:
        sentences_a = self._segment_text_to_sentences(text_a)
        sentences_b = self._segment_text_to_sentences(text_b)
        return self.pairwise_from_sentences(
            sentences_a,
            sentences_b,
            top_k=top_k,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
        )

    def pairwise_from_sentences(
        self,
        sentences_a: list[str],
        sentences_b: list[str],
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
    ) -> PairwiseView:
        embedding_a = self.embed_sentences(
            sentences_a,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            is_query=True,
        )
        embedding_b = self.embed_sentences(
            sentences_b,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            is_query=False,
        )
        matrix = cosine_similarity_matrix(embedding_a.embeddings, embedding_b.embeddings)
        matches = global_top_k_matches(matrix, sentences_a, sentences_b, top_k)
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=sentences_a,
            segments_b=sentences_b,
            similarity_matrix=matrix,
            matches=matches,
        )

    def pairwise_from_embedding_views(
        self,
        embedding_a: EmbeddingView,
        embedding_b: EmbeddingView,
        *,
        top_k: int = 100,
    ) -> PairwiseView:
        if embedding_a.model_id != embedding_b.model_id:
            raise ValueError("Embedding views must use the same model_id.")
        if embedding_a.embeddings.ndim != 2 or embedding_b.embeddings.ndim != 2:
            raise ValueError("Embedding views must contain rank-2 embedding arrays.")
        if embedding_a.embeddings.shape[0] != len(embedding_a.sentences):
            raise ValueError("Embedding view A row count must match its sentence count.")
        if embedding_b.embeddings.shape[0] != len(embedding_b.sentences):
            raise ValueError("Embedding view B row count must match its sentence count.")

        matrix = cosine_similarity_matrix(embedding_a.embeddings, embedding_b.embeddings)
        matches = global_top_k_matches(matrix, embedding_a.sentences, embedding_b.sentences, top_k)
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=embedding_a.sentences,
            segments_b=embedding_b.sentences,
            similarity_matrix=matrix,
            matches=matches,
        )

    def _segment_text_to_sentences(self, text: str) -> list[str]:
        view = self.segment_text(text)
        return [segment for segment in view.segments if segment.strip()]

    def _get_embedder(
        self,
        *,
        model_id: str,
        batch_size: int,
        device: Literal["auto", "cpu", "mps", "cuda"],
        embedding_progress: Literal["off", "batch", "sentence"],
        torch_dtype: TorchDTypeName | None,
        device_map: str | dict[str, int | str] | None,
        load_in_8bit: bool,
    ) -> TextEmbedder:
        cache_key = (
            model_id,
            device,
            torch_dtype,
            repr(device_map),
            load_in_8bit,
        )
        embedder = self._embedders.get(cache_key)
        if embedder is None:
            embedder = TextEmbedder(
                model_id=model_id,
                batch_size=batch_size,
                normalize_embeddings=True,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
            )
            self._embedders[cache_key] = embedder
        else:
            embedder.batch_size = batch_size
            embedder.embedding_progress = embedding_progress
        return embedder
