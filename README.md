# Tibetan Text Pipeline

This repository provides an end-to-end Tibetan research pipeline for:
- sentence segmentation (`botok_ours` default)
- Gemma-Mitra embeddings (`buddhist-nlp/gemma-2-mitra-e`)
- pairwise text-to-text sentence similarity (`A x B`) with global top-k matches
- notebook-first experimentation via `TibetanResearchSDK`

## Project Layout
- `tibetan_pipeline/`: core package (normalization, segmenters, embeddings, pairwise, SDK)
- `scripts/`: runnable workflows (`run_tibetan_pipeline.py`, `run_pairwise_text_similarity.py`, etc.)
- `notebooks/01_research_sdk_starter.ipynb`: starter notebook for modular testing
- `tests/`: `unittest` suite
- `output/`: local run artifacts (git-ignored)

## Install
Preferred conda setup:
```bash
conda env create -f environment.yml
conda run -n embedding-tibetan-env python -m pip install --no-build-isolation botok pyewts
conda run -n embedding-tibetan-env python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"
conda activate embedding-tibetan-env
```

If you prefer pip in an existing environment:
```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"
```

Optional but recommended for faster first run:
```bash
python scripts/download_gemma_mitra.py
```

## Core Workflows

### 1) Segmentation-only pipeline
```bash
python scripts/run_tibetan_pipeline.py \
  --input data/your_input.csv \
  --output-dir output/segmentation_smoke \
  --engine botok_ours \
  --input-format unicode
```

### 2) Pairwise text similarity (two .txt files)
```bash
python scripts/run_pairwise_text_similarity.py \
  --text-a path/to/text_a.txt \
  --text-b path/to/text_b.txt \
  --output-dir output/pairwise_run \
  --engine botok_ours \
  --input-format unicode \
  --model-id buddhist-nlp/gemma-2-mitra-e \
  --device cpu \
  --top-k 100 \
  --embedding-progress batch
```

Outputs:
- `topk_pairs.csv`
- `topk_pairs.jsonl`
- `run_manifest.json`
- optional `similarity_matrix.npy` (`--save-similarity-npy`)

## Notebook SDK
`TibetanResearchSDK` supports segmentation, embeddings, and pairwise analysis in Jupyter.

```python
from tibetan_pipeline import TibetanResearchSDK

sdk = TibetanResearchSDK(
    engine="botok_ours",
    device="cpu",
    embedding_progress="batch",  # off | batch | sentence
)

seg = sdk.segment_text(text)
emb_q = sdk.embed_sentences(seg.segments, is_query=True)
view = sdk.pairwise(text_a, text_b, top_k=20, embedding_progress="sentence")
```

## Embedding Backend Notes
For `buddhist-nlp/gemma-2-mitra-e`, the backend follows model-card retrieval behavior:
- query/corpus asymmetric encoding (`encode_queries` vs `encode_corpus`)
- last non-padding token pooling from final hidden state
- L2-normalized vectors for cosine similarity via dot product

## Tests
```bash
python -m unittest discover -s tests -v
```

## Troubleshooting
- HF warning about unauthenticated requests: set `HF_TOKEN` for better rate limits.
- MPS OOM on Apple Silicon: rerun with `--device cpu` for stability on large runs.
