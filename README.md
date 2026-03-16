# Tibetan Text Pipeline (Segmentation Stage)

This repository now contains a reusable Tibetan text pipeline with a completed **segmentation stage** and benchmarking workflow.

Current phase delivers:
- Normalization (`unicode` and `wylie`)
- Pluggable sentence segmentation backends
- Qualitative review artifacts
- Clumped pseudo-evaluation and multi-engine benchmarking
- Optional embedding export for segmented sentences

The next phase (separate scope) is pairwise text-to-text sentence similarity retrieval.

## Main Components

### Core package
- `tibetan_pipeline/normalization.py`: text normalization
- `tibetan_pipeline/segmenters/`: backend interface + engines
- `tibetan_pipeline/pipeline.py`: orchestration for segmentation (+ optional embeddings)
- `tibetan_pipeline/review.py`: review CSV writer
- `tibetan_pipeline/clumping.py`: synthetic clump builder from sentence rows
- `tibetan_pipeline/pseudo_eval.py`: pseudo metrics and summary utilities

### Engine backends
- `botok_ours`: Botok-backed segmenter implemented in this repo
- `botok_intellexus`: Intellexus Botok adapter
- `regex_intellexus`: Intellexus regex adapter

### Entry scripts
- `scripts/run_tibetan_pipeline.py`: segmentation and optional embeddings
- `scripts/run_clumped_segmentation_eval.py`: single-engine clumped pseudo-eval
- `scripts/run_engine_benchmarks.py`: multi-engine clumped benchmark comparison
- `scripts/run_one_file_engine_compare.py`: one-file side-by-side engine comparison
- `scripts/run_pairwise_text_similarity.py`: two-text sentence similarity with top-k outputs

### Jupyter SDK
- `tibetan_pipeline.sdk.TibetanResearchSDK`: high-level notebook API for modular experimentation
  - `segment_text(...)`
  - `embed_sentences(...)`
  - `pairwise(...)`
  - `pairwise_from_sentences(...)`
- Notebook starter: `notebooks/01_research_sdk_starter.ipynb`

## Installation

```bash
python -m pip install -r requirements.txt
```

Notes:
- `botok` and `pyewts` are required for Tibetan segmentation and Wylie conversion.
- Botok dialect packs are cached to `.cache/botok/dialect_packs` by default.

## Quick Start

### 1) Run segmentation on an input file

```bash
python scripts/run_tibetan_pipeline.py \
  --input data/your_input.csv \
  --output-dir output/segmentation_smoke \
  --input-format unicode \
  --engine botok_ours \
  --text-column input_text \
  --limit 100 \
  --botok-cache-dir .cache/botok/dialect_packs
```

Output: `output/segmentation_smoke/segmentation_review.csv`

### 2) Run segmentation + embeddings

```bash
python scripts/run_tibetan_pipeline.py \
  --input data/your_input.csv \
  --output-dir output/segmentation_embed \
  --input-format unicode \
  --engine botok_ours \
  --embed \
  --model-id buddhist-nlp/gemma-2-mitra-e
```

Outputs:
- `segmentation_review.csv`
- `embeddings.npy`
- `embeddings_metadata.json`

### 3) Run clumped pseudo-evaluation (single engine)

```bash
python scripts/run_clumped_segmentation_eval.py \
  --input data/tibetan_sentences.csv \
  --output-dir output/clumped_eval_smoke \
  --engine botok_ours \
  --clump-size 6 \
  --stride 3 \
  --limit 12000 \
  --botok-cache-dir .cache/botok/dialect_packs
```

Outputs:
- `clumped_segmentation_review.csv`
- `clumped_pseudo_eval.csv`
- `clumped_pseudo_eval_summary.json`

### 4) Run multi-engine benchmark comparison

```bash
python scripts/run_engine_benchmarks.py \
  --input data/tibetan_sentences.csv \
  --output-dir output/benchmarks \
  --engines botok_ours botok_intellexus regex_intellexus \
  --clump-size 6 \
  --stride 3 \
  --limit 12000 \
  --botok-cache-dir .cache/botok/dialect_packs
```

Output root includes:
- Per-engine review + pseudo-eval artifacts
- `comparison_summary.csv` across engines

### 5) Compare engines on one raw external text file

```bash
python scripts/run_one_file_engine_compare.py \
  --input-file /path/to/raw_text.txt \
  --output-dir output/one_file_compare/example \
  --engines botok_ours botok_intellexus regex_intellexus \
  --clump-size 6 \
  --stride 3 \
  --unit-limit 1200
```

Outputs:
- Per-engine `one_file_review.csv`
- `manual_review_side_by_side.csv`
- `run_manifest.json`

## Testing

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

Current suite covers normalization, segmenter behavior contracts, clumping/pseudo-eval basics, CLI wiring, engine resolver mapping, pairwise similarity utilities, and SDK behavior.

## Data and Artifacts Policy

- Large datasets in `data/` and generated outputs in `output/` are intentionally git-ignored.
- Code and reproducible scripts are versioned.
- If reviewers need benchmark outputs, provide a curated review pack (small CSV/JSON files) under `eval/`.

## Scope Boundary

This README documents the segmentation stage only.

Planned next stage:
- Two-text sentence embedding similarity
- Full `A x B` similarity matrix
- Top-k `(i, j)` sentence pair retrieval using `buddhist-nlp/gemma-2-mitra-e`
