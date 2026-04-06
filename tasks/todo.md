# Task Plan

## 2026-04-06 SDK Embedding Loading Optimization

- [x] Work in isolated hidden worktree `sdk-embedding-loading` on branch `fix/sdk-embedding-loading`.
- [x] Cache SDK `TextEmbedder` instances so repeated SDK calls do not reload the model.
- [x] Reuse the SDK's existing segmenter inside `pairwise()` instead of constructing throwaway segmenters.
- [x] Add model loading controls for dtype, device map, and optional 8-bit quantization.
- [x] Update focused unit tests proving embedder reuse, segmenter reuse, and model load kwargs.
- [x] Run the unit test suite and document validation results.

## 2026-04-06 SDK Manual Validation Notebook

- [x] Re-check the implemented SDK fixes and restate the expected user-visible behavior.
- [x] Add `notebooks/03_sdk_starter_v2.ipynb` with end-to-end SDK usage plus manual validation cells for caching, pairwise segmentation reuse, and model-loading controls.
- [x] Validate the notebook JSON structure and document the new notebook in the review notes.

## 2026-04-06 Pairwise From Embeddings API

- [x] Add a notebook-friendly SDK method that computes pairwise similarity from precomputed embedding views without re-embedding.
- [x] Add focused tests proving the new path does not invoke embedding again.
- [x] Update the manual notebook to demonstrate the no-reembedding long-text workflow.

- [x] Create a clean feature branch for the implementation work.
- [x] Create and validate the global `progress-logger` skill.
- [x] Capture the implementation plan in repo task tracking files.
- [x] Build a reusable Tibetan text pipeline package with normalization, segmentation, qualitative review, and embedding stages.
- [x] Add a CLI that runs segmentation-only and segmentation-plus-embedding workflows from files.
- [x] Add tests for normalization, segmentation contracts, review artifact generation, and CLI behavior.
- [x] Verify the pipeline with targeted test runs and document the results.

## 2026-03-16 Conda Environment YAML

- [x] Audit runtime and development dependencies used by the repo.
- [x] Add a project-level conda environment YAML for reproducible setup.
- [x] Update setup docs to prefer the new conda workflow and verify the YAML shape.

## Review

- Added SDK-level embedder caching in [tibetan_pipeline/sdk.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tibetan_pipeline/sdk.py) keyed by heavyweight load settings (`model_id`, `device`, `torch_dtype`, `device_map`, `load_in_8bit`) so changing `batch_size` or progress logging does not force a model reload.
- Reworked SDK pairwise segmentation to reuse the SDK's existing segmenter via `segment_text()` instead of calling the pairwise helper that rebuilds segmenters.
- Added optional embedding load controls in [tibetan_pipeline/embeddings.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tibetan_pipeline/embeddings.py): `torch_dtype`, `device_map`, and `load_in_8bit`, with logic to avoid calling `.to(...)` when quantized or device-mapped loading already owns placement.
- Expanded tests in [tests/test_sdk.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tests/test_sdk.py) and [tests/test_embeddings_device.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tests/test_embeddings_device.py) to prove cache reuse, SDK segmenter reuse, dtype/device-map forwarding, and 8-bit quantization wiring.
- Verified with `conda run -n embedding-tibetan-env python -m unittest discover -s tests -v` in the isolated worktree; all 30 tests passed.
- Added [notebooks/03_sdk_starter_v2.ipynb](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/notebooks/03_sdk_starter_v2.ipynb) with end-to-end SDK cells and explicit manual checks for embedder cache reuse, `pairwise()` segmenter reuse, and forwarded model-loading controls.
- Verified the new notebook file parses as valid notebook JSON (`nbformat=4`, `nbformat_minor=5`).
- Added `pairwise_from_embedding_views(...)` to [tibetan_pipeline/sdk.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tibetan_pipeline/sdk.py) so notebook workflows can reuse precomputed embeddings instead of paying a second embedding pass inside `pairwise()`.
- Expanded [tests/test_sdk.py](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/tests/test_sdk.py) to prove the new API does not call the embedder again.
- Updated [notebooks/03_sdk_starter_v2.ipynb](/Users/ten-jampa/Documents/personal_projects/.worktrees/embedding-model-for-tibetan/sdk-embedding-loading/notebooks/03_sdk_starter_v2.ipynb) with a long-text example that uses the new API and prints the split between embedding time and similarity-only time.
- Re-verified with `conda run -n embedding-tibetan-env python -m unittest discover -s tests -v`; all 31 tests passed.

- Added [environment.yml](/Users/tenzinjampa/Documents/personal-projects/embedding-model-for-tibetan/environment.yml) with `python=3.11`, a repo-local env name (`embedding-tibetan-env`), Jupyter support (`ipykernel`, `jupyterlab`), and a compatible `setuptools<81` pin for `pyewts`.
- Updated [README.md](/Users/tenzinjampa/Documents/personal-projects/embedding-model-for-tibetan/README.md) to make `conda env create -f environment.yml` the default setup flow while keeping a pip fallback.
- Synced [requirements.txt](/Users/tenzinjampa/Documents/personal-projects/embedding-model-for-tibetan/requirements.txt) with actual imports by adding `huggingface_hub`, which is required by `scripts/download_gemma_mitra.py`.
- Verified the real setup with `conda env update -n embedding-tibetan-env -f environment.yml --prune`, `conda run -n embedding-tibetan-env python -m pip install --no-build-isolation botok pyewts`, and `python -m ipykernel install --user --name embedding-tibetan-env`.

- Added a new `tibetan_pipeline` package with pluggable segmentation interfaces, a Botok-backed backend, qualitative review CSV generation, and an optional embedding stage.
- Added `scripts/run_tibetan_pipeline.py` as a thin CLI wrapper and updated `requirements.txt` with `botok` and `pyewts`.
- Verified the code with `python -m unittest discover -s tests -v` and a real Botok smoke test on `data/tibetan_sentences.csv` writing to `output/botok_smoke/segmentation_review.csv`.
- Residual risk: the embedding stage is verified by unit tests and generic backend logic, but I did not run a full real-model smoke test with `buddhist-nlp/gemma-2-mitra-e` because that model is large and expensive to pull during this pass.
- Added Intellexus engine integration through adapters (`botok_intellexus`, `regex_intellexus`) and expanded segmenter resolution and CLI engine choices.
- Added boundary-level pseudo-evaluation metrics (precision/recall/F1) alongside strict sentence exact-match metrics.
- Added `scripts/run_engine_benchmarks.py` and generated per-engine benchmark outputs under `output/benchmarks/` plus a consolidated `comparison_summary.csv` on the 12k/6/3 slice.
