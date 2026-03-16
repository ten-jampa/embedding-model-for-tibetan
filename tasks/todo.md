# Task Plan

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
