# Progress Log

## 2026-03-11 21:44:20 EDT
- Status: started
- Summary: Began the RKTS pilot ingestion task in a dedicated git worktree.
- Context: Another agent is using the main checkout, so this work needs to stay isolated in a separate worktree and avoid relying on uncommitted files from the original directory.
- Details: Created the `rkts-pilot-ingestion` branch in `../embedding-model-for-tibetan-rkts-pilot` and verified the committed baseline only contains `.gitignore`.

## 2026-03-11 21:44:20 EDT
- Status: in_progress
- Summary: Scaffolded the minimal RKTS pilot ingestion project structure inside the worktree.
- Context: The committed baseline was effectively empty, so this task needed its own parser, scraper, tests, and task-tracking files without pulling in the main checkout's uncommitted state.
- Details: Added `rkts_ingestion/` with parser and scraper modules, `scripts/scrape_rkts_repository.py`, fixture-backed tests under `tests/`, and fresh `tasks/` files for this worktree.

## 2026-03-11 22:07:23 EDT
- Status: in_progress
- Summary: Fixed the collection-page source-origin parser and verified the live RKTS scrape.
- Context: RKTS collection pages use inconsistent legacy table markup, so the parser needed to handle both clean fixture HTML and the live site structure without breaking tests.
- Details: Updated `rkts_ingestion/parser.py` to try multiple source-origin patterns, reran `python -m unittest discover -s tests -v`, and reran `python scripts/scrape_rkts_repository.py --output-dir smoke_output --overwrite` successfully against the live site.

## 2026-03-11 22:07:23 EDT
- Status: completed
- Summary: Completed the RKTS pilot ingestion implementation in the isolated worktree.
- Context: The goal was to leave a small, reviewable archive of real RKTS source files plus provenance metadata while keeping all writes out of the main checkout used by the other agent.
- Details: The live run produced 8 raw files under `smoke_output/data/raw/rkts/` and `smoke_output/data/metadata/rkts/rkts_pilot_manifest.csv`, with source-origin labels, revision dates, and line-shape statistics populated.

## 2026-03-13 20:11:23 EDT
- Status: in_progress
- Summary: Started implementing the conjoined-text post-processing stage for the RKTS pilot corpus.
- Context: The raw RKTS files are line-addressed transliteration records, and downstream conversion/review needs derived text where structural line references are removed while token boundaries are preserved.
- Details: Preparing a new local-only post-processing module, CLI entrypoint, and tests that consume `smoke_output/data/raw/rkts` and emit conjoined outputs plus a summary manifest.

## 2026-03-13 20:12:51 EDT
- Status: in_progress
- Summary: Added the post-processing implementation and test coverage for conjoined transliteration outputs.
- Context: This stage needs to be deterministic and auditable, so the implementation writes a derived text tree plus a manifest with dropped-line statistics while preserving raw files.
- Details: Added `rkts_ingestion/postprocess.py`, `scripts/postprocess_rkts_texts.py`, and `tests/test_postprocess.py` with unit and temporary-directory integration checks.

## 2026-03-13 20:13:59 EDT
- Status: in_progress
- Summary: Verified post-processing behavior and generated conjoined outputs for the full pilot dataset.
- Context: The new stage must prove both transformation correctness and practical usability on real files, not only synthetic tests.
- Details: Reran `python -m unittest discover -s tests -v` (9 passing tests) and executed `python scripts/postprocess_rkts_texts.py --input-dir smoke_output/data/raw --output-dir smoke_output/data/processed --overwrite`, producing 8 conjoined files and a manifest at `smoke_output/data/processed/metadata/rkts/rkts_conjoined_manifest.csv`.

## 2026-03-13 20:13:59 EDT
- Status: completed
- Summary: Completed the conjoined-text post-processing layer for RKTS pilot files.
- Context: This closes the requested “one more level of processing” while keeping transliteration intact for a later dedicated `pyewts` conversion step.
- Details: Output files are under `smoke_output/data/processed/rkts_conjoined/` with per-volume drop/retention stats captured in the new manifest.

## 2026-03-13 21:16:17 EDT
- Status: in_progress
- Summary: Started the bidirectional `pyewts` conversion stage for conjoined transliteration outputs.
- Context: The current pipeline stops at conjoined transliteration, but manual review and downstream usage also need Tibetan Unicode and round-trip conversion artifacts.
- Details: Confirmed `pyewts` is not installed in the active environment and prepared to add a dedicated conversion module, CLI, and tests.

## 2026-03-13 21:27:58 EDT
- Status: in_progress
- Summary: Implemented and validated a dedicated `pyewts` conversion stage using the `embedding-tibetan-env` conda environment.
- Context: The conversion pipeline needs to be executable in the project environment and include per-file warning visibility without failing the full run on noisy inputs.
- Details: Added `rkts_ingestion/conversion.py`, `scripts/convert_rkts_texts.py`, `tests/test_conversion.py`, and dependency pin `requirements.txt`; ran `conda run -n embedding-tibetan-env python -m unittest discover -s tests -v` (12 passing tests) and a full conversion run over 8 conjoined files.

## 2026-03-13 21:27:58 EDT
- Status: completed
- Summary: Completed bidirectional Wylie↔Unicode conversion outputs for the pilot corpus.
- Context: This closes the requested pyewts pipeline step and provides both Unicode text for downstream use and round-trip transliteration for quality checks.
- Details: Generated `smoke_output/data/converted/unicode/*/*.txt`, `smoke_output/data/converted/wylie_roundtrip/*/*.txt`, and `smoke_output/data/converted/metadata/rkts/rkts_conversion_manifest.csv`.
