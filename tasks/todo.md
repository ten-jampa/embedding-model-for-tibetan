# Task Plan

- [x] Create a separate git worktree for the RKTS pilot so the main checkout remains untouched.
- [x] Scaffold minimal project files needed for RKTS ingestion in the new worktree.
- [x] Implement a fetch-only RKTS scraper with pilot targets, metadata extraction, and raw-file preservation.
- [x] Add tests for collection parsing, info-page parsing, and line-shape parsing.
- [x] Run targeted verification including unit tests and live smoke scrapes, then document outcomes.
- [x] Add a post-processing stage that converts line-addressed raw files into conjoined transliteration text outputs.
- [x] Add tests for conjoined-text transformation and post-process manifest generation.
- [x] Verify post-processing against live pilot files and record resulting artifacts.
- [x] Add a dedicated `pyewts` conversion stage that produces Unicode and round-trip transliteration outputs.
- [x] Add conversion tests and run the full test suite in `embedding-tibetan-env`.
- [x] Run full conversion over the 8 conjoined pilot volumes and capture conversion metadata.

## Review

- Added a small stdlib-only RKTS ingestion package with pilot targets, collection/info parsing, line-shape parsing, and a fetch-only scraper CLI.
- Verified fixture-backed parsing with `python -m unittest discover -s tests -v` passing all 6 tests.
- Verified a live scrape of 8 RKTS volumes into `smoke_output/data/raw/rkts/` and `smoke_output/data/metadata/rkts/rkts_pilot_manifest.csv`.
- Added a conjoined-text processor and CLI that remove line refs, drop empty/placeholder lines, and write processed outputs plus `rkts_conjoined_manifest.csv`.
- Verified the full suite with `python -m unittest discover -s tests -v` (9 passing tests) and a real post-process run writing 8 processed files under `smoke_output/data/processed/rkts_conjoined/`.
- Added `pyewts` conversion module + CLI producing both Tibetan Unicode and round-trip transliteration files, with warning-aware conversion manifest reporting.
- Verified end-to-end in the requested conda env via `conda run -n embedding-tibetan-env python -m unittest discover -s tests -v` (12 passing tests).
- Completed full conversion run for all 8 pilot files to `smoke_output/data/converted/`.
- Residual risk: conversion warnings are heuristic (`unicode_contains_ascii_tokens`) and indicate candidates for manual cleanup, not definitive conversion failure.
