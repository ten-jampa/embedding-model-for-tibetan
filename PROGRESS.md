# Progress Log

## 2026-03-11 20:52:05 EDT
- Status: started
- Summary: Began implementation of the Tibetan segmentation and embedding baseline pipeline.
- Context: The repo instructions require a clean branch, explicit task tracking, and a running work log so implementation decisions remain visible after code changes.
- Details: Created the feature branch `tibetan-segmentation-pipeline`, verified external identifiers for Botok and pyewts, and set up the global `progress-logger` skill before repo edits.

## 2026-03-11 20:54:45 EDT
- Status: in_progress
- Summary: Confirmed the segmentation architecture should stay pluggable while starting with a Botok-first backend.
- Context: Intellexus treats sentence segmentation as an engine interface with interchangeable backends, which is a better long-term shape than binding the whole pipeline directly to one package.
- Details: Compared the Intellexus regex and Botok segmentation approach against OpenPecha Botok and decided to implement a backend interface with Botok wired first and regex/model-based engines left as extension points.

## 2026-03-11 20:55:19 EDT
- Status: in_progress
- Summary: Hit a packaging issue while installing the Tibetan-specific dependencies.
- Context: The pipeline should use real Botok and Wylie conversion libraries, but `pyewts` failed under build isolation even though the underlying `setuptools/pkg_resources` dependency is available in the environment.
- Details: Initial `python -m pip install botok pyewts` failed because `pyewts` could not import `pkg_resources` during its isolated build step; verified that neither `botok` nor `pyewts` was installed afterward.

## 2026-03-11 20:56:00 EDT
- Status: in_progress
- Summary: Installed the missing Tibetan-specific dependencies with a targeted packaging fix.
- Context: Botok and pyewts should be real runtime dependencies for the first backend rather than deferred TODOs, and fixing the install path is better than coding around a packaging quirk.
- Details: Installed `botok==1.1.2` and `pyewts==0.2.0`, retrying pyewts with `--no-build-isolation` after the initial build-isolation failure.

## 2026-03-11 20:57:00 EDT
- Status: in_progress
- Summary: Added the initial reusable pipeline package, CLI wrapper, and unit tests.
- Context: The code needs to be importable from notebooks and scriptable from the command line, so the package shape comes first and the CLI stays thin.
- Details: Added `tibetan_pipeline/` modules for normalization, I/O, Botok-backed segmentation, review artifact generation, embeddings, and pipeline orchestration; added `scripts/run_tibetan_pipeline.py`, `tests/`, and updated `requirements.txt`.

## 2026-03-11 20:58:00 EDT
- Status: in_progress
- Summary: Verified the new package structure and test coverage before the first real Botok run.
- Context: It is cheaper to catch import and interface mistakes with fast unit tests before triggering Botok dialect-pack setup and a real CLI execution.
- Details: `python -m unittest discover -s tests -v` passed with 8 tests; `python -m compileall tibetan_pipeline scripts/run_tibetan_pipeline.py tests` completed successfully.

## 2026-03-11 20:59:00 EDT
- Status: in_progress
- Summary: Fixed the CLI wrapper so the reusable package resolves correctly from the `scripts/` entrypoint.
- Context: The wrapper should be a thin convenience layer, but it still needs to add the repo root to `sys.path` when executed directly or it fails before the pipeline code is reached.
- Details: Updated `scripts/run_tibetan_pipeline.py` to prepend the project root to `sys.path`, reran unit tests successfully, and retried the Botok smoke test.

## 2026-03-11 21:00:00 EDT
- Status: completed
- Summary: Verified the Botok-backed segmentation pipeline against real repo data.
- Context: Mock-based tests are not enough for a segmentation backend that relies on an external dialect pack and tokenizer behavior; a real run was needed to prove the pipeline works in this environment.
- Details: Ran `python scripts/run_tibetan_pipeline.py --input data/tibetan_sentences.csv --output-dir output/botok_smoke --limit 3 --engine botok --botok-cache-dir .cache/botok/dialect_packs`; Botok downloaded the `general` dialect pack into `.cache/botok/dialect_packs/general` and produced `output/botok_smoke/segmentation_review.csv`.

## 2026-03-11 21:01:00 EDT
- Status: completed
- Summary: Cached the Botok tokenizer inside the segmenter instance and cleaned up task bookkeeping.
- Context: Rebuilding the tokenizer repeatedly would add unnecessary cost during batch usage, and the task tracker should reflect the final verified state without duplicate checklist items.
- Details: Updated `tibetan_pipeline/segmenters/botok.py` to memoize the tokenizer instance and corrected `tasks/todo.md`.

## 2026-03-11 21:07:31 EDT
- Status: completed
- Summary: Added a project-level tmux convention for long-running agent processes.
- Context: Long-lived scripts and shells should be visible to the user and future agents, so the repo now documents that persistent or slow processes belong in clearly named tmux sessions.
- Details: Added `AGENTS.md` at the repo root with rules for tmux session usage and naming, including the same requirement for sub-agents.

## 2026-03-11 21:37:14 EDT
- Status: in_progress
- Summary: Started the pseudo-evaluation workflow for clumped Tibetan text.
- Context: The existing `data/tibetan_sentences.csv` is not real gold, but it is still useful for creating longer contiguous passages and checking how much of the upstream segmentation the Botok pipeline can recover before manual review.
- Details: Inspected the first rows of `data/tibetan_sentences.csv` and confirmed the file is suitable for concatenating neighboring rows into passage-sized clumps for segmentation stress testing.

## 2026-03-11 21:39:00 EDT
- Status: in_progress
- Summary: Added clumping and pseudo-evaluation utilities for upstream-segmented sentence rows.
- Context: We need a practical way to create longer passages from existing rows, run the Botok segmenter on them, and compare recovery against the upstream segmentation without pretending it is true gold.
- Details: Added `tibetan_pipeline/clumping.py`, `tibetan_pipeline/pseudo_eval.py`, `scripts/run_clumped_segmentation_eval.py`, and `tests/test_clumping.py`.

## 2026-03-11 21:41:00 EDT
- Status: completed
- Summary: Verified the clumped pseudo-evaluation workflow on real repo data.
- Context: A small real run is enough to confirm the new workflow produces actionable review artifacts and exposes where Botok splits differently from the upstream sentence rows.
- Details: `python -m unittest discover -s tests -v` passed with 10 tests, and `python scripts/run_clumped_segmentation_eval.py --input data/tibetan_sentences.csv --output-dir output/clumped_eval_smoke --clump-size 4 --stride 4 --limit 12 --engine botok --botok-cache-dir .cache/botok/dialect_packs` produced `clumped_segmentation_review.csv`, `clumped_pseudo_eval.csv`, and a summary with mean source recall `0.9167` and mean predicted precision `0.8333`.

## 2026-03-11 21:45:26 EDT
- Status: in_progress
- Summary: Began the larger clumped segmentation evaluation run.
- Context: The next useful step is a broader pseudo-evaluation sweep over the existing sentence CSV so the user can inspect lower-scoring clumps and decide whether simple heuristics or a learned boundary model is warranted.
- Details: Prepared to run `scripts/run_clumped_segmentation_eval.py` on the full upstream sentence file in a named tmux session `embedding-model-clumped-eval`, writing artifacts under `output/clumped_eval_full`.

## 2026-03-11 21:53:05 EDT
- Status: in_progress
- Summary: Fixed the tmux execution path for the larger evaluation run.
- Context: The first tmux attempt picked up a different Python environment and also exposed that segmentation-only workflows should not import the embedding stack eagerly from `tibetan_pipeline.__init__`.
- Details: Removed eager embedding imports from `tibetan_pipeline/__init__.py`, reran the 10-test suite successfully, and prepared to restart the tmux session using `/opt/homebrew/Caskroom/miniconda/base/envs/embedding-tibetan-env/bin/python`.

## 2026-03-11 21:55:30 EDT
- Status: in_progress
- Summary: Switching the larger evaluation back to direct shell execution.
- Context: The tmux workflow added unnecessary friction for this task, so the simpler and more reliable path is to run the clumped evaluation directly and summarize the outputs in-line.
- Details: Preparing to kill the `embedding-model-clumped-eval` tmux session and rerun the larger clumped segmentation evaluation in the current shell.

## 2026-03-11 22:00:30 EDT
- Status: in_progress
- Summary: Re-scoping the larger evaluation to a capped sample instead of the full sentence CSV.
- Context: `data/tibetan_sentences.csv` contains roughly 2 million rows, so a full-corpus clumped pass is not an interactive review step; it is a batch job and would delay the actual qualitative inspection work.
- Details: After starting the direct full-file pass and confirming it was processing, decided to stop it and rerun with an explicit record limit so we get actionable review artifacts quickly.

## 2026-03-11 22:11:23 EDT
- Status: completed
- Summary: Recorded the dataset-scale lesson explicitly in the repo lessons file.
- Context: The operational mistake was not just the long run itself; it was skipping the scale check that should have happened before choosing between a sample evaluation and a batch job.
- Details: Added a new lesson to `tasks/lessons.md` stating that corpus size must be checked before launching non-trivial pipeline jobs and that interactive evaluation should use an explicit sample limit.

## 2026-03-13 21:06:24 EDT
- Status: in_progress
- Summary: Started implementation of Intellexus engine benchmarking in the existing clumped evaluation pipeline.
- Context: We need apples-to-apples metrics for three engines (`botok_ours`, `botok_intellexus`, `regex_intellexus`) on the same 12k/6/3 slice, with both strict sentence overlap and boundary-level scoring.
- Details: Beginning with engine adapter wiring and CLI/resolver updates, then adding boundary metrics and consolidated benchmark reporting.

## 2026-03-13 21:10:30 EDT
- Status: in_progress
- Summary: Wired Intellexus engines into the pipeline via adapters and added benchmark orchestration.
- Context: Intellexus engine code needs a valid importable package path and a thin adapter layer so all engines can run through the same clumped evaluation harness and emit comparable artifacts.
- Details: Added `intellexus_engine_code/`, created `tibetan_pipeline/segmenters/intellexus.py`, extended engine resolution/CLI choices, added boundary metrics to pseudo-eval outputs, and created `scripts/run_engine_benchmarks.py` for per-engine + consolidated reporting.

## 2026-03-13 21:18:40 EDT
- Status: in_progress
- Summary: Corrected boundary metric computation before finalizing benchmark numbers.
- Context: Raw segment end offsets can include trailing whitespace and artificially depress boundary precision/recall even when sentence boundaries are otherwise aligned.
- Details: Updated `tibetan_pipeline/pseudo_eval.py` to trim trailing whitespace from predicted boundary offsets before scoring; rerunning tests and engine benchmarks with the corrected metric.

## 2026-03-13 21:25:10 EDT
- Status: completed
- Summary: Completed multi-engine benchmark implementation and produced baseline comparison outputs.
- Context: We needed side-by-side numbers for `botok_ours`, `botok_intellexus`, and `regex_intellexus` on the same clump slice with both strict overlap and boundary metrics.
- Details: Added Intellexus adapters, boundary-aware pseudo-eval, benchmark runner script, and consolidated report at `output/benchmarks/comparison_summary.csv`; reran full unit test suite (13 tests passing) and benchmark run on `--limit 12000 --clump-size 6 --stride 3`.

## 2026-03-13 21:20:19 EDT
- Status: completed
- Summary: Consolidated Intellexus code to a single canonical package path.
- Context: Keeping both `intellexus-engine-code/` and `intellexus_engine_code/` duplicates source and creates maintenance risk; long-term best practice is one canonical source of truth.
- Details: Kept `intellexus_engine_code/` as the runtime package and removed duplicate files from `intellexus-engine-code/`; verified runtime references point only to the canonical package.

## 2026-03-13 21:27:40 EDT
- Status: completed
- Summary: Ran all three engines on a fresh holdout slice and generated side-by-side manual review artifacts.
- Context: The first benchmark slice may have been familiar; this run uses a later unseen region of the dataset so manual comparison and baseline conclusions are less biased by prior inspection.
- Details: Created `output/benchmarks_unseen/tibetan_sentences_unseen_12000.csv` from a later section of `data/tibetan_sentences.csv` (12k rows), ran `scripts/run_engine_benchmarks.py` on that input for `botok_ours`, `botok_intellexus`, and `regex_intellexus`, and generated `output/benchmarks_unseen/results/manual_review_side_by_side.csv` (4000 clumps) for direct cross-engine sentence review.

## 2026-03-13 21:23:58 EDT
- Status: in_progress
- Summary: Started holdout benchmarking on unseen text clumps for manual cross-engine review.
- Context: The first benchmark slice used the beginning of the dataset; to reduce familiarity bias we need a later dataset slice and side-by-side outputs for `botok_ours`, `botok_intellexus`, and `regex_intellexus`.
- Details: Preparing a new 12k-row holdout slice from a later section of `data/tibetan_sentences.csv` and running `scripts/run_engine_benchmarks.py` on that input.

## 2026-03-13 21:39:29 EDT
- Status: completed
- Summary: Added and executed one-file end-to-end three-engine comparison on unseen external unicode data.
- Context: The previous clump benchmark depended on an upstream segmented CSV, so we needed a less circular manual-review workflow on raw text from the separate worktree where segmentation assumptions are not inherited from the same upstream process.
- Details: Added `scripts/run_one_file_engine_compare.py` to split raw Tibetan text into shad-delimited units, clump units (`6/3`), run `botok_ours`, `botok_intellexus`, and `regex_intellexus`, and export per-engine review CSVs plus `output/one_file_compare/kanjurderge_001/manual_review_side_by_side.csv`; executed against `.../converted/unicode/KanjurDerge/001.txt` with `unit_limit=1200` producing 400 clumps for manual side-by-side inspection.

## 2026-03-13 22:02:34 EDT
- Status: completed
- Summary: Rewrote project README to document the segmentation stage as a complete, PR-ready deliverable.
- Context: We completed substantial segmentation and benchmarking work across multiple scripts/engines; without a consolidated README, reviewers on `origin/main` would not have a single source of truth for setup, usage, scope, and reproducibility expectations.
- Details: Updated `README.md` with architecture overview, engine descriptions, installation notes, exact CLI commands for segmentation and benchmark workflows, test command, artifact/data policy, and an explicit boundary between the completed segmentation stage and the upcoming pairwise similarity stage.

## 2026-03-13 22:11:35 EDT
- Status: completed
- Summary: Implemented the first end-to-end pairwise similarity stage on top of the segmentation foundation.
- Context: The project goal now shifts from segmentation benchmarking to retrieval readiness, requiring a concrete `Text A` vs `Text B` path that produces ranked sentence-level similarities with reviewable artifacts.
- Details: Added `tibetan_pipeline/pairwise.py` and `scripts/run_pairwise_text_similarity.py` to run segment -> embed -> cosine matrix -> global top-k output (`CSV`, `JSONL`, and run manifest), plus `tests/test_pairwise.py` covering similarity math, ranking behavior, and mocked end-to-end artifact generation.

## 2026-03-16 11:00:00 EDT Add conda environment spec
**Status**: ✅ Complete
**Verification**: Added `environment.yml`, then ran a Python YAML parse/assertion that printed `environment.yml parsed and contains expected conda/pip entries`.
**Notes**: The env uses `python=3.11` for broad ML package compatibility and installs `requirements.txt` with `--no-build-isolation` because this repo has already seen `pyewts` fail under pip build isolation.

## 2026-03-16 11:35:00 EDT Complete real conda env and Jupyter kernel setup
**Status**: ✅ Complete
**Verification**: `conda env update -n embedding-tibetan-env -f environment.yml --prune` completed successfully; `conda run -n embedding-tibetan-env python -m pip install --no-build-isolation botok pyewts` completed successfully; `conda run -n embedding-tibetan-env python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"` installed the kernelspec; final smoke test printed `imports_ok`, and the kernel list now includes `embedding-tibetan-env`.
**Notes**: The final working setup is hybrid: conda handles the main scientific/Jupyter stack, while `botok` and `pyewts` are installed afterward with pip because they are not available on conda-forge and `pyewts` needs a compatible `setuptools` plus `--no-build-isolation`.
