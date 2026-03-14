# Lessons

- `pyewts` may fail under pip build isolation even when `setuptools` is present; `python -m pip install --no-build-isolation pyewts` works in this environment.
- Botok defaults to a machine-specific cache under `~/Documents/pybo/dialect_packs`; set a repo-local dialect-pack directory such as `.cache/botok/dialect_packs` to keep the backend reproducible inside the workspace.
- Always check corpus size before launching a non-trivial pipeline job; choose an explicit sample limit for interactive evaluation and reserve full-corpus runs for deliberate batch jobs.
- Boundary scoring based on raw segment spans can be misleading if spans include trailing spaces; trim right-side whitespace before computing boundary positions.
