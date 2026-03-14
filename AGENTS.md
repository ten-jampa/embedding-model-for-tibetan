# Project AGENTS Instructions

This repo follows the global Codex agent instructions in `~/.codex/AGENTS.md` and adds the project-specific rule below.

## Long-Running Commands

- Any script, build, server, training job, watch process, or shell command that may take a long time or persist should be started inside `tmux`.
- Name the `tmux` session clearly so the user can attach to it directly. Prefer names shaped like `repo-purpose`, for example `embedding-model-segmentation`, `embedding-model-train`, or `embedding-model-tests`.
- If a sub-agent starts a long-running process, it must follow the same rule and use its own clearly named `tmux` session.
- For short, bounded commands such as quick file inspection, unit tests, or one-off checks, `tmux` is not required.
- When starting a long-running process in `tmux`, tell the user the session name so they can inspect it themselves.
