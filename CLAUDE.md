## Python environment

Canonical environment: `.venv_harpy`. Use it as-is — do NOT sync, update, or install
into it (e.g. no `uv sync`/`uv run`); the maintainer manages env updates manually.

Run Python, tests, lint, and tooling by calling the environment's binaries directly via
their `.venv_harpy/bin/` path.

```bash
.venv_harpy/bin/pytest
.venv_harpy/bin/python -m pytest
.venv_harpy/bin/pre-commit run ruff --all-files
```

## Claude config

Repository-local Claude Code settings for this project live in `.claude/settings.json`.

When checking or updating Claude cache, permission, or environment settings for this repo,
use `.claude/settings.json` first rather than `~/.claude/settings.json`.
