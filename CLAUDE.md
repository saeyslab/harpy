## Python environment
Canonical environment: `.venv_harpy`. Use it as-is — do NOT sync, update, or install
into it (e.g. no `uv sync`/`uv run`); the maintainer manages env updates manually.

Activate `.venv_harpy` explicitly before running Python, tests, lint, or tooling. Because
shell state does not persist between commands, chain the activation in the same call:

```bash
source .venv_harpy/bin/activate && pytest
source .venv_harpy/bin/activate && ruff check .
```

## Claude config
Repository-local Claude Code settings for this project live in `.claude/settings.json`.

When checking or updating Claude cache, permission, or environment settings for this repo,
use `.claude/settings.json` first rather than `~/.claude/settings.json`.
