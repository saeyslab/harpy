## Python environment

Canonical environment: `.venv`. Use it as-is — do NOT sync, update, or install
into it (e.g. no `uv sync`/`uv run`); the maintainer manages env updates manually.


Run Python, tests, lint, and tooling by calling the environment's binaries directly via
their `.venv/bin/` path rather than sourcing `activate`:

```bash
.venv/bin/pytest
.venv/bin/python -m pytest
.venv/bin/pre-commit run ruff --all-files
```

## Codex config

Repository-local Codex settings for this project live in `.codex/config.toml`.

When checking or updating Codex sandbox, cache, approval, or environment settings for this repo,
use `.codex/config.toml` first rather than `~/.codex/config.toml`.
