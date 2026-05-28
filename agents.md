## Python environment

Canonical environment: `.venv_harpy`

Run Python, tests, lint, and tooling by calling the environment's binaries directly via
their `.venv_harpy/bin/` path rather than sourcing `activate`:

```bash
.venv_harpy/bin/pytest
.venv_harpy/bin/python -m pytest
.venv_harpy/bin/pre-commit run ruff --all-files
```

## Codex config

Repository-local Codex settings for this project live in `.codex/config.toml`.

When checking or updating Codex sandbox, cache, approval, or environment settings for this repo,
use `.codex/config.toml` first rather than `~/.codex/config.toml`.
