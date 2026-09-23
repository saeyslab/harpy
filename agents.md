## Python environment

Canonical environment: `.venv`. Use it as-is — do NOT sync, update, or install
into it (e.g. no `uv sync`/`uv run`); the maintainer manages env updates manually.

Run Python, tests, lint, and tooling by calling the environment's binaries directly via
their `.venv/bin/` path rather than sourcing `activate`:

```bash
.venv/bin/pytest
.venv/bin/python -m pytest
.venv/bin/pre-commit run --files path/to/changed_file.py
```

## Test scope

Run only the focused unit tests directly affected by a change.

Do not run the full test suite by default. Run it only when:

- the user explicitly requests it; or
- the change is sufficiently broad that focused tests cannot provide reasonable coverage, in which case ask the user first.

Prefer focused commands such as:

```bash
.venv/bin/pytest -q path/to/test_module.py
.venv/bin/pytest -q path/to/test_module.py::test_specific_behavior
```

Run linting only on the changed or directly affected files where possible.

## Pre-commit checks

Before finishing changes, run pre-commit on the files you changed:

```bash
.venv/bin/pre-commit run --files <changed-file-paths>
```

Review any automatic fixes and rerun until checks pass. Report any unresolved
failures. Do not install or update packages in `.venv`.

Run across all tracked files only when changing hook versions or investigating a
repository-wide CI failure.

## Code explanation references

When explaining or reviewing repository code:

- Include clickable Markdown links with the exact current line number for the functions, branches, and call sites central to the explanation.
- For event flows, link each material step to its implementation rather than linking only the containing class or module.
- Re-check line numbers against the current working tree immediately before responding because edits may move them.
- Use absolute paths with one starting line, not line ranges.
- Do not overload explanations with links to incidental symbols.

## Codex config

Repository-local Codex settings for this project live in `.codex/config.toml`.

When checking or updating Codex sandbox, cache, approval, or environment settings for this repo,
use `.codex/config.toml` first rather than `~/.codex/config.toml`.
