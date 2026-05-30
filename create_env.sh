#uv venv .venv_harpy --python 3.13
#source .venv_harpy/bin/activate

#uv pip install -e '.[dev]'
#uv pip install -e '.[napari]'
#uv pip install squidpy

# Use uv sync so pyproject.toml, including [tool.uv] dependency overrides, is
# the single source of truth for the environment.
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.13 --locked