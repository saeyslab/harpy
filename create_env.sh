uv venv .venv_harpy --python 3.13
source .venv_harpy/bin/activate

uv pip install -e '.[dev]'
uv pip install -e '.[napari]'
uv pip install squidpy
