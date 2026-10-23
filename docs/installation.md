# Installation

```bash
pip install harpy-analysis
```

**With extras**

```bash
pip install "harpy-analysis[extra]"
```

`[extra]` installs optional dependencies for:

- Segmentation: `cellpose`
- OpenCV support: `opencv-python-headless`
- FlowSOM Clustering: `flowsom`, `scikit-learn`
- Vitessce: `harpy-vitessce`
- Notebook workflows: `ipywidgets`, `tqdm`, `bokeh`, `textalloc`, `joypy`, `supervenn`, `nbconvert`, `ipython`

**With spatiato**

[spatiato](https://github.com/vibspatial/spatiato) is a spatial omics interface for [napari](https://napari.org/stable/), built on `SpatialData` and `Harpy`, for interactive exploration and analysis of `SpatialData` stores.

```bash
pip install "harpy-analysis[spatiato]"
```

`[spatiato]` adds:

- `napari[all]`
- `spatiato`

**Only for developers.** Clone this repository locally, install the `.[dev]` instead of the `.[extra]` dependencies and read the [contribution guide](contributing.md).

```bash
# Clone repository from GitHub
uv venv --python=3.12  # create venv using uv, set python version (>=3.11)
source .venv/bin/activate  # activate the virtual environment
uv pip install -e '.[dev]'  # editable install with dev tooling
python -c 'import harpy; print(harpy.__version__)'  # check if the package is installed
# make changes
python -m pytest  # run the tests
```
