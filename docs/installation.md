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
- Notebook workflows: `ipywidgets`, `tqdm`, `bokeh`, `textalloc`, `joypy`, `supervenn`, `nbconvert`, `ipython`
- CLI workflows: `hydra-core`, `hydra-colorlog`, `submitit`, `hydra-submitit-launcher`

**With extras and napari**

```bash
pip install "harpy-analysis[extra,napari]"
```

`[napari]` adds:

- `napari[all]`
- `napari-spatialdata`

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

## Installation using conda

It is possible to install Harpy using Anaconda although we recommend [uv](https://github.com/astral-sh/uv), and we provide an [`environment.yml`](../environment.yml).

### 1. Create the conda environment:

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate harpy
```

If you plan to use the `Harpy` function `harpy.im.tiling_correction`, please install `jax` and `basicpy`. On Mac and Linux, this can be done via `pip install ...`, on Windows you will have to run the following commands:

```bash
pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy
```

On Mac, please comment out the line `mkl=2024.0.0` in `environment.yml`.

For a mimimal list of requirements for `Harpy`, we refer to the [pyproject.toml](../pyproject.toml).

### 2. Install `Harpy`:

```
pip install "harpy-analysis[extra]"
```

### 3. Additional dependencies

To be able to run the unit tests:

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=harpy-analysis[dev]"
```

## NVIDIA GPU support

We recommend using [uv](https://github.com/astral-sh/uv), however, it is possible to use Anaconda, and we provide [environment_vib_compute.yml](../environment_vib_compute.yml) that will install `torch` with NVIDIA GPU support on Linux (tested on CentOS). After creation of the environment via `conda env create -f environment_vib_compute.yml`, activate the environment, and install `Harpy` via `pip install git+https://github.com/saeyslab/harpy.git`.

For VIB members we also refer to [this document](./tutorials/hpc/vib_compute.md), for an example on how to use the VIB compute cluster with GPU support.
