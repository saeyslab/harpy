<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/harpy.svg?color=green)](https://github.com/saeyslab/harpy/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/harpy-analysis.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/saeyslab/harpy/graph/badge.svg?token=7UXMDWVYFZ)](https://codecov.io/gh/saeyslab/harpy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/harpy)](https://napari-hub.org/plugins/harpy)
-->

# **Harpy: single-cell spatial proteomics analysis that makes you happy** <img src="./docs/_static/img/logo.png" align ="right" alt="" width ="150"/>

[![PyPI](https://img.shields.io/pypi/v/harpy-analysis.svg)](https://pypi.org/project/harpy-analysis)
[![Downloads](https://static.pepy.tech/badge/harpy-analysis)](https://pepy.tech/project/harpy-analysis)
[![Build Status](https://github.com//saeyslab/harpy/actions/workflows/build.yaml/badge.svg)](https://github.com//saeyslab/harpy/actions/)
[![documentation badge](https://readthedocs.org/projects/harpy/badge/?version=latest)](https://harpy.readthedocs.io/en/latest/)
[![Test Status](https://github.com//saeyslab/harpy/actions/workflows/run_tests.yml/badge.svg)](https://github.com//saeyslab/harpy/actions/)
[![codecov](https://codecov.io/gh/saeyslab/harpy/branch/main/graph/badge.svg)](https://codecov.io/gh/saeyslab/harpy)
[![License](https://img.shields.io/badge/license-Academic%20Non--commercial-blue)](./LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/saeyslab/harpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> 💫 **If you find Harpy useful, please give us a [⭐](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

Note: This package is still under active development.

## Installation

**Recommended** for end-users.

<!--Install the latest `harpy-analysis` [PyPI package](https://pypi.org/project/harpy-analysis) with the `extra` dependencies in a local Python environment.-->

```bash
uv venv --python=3.12 # set python version
source .venv/bin/activate # activate the virtual environment
uv pip install "git+https://github.com/saeyslab/harpy.git#egg=harpy-analysis[extra]" # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)' # check if the package is installed
```

**Only for developers.** Clone this repository locally, install the `.[dev]` instead of the `[extra]` dependencies and read the contribution guide.

```bash
# Clone repository from GitHub
uv venv --python=3.12 # set python version
source .venv/bin/activate # activate the virtual environment
uv pip install -e '.[dev]' # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)' # check if the package is installed
# make changes
python -m pytest # run the tests
```

Checkout the docs for [installation instructions](https://github.com/saeyslab/harpy/blob/main/docs/installation.md) using [conda](https://github.com/conda/conda).

## 🧭 Tutorials and Guides

Explore how to use Harpy for segmentation, shallow and deep feature extraction, clustering, and spatial analysis of gigapixel-scale multiplexed data with these step-by-step notebooks:

- **🚀 Basic Usage of Harpy**
  Learn how to read in data, perform **tiled segmentation** using [**Cellpose**](https://github.com/MouseLand/cellpose) and [**Dask-CUDA**](https://docs.rapids.ai/api/dask-cuda/stable/), extract features, and carry out clustering.
  📘 [Open notebook →](./docs/tutorials/general/Harpy_feature_calculation.ipynb)

- **🧩 Pixel and Cell Clustering**
  Learn how to perform unsupervised pixel- and cell-level clustering using `Harpy` together with [**FlowSOM**](https://github.com/saeyslab/FlowSOM_Python).
  📘 [Open notebook →](./docs/tutorials/general/FlowSOM_for_pixel_and_cell_clustering.ipynb)

- **🔬 Cell Segmentation**
  Explore segmentation workflows in `Harpy` using different tools:
  - With [**Instanseg**](https://github.com/instanseg/instanseg)
    📘 [Open notebook →](./docs/tutorials/general//Harpy_instanseg.ipynb)

  - With [**Cellpose**](https://github.com/MouseLand/cellpose)
    📘 [Open notebook →](./docs/tutorials/general/Harpy_feature_calculation.ipynb)

  💡 Want us to add support for another segmentation method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🧠 Deep Feature Extraction**
  Discover how `Harpy` enables fast, scalable extraction of deep, cell-level features from multiplex imaging data with the [**KRONOS**](https://github.com/mahmoodlab/KRONOS) foundation model for proteomics.
  📘 [Open notebook →](./docs/tutorials/general/Featurize_with_kronos.ipynb)

  💡 Want us to add support for another deep feature extraction method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🔬 Shallow Feature Extraction**
  Learn to extract shallow features—such as **mean**, **median**, and **standard deviation** of intensities—from multiplex imaging data with `Harpy`.
  📘 [Open notebook →](./docs/tutorials/advanced/Harpy_aggregate_rasters.ipynb)

- **🧬 Spatial Transcriptomics**
  Learn how to analyze spatial transcriptomics data with `Harpy`. For detailed information, refer to the [**SPArrOW documentation**](https://sparrow-pipeline.readthedocs.io/en/latest).
  📘 [Open notebook →](./docs/tutorials/advanced/Harpy_transcriptomics.ipynb)

📚 For a complete list of tutorials, visit the [**Harpy documentation**](https://harpy.readthedocs.io/en/latest/tutorials).

## Usage

[Learn](https://github.com/saeyslab/harpy/blob/main/docs/usage.md) how `Harpy` can be integrated into your workflow.

## Contributing

See [here](https://github.com/saeyslab/harpy/blob/main/docs/contributing.md) for info on how to contribute to Harpy.

## References

- https://github.com/ashleve/lightning-hydra-template

## License

Check the [license](https://github.com/saeyslab/harpy/blob/main/LICENSE). Harpy is free for academic usage.
For commercial usage, please contact Saeyslab.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/saeyslab/harpy/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
