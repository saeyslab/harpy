<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/harpy.svg?color=green)](https://github.com/saeyslab/harpy/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/harpy-analysis.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/saeyslab/harpy/graph/badge.svg?token=7UXMDWVYFZ)](https://codecov.io/gh/saeyslab/harpy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/harpy)](https://napari-hub.org/plugins/harpy)
-->

<div align="center">
  <img src="./docs/_static/img/logo.png" alt="Harpy logo" width="200" />
  <h2>Single-cell spatial omics analysis that makes you happy.</h2>
</div>

[![PyPI](https://img.shields.io/pypi/v/harpy-analysis.svg)](https://pypi.org/project/harpy-analysis)
[![Downloads](https://static.pepy.tech/badge/harpy-analysis)](https://pepy.tech/project/harpy-analysis)
[![Build Status](https://github.com/saeyslab/harpy/actions/workflows/build.yaml/badge.svg)](https://github.com/saeyslab/harpy/actions/)
[![documentation badge](https://readthedocs.org/projects/harpy/badge/?version=latest)](https://harpy.readthedocs.io/en/latest/)
[![Test Status](https://github.com/saeyslab/harpy/actions/workflows/run_tests.yml/badge.svg)](https://github.com/saeyslab/harpy/actions/)
[![codecov](https://codecov.io/gh/saeyslab/harpy/branch/main/graph/badge.svg)](https://codecov.io/gh/saeyslab/harpy)
[![License](https://img.shields.io/badge/license-Academic%20Non--commercial-blue)](./LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/saeyslab/harpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> 💫 **If you find Harpy useful, please give us a [⭐](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

## Why Harpy?

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **End-to-end workflows** for segmentation, feature extraction, clustering, and spatial analysis.

## Installation

**Recommended** for end-users (Python `>=3.10`).

```bash
uv venv --python=3.12  # set python version
source .venv/bin/activate  # activate the virtual environment
uv pip install "harpy-analysis[extra]"  # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)'  # check if the package is installed
```

**Only for developers.** Clone this repository locally, install the `.[dev]` instead of the `[extra]` dependencies and read the contribution guide.

```bash
# Clone repository from GitHub
uv venv --python=3.12  # set python version
source .venv/bin/activate  # activate the virtual environment
uv pip install -e '.[dev]'  # editable install with dev tooling
python -c 'import harpy; print(harpy.__version__)'  # check if the package is installed
# make changes
python -m pytest  # run the tests
```

Checkout the docs for [installation instructions](https://github.com/saeyslab/harpy/blob/main/docs/installation.md) using [conda](https://github.com/conda/conda).

## Quickstart

See the short, runnable [guide](./docs/quickstart.md).

## 🧭 Tutorials and Guides

Explore how to use Harpy for segmentation, shallow and deep feature extraction, clustering, and spatial analysis of gigapixel-scale multiplexed data with these step-by-step notebooks:

- **🚀 Basic Usage of Harpy**

  Learn how to read in data, perform **tiled segmentation** using [**Cellpose**](https://github.com/MouseLand/cellpose) and [**Dask-CUDA**](https://docs.rapids.ai/api/dask-cuda/stable/), extract features, and carry out clustering. 👉 [Tutorial](./docs/tutorials/general/Harpy_feature_calculation.ipynb)

- **🔧 Technology-specific advice**

  Learn which technologies Harpy supports. 👉 [Notebook](./docs/tutorials/general/techno_specific.ipynb)

- **🧩 Pixel and Cell Clustering**

  Learn how to perform unsupervised pixel- and cell-level clustering using `Harpy` together with [**FlowSOM**](https://github.com/saeyslab/FlowSOM_Python). 👉 [Tutorial](./docs/tutorials/general/FlowSOM_for_pixel_and_cell_clustering.ipynb)

- **✂️ Cell Segmentation**

  Explore segmentation workflows in `Harpy` using different tools:
  - With [**Instanseg**](https://github.com/instanseg/instanseg) 👉 [Tutorial](./docs/tutorials/general/Harpy_instanseg.ipynb)

  - With [**Cellpose**](https://github.com/MouseLand/cellpose) 👉 [Tutorial](./docs/tutorials/general/Harpy_feature_calculation.ipynb)

  💡 Want us to add support for another segmentation method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🧪 Single-cell representations from highly multiplexed images and downstream use with [PyTorch](https://pytorch.org/)**

  Learn how single-cell representations can be generated from highly multiplexed images. These representations can then be used downstream to train classifiers in PyTorch. 👉 [Tutorial](./docs/tutorials/general/generate_single_cell_representations.ipynb)

- **🧠 Deep Feature Extraction**

  Discover how `Harpy` enables fast, scalable extraction of deep, cell-level features from multiplex imaging data with the [**KRONOS**](https://github.com/mahmoodlab/KRONOS) foundation model for proteomics. 👉 [Tutorial](./docs/tutorials/general/Featurize_with_kronos.ipynb)

  💡 Want us to add support for another deep feature extraction method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🔬 Shallow Feature Extraction**

  Learn to extract shallow features—such as **mean**, **median**, and **standard deviation** of intensities—from multiplex imaging data with `Harpy`. 👉 [Tutorial](./docs/tutorials/advanced/Harpy_aggregate_rasters.ipynb)

- **🧬 Spatial Transcriptomics**

  Learn how to analyze spatial transcriptomics data with `Harpy`. For detailed information, refer to the [**SPArrOW documentation**](https://sparrow-pipeline.readthedocs.io/en/latest).

  👉 [Tutorial (Mouse Liver, Resolve Molecular Cartography)](./docs/tutorials/advanced/Harpy_transcriptomics.ipynb)

  👉 [Tutorial (Human Ovarian Cancer, Xenium 10x Genomics)](./docs/tutorials/advanced/Harpy_transcriptomics_xenium.ipynb)

---

- **🌐 Multiple samples and coordinate systems**

  Learn how to work with multiple samples, intrinsic and micron coordinates. 👉 [Tutorial](./docs/tutorials/advanced/coordinate_systems.ipynb)

---

- **📐 Rasterize and vectorize labels and shapes**

  Learn how to convert a segmentation mask (array) into its vectorized form, and segmentation boundaries (polygons) into their rasterized equivalents. This conversion is useful, for example, when integrating annotations (e.g., from [QuPath](https://qupath.github.io/)) into downstream spatial omics analysis.👉 [Tutorial](./docs/tutorials/advanced/Rasterize_and_vectorize.ipynb)

---

📚 For a complete list of tutorials, visit the [**Harpy documentation**](https://harpy.readthedocs.io/en/latest/tutorials).

## Computational benchmark

Explore the benchmark performance of **Harpy** on a large MACSima tonsil proteomics dataset. 👉 [Results](./docs/tutorials/general/benchmark.ipynb)

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

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/saeyslab/harpy/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
