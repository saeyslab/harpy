<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/harpy.svg?color=green)](https://github.com/saeyslab/harpy/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/harpy-analysis.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/saeyslab/harpy/graph/badge.svg?token=7UXMDWVYFZ)](https://codecov.io/gh/saeyslab/harpy)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/harpy)](https://napari-hub.org/plugins/harpy)
-->

<div align="center">
  <img src="https://raw.githubusercontent.com/saeyslab/harpy/main/docs/_static/img/logo.png" alt="Harpy logo" width="200" />
  <h2>Single-cell spatial omics analysis that makes you happy.</h2>
</div>

<div align="center">

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

</div>

<p align="center">
  <a href="https://harpy.readthedocs.io/en/latest/">Documentation</a>
  ·
  <a href="https://harpy.readthedocs.io/en/latest/quickstart.html">Quick Start</a>
  ·
  <a href="https://harpy.readthedocs.io/en/latest/tutorials/index.html">Tutorials</a>
  ·
  <a href="https://github.com/vibspatial/harpy_vitessce">Harpy Vitessce</a>
</p>

> 💫 **If you find Harpy useful, please give us a [⭐](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

## Why Harpy?

Harpy is a spatial omics analysis library for spatial transcriptomics and proteomics. Within the [`scverse`](https://scverse.org/) stack, it bridges [`SpatialData`](https://spatialdata.scverse.org/) and downstream analysis tools such as [`AnnData`](https://anndata.readthedocs.io/), [`Scanpy`](https://scanpy.readthedocs.io/), and [`Squidpy`](https://squidpy.readthedocs.io/). It provides scalable, image- and geometry-aware computation to transform raw spatial data into analysis-ready representations, with a strong emphasis on interoperability and large-scale workflows.

In practice, Harpy offers fast, out-of-core image preprocessing, tiled segmentation, along with efficient aggregation workflows to generate `AnnData` tables and compute per-cell features from images, segmentation masks, and transcript coordinates. It also supports deep feature extraction, pixel- and cell-level clustering, and the construction of single-cell representations from highly multiplexed images.

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **Scalable computational building blocks** for segmentation, feature extraction, clustering, and spatial analysis.

## Installation

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
- FlowSOM Clustering: `flowsom`
- Notebook workflows: `ipywidgets`, `tqdm`, `bokeh`, `textalloc`, `joypy`, `supervenn`, `nbconvert`, `ipython`
- CLI workflows: `hydra-core`

**With extras and napari**

```bash
pip install "harpy-analysis[extra,napari]"
```

`[napari]` adds:

- `napari[all]`
- `napari-spatialdata`

**Only for developers.** Clone this repository locally, install the `.[dev]` instead of the `[extra]` dependencies and read the [contribution guide](https://harpy.readthedocs.io/en/latest/contributing.html).

```bash
# Clone repository from GitHub
uv venv --python=3.12  # create venv, set python version (>=3.11)
source .venv/bin/activate  # activate the virtual environment
uv pip install -e '.[dev]'  # editable install with dev tooling
python -c 'import harpy; print(harpy.__version__)'  # check if the package is installed
# make changes
python -m pytest  # run the tests
```

It is possible to install Harpy using Anaconda although we recommend [uv](https://github.com/astral-sh/uv), see the [installation guide](./docs/installation.md).

## Quickstart

See the short, runnable [guide](./docs/quickstart.md).

## 🧭 Tutorials and Guides

Explore how to use Harpy for segmentation, shallow and deep feature extraction, clustering, and spatial analysis of gigapixel-scale multiplexed data with these step-by-step notebooks:

- **🚀 Basic Usage of Harpy**

  Learn how to read in data, perform **tiled segmentation** using [**Cellpose**](https://github.com/MouseLand/cellpose) and [**Dask-CUDA**](https://docs.rapids.ai/api/dask-cuda/stable/), extract features, perform QC and analyze results downstream with `Scanpy` and `Squidpy`.

  👉 [Tutorial image based transcriptomics, Human Ovarian Cancer, Xenium 10x Genomics](https://github.com/vibspatial/harpy_notebooks/blob/main/general/Harpy_xenium_transcriptomics_subset.ipynb)

  👉 [Tutorial proteomics, MACSima](https://github.com/vibspatial/harpy_notebooks/blob/main/general/Harpy_feature_calculation.ipynb)

- **🔧 Technology-specific advice**

  Learn which technologies Harpy supports. 👉 [Notebook](https://github.com/vibspatial/harpy_notebooks/blob/main/general/techno_specific.ipynb)

- **🧩 Pixel and Cell Clustering**

  Learn how to perform unsupervised pixel- and cell-level clustering using `Harpy` together with [**FlowSOM**](https://github.com/saeyslab/FlowSOM_Python). 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/general/FlowSOM_for_pixel_and_cell_clustering.ipynb)

- **✂️ Cell Segmentation**

  Explore segmentation workflows in `Harpy` using different tools:
  - With [**Instanseg**](https://github.com/instanseg/instanseg) 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/general/Harpy_instanseg.ipynb)

  - With [**Cellpose**](https://github.com/MouseLand/cellpose) 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/general/Harpy_feature_calculation.ipynb)

  💡 Want us to add support for another segmentation method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🧪 Single-cell representations from highly multiplexed images and downstream use with [PyTorch](https://pytorch.org/)**

  Learn how single-cell representations can be generated from highly multiplexed images. These representations can then be used downstream to train classifiers in PyTorch. 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/general/generate_single_cell_representations.ipynb)

- **🧠 Deep Feature Extraction**

  Discover how `Harpy` enables fast, scalable extraction of deep, cell-level features from multiplex imaging data with the [**KRONOS**](https://github.com/mahmoodlab/KRONOS) foundation model for proteomics. 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/general/Featurize_with_kronos.ipynb)

  💡 Want us to add support for another deep feature extraction method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

- **🔬 Shallow Feature Extraction**

  Learn to extract shallow features—such as **mean**, **median**, and **standard deviation** of intensities—from multiplex imaging data with `Harpy`. 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/advanced/Harpy_aggregate_rasters.ipynb)

- **🧬 Spatial Transcriptomics**

  Learn how to analyze spatial transcriptomics data with `Harpy`. For detailed information, refer to the [**SPArrOW documentation**](https://sparrow-pipeline.readthedocs.io/en/latest).

  👉 [Tutorial (Mouse Liver, Resolve Molecular Cartography)](https://github.com/vibspatial/harpy_notebooks/blob/main/advanced/Harpy_transcriptomics.ipynb)

  👉 [Tutorial (Human Ovarian Cancer, Xenium 10x Genomics)](https://github.com/vibspatial/harpy_notebooks/blob/main/advanced/Harpy_transcriptomics_xenium.ipynb)

---

- **🌐 Multiple samples and coordinate systems**

  Learn how to work with multiple samples, intrinsic and micron coordinates. 👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/advanced/coordinate_systems.ipynb)

---

- **📐 Rasterize and vectorize labels and shapes**

  Learn how to convert a segmentation mask (array) into its vectorized form, and segmentation boundaries (polygons) into their rasterized equivalents. This conversion is useful, for example, when integrating annotations (e.g., from [QuPath](https://qupath.github.io/)) into downstream spatial omics analysis.👉 [Tutorial](https://github.com/vibspatial/harpy_notebooks/blob/main/advanced/Rasterize_and_vectorize.ipynb)

---

📚 For a complete list of tutorials, visit the [**Harpy documentation**](https://harpy.readthedocs.io/en/latest/tutorials).

## Computational benchmark

Explore the benchmark performance of **Harpy** on a large MACSima tonsil proteomics dataset. 👉 [Results](https://github.com/vibspatial/harpy_notebooks/blob/main/general/benchmark.ipynb)

## Contributing

See the [contribution guide](https://harpy.readthedocs.io/en/latest/contributing.html) for info on how to contribute to Harpy.

## Citation

If you use Harpy in your work, please cite:

> Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Evelien Van Hamme, Sofie Van Gassen, Ruth Seurinck, Yvan Saeys. _Scalable analysis of whole slide spatial proteomics with Harpy_. _Bioinformatics_ (2026), btag122. [https://doi.org/10.1093/bioinformatics/btag122](https://doi.org/10.1093/bioinformatics/btag122)

If you use Harpy for spatial transcriptomics analysis, please cite:

> Lotte Pollaris, Bavo Vanneste, Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Wout Vanhenden, Liesbet Martens, Tinne Thone, Jean-Francois Hastir, Anna Bujko, Wouter Saelens, Jean-Christophe Marine, Hilde Nelissen, Evelien Van Hamme, Ruth Seurinck, Charlotte L. Scott, Martin Guilliams, Yvan Saeys. _SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis_. [https://doi.org/10.1101/2024.07.04.601829](https://doi.org/10.1101/2024.07.04.601829)

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
