<!-- These badges won't work while the GitHub repo is private:
[![License BSD-3](https://img.shields.io/pypi/l/harpy-analysis.svg?color=green)](https://github.com/saeyslab/harpy/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/harpy-analysis.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/saeyslab/harpy/graph/badge.svg?token=7UXMDWVYFZ)](https://codecov.io/gh/saeyslab/harpy)
-->

<div align="center">
  <img src="https://raw.githubusercontent.com/saeyslab/harpy/main/docs/_static/img/logo.png" alt="Harpy logo" width="200" />
  <h2>Single-cell spatial omics analysis that makes you happy.</h2>
</div>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/harpy-analysis?style=flat-square&label=pypi)](https://pypi.org/project/harpy-analysis)
[![Release Status](https://img.shields.io/github/actions/workflow/status/saeyslab/harpy/release.yaml?event=release&style=flat-square&label=release)](https://github.com/saeyslab/harpy/actions/workflows/release.yaml)
[![Build Status](https://img.shields.io/github/actions/workflow/status/saeyslab/harpy/build.yaml?branch=main&style=flat-square&label=build)](https://github.com/saeyslab/harpy/actions/)
[![Test Status](https://img.shields.io/github/actions/workflow/status/saeyslab/harpy/run_tests.yml?branch=main&style=flat-square&label=tests)](https://github.com/saeyslab/harpy/actions/)
[![pre-commit.ci status](https://img.shields.io/github/checks-status/saeyslab/harpy/main?style=flat-square&label=pre-commit)](https://results.pre-commit.ci/latest/github/saeyslab/harpy/main)
[![documentation badge](https://img.shields.io/readthedocs/harpy/latest?style=flat-square&label=docs)](https://harpy.readthedocs.io/en/latest/)
[![codecov](https://img.shields.io/codecov/c/github/saeyslab/harpy/main?style=flat-square&label=coverage)](https://codecov.io/gh/saeyslab/harpy)
[![Downloads](https://img.shields.io/pepy/dt/harpy-analysis?style=flat-square&label=downloads)](https://pepy.tech/project/harpy-analysis)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue?style=flat-square)](./LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15546703-blue?style=flat-square)](https://doi.org/10.5281/zenodo.15546703)
[![Paper](https://img.shields.io/badge/paper-Bioinformatics%20btag122-blue?style=flat-square)](https://doi.org/10.1093/bioinformatics/btag122)
[![scverse ecosystem](https://img.shields.io/badge/scverse-ecosystem-blue?style=flat-square)](https://scverse.org/packages/#ecosystem)

</div>

<p align="center">
  <a href="https://harpy.readthedocs.io/en/latest/">Documentation</a>
  ·
  <a href="https://harpy.readthedocs.io/en/latest/quickstart.html">Quick Start</a>
  ·
  <a href="https://github.com/vibspatial/spatial_multi_omics_training">Tutorials</a>
  ·
  <a href="https://github.com/vibspatial/spatiato">spatiato</a>
  ·
  <a href="https://github.com/vibspatial/harpy_vitessce">harpy-vitessce</a>
</p>

> 💫 **If you find Harpy useful, please give us a [⭐](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

## Why Harpy?

Harpy is a spatial omics analysis library for spatial transcriptomics and proteomics, and part of the [scverse ecosystem](https://scverse.org/packages/#ecosystem). It bridges [`SpatialData`](https://spatialdata.scverse.org/) and downstream analysis tools such as [`AnnData`](https://anndata.readthedocs.io/), [`Scanpy`](https://scanpy.readthedocs.io/), and [`Squidpy`](https://squidpy.readthedocs.io/). It provides scalable, image- and geometry-aware computation to transform raw spatial data into analysis-ready representations, with a strong emphasis on interoperability and large-scale workflows.

In practice, Harpy offers fast, out-of-core image preprocessing, tiled segmentation, along with efficient aggregation workflows to generate `AnnData` tables and compute per-cell features from images, segmentation masks, and transcript coordinates. It also supports deep feature extraction, pixel- and cell-level clustering, and the construction of single-cell representations from highly multiplexed images.

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **SpatialData and AnnData ergonomics**: read single tables and write only the components that changed (`obs`, `obsm`, `uns`, …) to a backed Zarr store with [`harpy.tb.io`](https://harpy.readthedocs.io/en/latest/api.html#table-io); add images, labels, shapes and points with [`harpy.im`](https://harpy.readthedocs.io/en/latest/api.html#image), [`harpy.sh`](https://harpy.readthedocs.io/en/latest/api.html#shape) and [`harpy.pt`](https://harpy.readthedocs.io/en/latest/api.html#points).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **Scalable computational building blocks** for segmentation, feature extraction, clustering, and spatial analysis.

## Development and support

Harpy is actively developed and maintained by the [Saeys Lab](https://saeyslab.sites.vib.be/en) and [VIB Spatial Catalyst](https://vib.be/en/technologies/technology-portfolio/spatial-catalyst), part of VIB Technologies. VIB funds a dedicated development team, so Harpy gets long-term maintenance and continued development.

## Installation

```bash
pip install harpy-analysis
```

Optional extras:

| Extra        | Installs                                                                                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[extra]`    | Segmentation (`cellpose`, `instanseg-torch`), OpenCV (`opencv-python-headless`), Vitessce (`harpy-vitessce`) and notebook tools (`marimo`, `ipywidgets`, `tqdm`, …) |
| `[spatiato]` | [spatiato](https://github.com/vibspatial/spatiato), the spatial omics interface for [napari](https://napari.org/stable/)                                            |

```bash
pip install "harpy-analysis[extra]"
```

For the developer setup and more details, see the [installation guide](https://harpy.readthedocs.io/en/latest/installation.html).

## Quickstart

See the short, runnable [guide](https://harpy.readthedocs.io/en/latest/quickstart.html).

## Ecosystem

- **[spatiato](https://github.com/vibspatial/spatiato)**: a spatial omics interface for [napari](https://napari.org/stable/), built on `SpatialData` and Harpy, for interactive exploration and analysis of `SpatialData` stores.
- **[harpy-vitessce](https://github.com/vibspatial/harpy_vitessce)**: interactive visualization of Harpy outputs with [Vitessce](https://github.com/vitessce/vitessce).
- **[Spatial multi-omics training](https://github.com/vibspatial/spatial_multi_omics_training)**: course notebooks on `SpatialData` objects, targeted transcriptomics and spatial proteomics with Harpy and spatiato.

## Computational benchmark

Explore the benchmark performance of **Harpy** on a large MACSima tonsil proteomics dataset. 👉 [Results](https://harpy.readthedocs.io/en/latest/tutorials/general/benchmark.html)

## Contributing

See the [contribution guide](https://harpy.readthedocs.io/en/latest/contributing.html) for info on how to contribute to Harpy.

## Citation

If you use Harpy in your work, please cite:

> Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Evelien Van Hamme, Sofie Van Gassen, Ruth Seurinck, Yvan Saeys. _Scalable analysis of whole slide spatial proteomics with Harpy_. _Bioinformatics_ (2026), btag122. [https://doi.org/10.1093/bioinformatics/btag122](https://doi.org/10.1093/bioinformatics/btag122)

If you use Harpy for spatial transcriptomics analysis, please cite:

> Lotte Pollaris, Bavo Vanneste, Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Wout Vanhenden, Liesbet Martens, Tinne Thone, Jean-Francois Hastir, Anna Bujko, Wouter Saelens, Jean-Christophe Marine, Hilde Nelissen, Evelien Van Hamme, Ruth Seurinck, Charlotte L. Scott, Martin Guilliams, Yvan Saeys. _SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis_. [https://doi.org/10.1101/2024.07.04.601829](https://doi.org/10.1101/2024.07.04.601829)

<details>
<summary>BibTeX</summary>

```bibtex
@article{rombaut2026harpy,
  title   = {Scalable analysis of whole slide spatial proteomics with Harpy},
  author  = {Rombaut, Benjamin and Defauw, Arne and Vernaillen, Frank and Mortier, Julien and Van Hamme, Evelien and Van Gassen, Sofie and Seurinck, Ruth and Saeys, Yvan},
  journal = {Bioinformatics},
  volume  = {42},
  number  = {3},
  pages   = {btag122},
  year    = {2026},
  doi     = {10.1093/bioinformatics/btag122}
}

@article{pollaris2024sparrow,
  title   = {SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis},
  author  = {Pollaris, Lotte and Vanneste, Bavo and Rombaut, Benjamin and Defauw, Arne and Vernaillen, Frank and Mortier, Julien and Vanhenden, Wout and Martens, Liesbet and Thon{\'e}, Tinne and Hastir, Jean-Francois and Bujko, Anna and Saelens, Wouter and Marine, Jean-Christophe and Nelissen, Hilde and Van Hamme, Evelien and Seurinck, Ruth and Scott, Charlotte L. and Guilliams, Martin and Saeys, Yvan},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.07.04.601829}
}
```

</details>

## License

Harpy is licensed under the [BSD 3-Clause License](./LICENSE).

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/saeyslab/harpy/issues
