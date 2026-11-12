# Harpy

<style>
section#harpy > h1 {
  display: none;
}
</style>

<div align="center">
  <img src="_static/img/logo.png" alt="Harpy logo" width="220" />
  <p><strong><span style="font-size:1.8em;">Single-cell spatial omics analysis that makes you happy.</span></strong></p>
</div>

<p align="center">
  <a href="https://pypi.org/project/harpy-analysis"><img src="https://img.shields.io/pypi/v/harpy-analysis?style=flat-square&label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/saeyslab/harpy/actions/"><img src="https://img.shields.io/github/actions/workflow/status/saeyslab/harpy/run_tests.yml?branch=main&style=flat-square&label=tests" alt="Test Status" /></a>
  <a href="https://codecov.io/gh/saeyslab/harpy"><img src="https://img.shields.io/codecov/c/github/saeyslab/harpy/main?style=flat-square&label=coverage" alt="codecov" /></a>
  <a href="https://pepy.tech/project/harpy-analysis"><img src="https://img.shields.io/pepy/dt/harpy-analysis?style=flat-square&label=downloads" alt="Downloads" /></a>
  <a href="https://github.com/saeyslab/harpy/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD%203--Clause-blue?style=flat-square" alt="License" /></a>
  <a href="https://doi.org/10.5281/zenodo.15546703"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15546703-blue?style=flat-square" alt="DOI" /></a>
  <a href="https://doi.org/10.1093/bioinformatics/btag122"><img src="https://img.shields.io/badge/paper-Bioinformatics%20btag122-blue?style=flat-square" alt="Paper" /></a>
  <a href="https://scverse.org/packages/#ecosystem"><img src="https://img.shields.io/badge/scverse-ecosystem-blue?style=flat-square" alt="scverse ecosystem" /></a>
</p>

> 💫 **If you find Harpy useful, please give us a [⭐ on GitHub](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

## Why Harpy?

Harpy is a spatial omics analysis library for spatial transcriptomics and proteomics, and part of the [scverse ecosystem](https://scverse.org/packages/#ecosystem). It bridges [`SpatialData`](https://spatialdata.scverse.org/) and downstream analysis tools such as [`AnnData`](https://anndata.readthedocs.io/), [`Scanpy`](https://scanpy.readthedocs.io/), and [`Squidpy`](https://squidpy.readthedocs.io/). It provides scalable, image- and geometry-aware computation to transform raw spatial data into analysis-ready representations, with a strong emphasis on interoperability and large-scale workflows.

In practice, Harpy offers fast, out-of-core image preprocessing, tiled segmentation, and efficient aggregation workflows to generate `AnnData` tables and compute per-cell features from images, segmentation masks, and transcript coordinates. It also supports deep feature extraction, pixel- and cell-level clustering, and the construction of single-cell representations from highly multiplexed images.

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **SpatialData and AnnData ergonomics**: read single tables and write only the components that changed (`obs`, `obsm`, `uns`, …) to a backed Zarr store with [`harpy.tb.io`](#table-io); add images, labels, shapes and points with [`harpy.im`](api.md#image), [`harpy.sh`](api.md#shape) and [`harpy.pt`](api.md#points).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **Scalable computational building blocks** for segmentation, feature extraction, clustering, and spatial analysis.

## Explore the documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Learn how to install Harpy.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Run a short, end-to-end example.
:::

:::{grid-item-card} Tutorials
:link: https://github.com/vibspatial/spatial_multi_omics_training
:link-type: url

Work through course notebooks on `SpatialData` objects, targeted transcriptomics and spatial proteomics.
:::

:::{grid-item-card} Technology-specific advice
:link: tutorials/general/techno_specific
:link-type: doc

Learn which technologies Harpy supports.
:::

:::{grid-item-card} API
:link: api
:link-type: doc

Browse the full API reference.
:::

:::{grid-item-card} Computational benchmark
:link: tutorials/general/benchmark
:link-type: doc

Explore Harpy's performance on a large MACSima tonsil proteomics dataset.
:::

:::{grid-item-card} HPC
:link: tutorials/hpc/index
:link-type: doc

Learn how to run Harpy in a High-Performance Computing (HPC) environment.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

Learn how to contribute to Harpy.
:::

::::

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

For the developer setup and more details, see the [installation guide](installation.md). To try Harpy, run the short [quickstart](quickstart.md).

## Ecosystem

- **[spatiato](https://github.com/vibspatial/spatiato)**: a spatial omics interface for [napari](https://napari.org/stable/), built on `SpatialData` and Harpy, for interactive exploration and analysis of `SpatialData` stores.
- **[harpy-vitessce](https://github.com/vibspatial/harpy_vitessce)**: interactive visualization of Harpy outputs with [Vitessce](https://github.com/vitessce/vitessce).
- **[Spatial multi-omics training](https://github.com/vibspatial/spatial_multi_omics_training)**: course notebooks on `SpatialData` objects, targeted transcriptomics and spatial proteomics with Harpy and spatiato.

## Development and support

Harpy is actively developed and maintained by the [Saeys Lab](https://saeyslab.sites.vib.be/en) and [VIB Spatial Catalyst](https://vib.be/en/technologies/technology-portfolio/spatial-catalyst), part of VIB Technologies. VIB funds a dedicated development team, so Harpy gets long-term maintenance and continued development.

## Citation

If you use Harpy in your work, please cite:

> Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Evelien Van Hamme, Sofie Van Gassen, Ruth Seurinck, Yvan Saeys. _Scalable analysis of whole slide spatial proteomics with Harpy_. _Bioinformatics_ (2026), btag122. [https://doi.org/10.1093/bioinformatics/btag122](https://doi.org/10.1093/bioinformatics/btag122)

If you use Harpy for spatial transcriptomics analysis, please cite:

> Lotte Pollaris, Bavo Vanneste, Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Wout Vanhenden, Liesbet Martens, Tinne Thone, Jean-Francois Hastir, Anna Bujko, Wouter Saelens, Jean-Christophe Marine, Hilde Nelissen, Evelien Van Hamme, Ruth Seurinck, Charlotte L. Scott, Martin Guilliams, Yvan Saeys. _SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis_. [https://doi.org/10.1101/2024.07.04.601829](https://doi.org/10.1101/2024.07.04.601829)

:::{dropdown} BibTeX

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

:::

## License

Harpy is licensed under the [BSD 3-Clause License](https://github.com/saeyslab/harpy/blob/main/LICENSE).

## Issues

If you encounter any problems, please [file an issue](https://github.com/saeyslab/harpy/issues) along with a detailed description.

```{toctree}
:hidden: true
:maxdepth: 2

installation.md
quickstart.md
usage.md
Tutorials <https://github.com/vibspatial/spatial_multi_omics_training>
tutorials/general/techno_specific.ipynb
api.md
canonical_centers.md
tutorials/general/benchmark.ipynb
tutorials/hpc/index.md
contributing.md
```
