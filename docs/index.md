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

> 💫 **If you find Harpy useful, please give us a [⭐ on GitHub](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

Why Harpy?

Harpy is a spatial omics analysis library for spatial transcriptomics and proteomics. Within the [`scverse`](https://scverse.org/) stack, it bridges [`SpatialData`](https://spatialdata.scverse.org/) and downstream analysis tools such as [`AnnData`](https://anndata.readthedocs.io/), [`Scanpy`](https://scanpy.readthedocs.io/), and [`Squidpy`](https://squidpy.readthedocs.io/). It provides scalable, image- and geometry-aware computation to transform raw spatial data into analysis-ready representations, with a strong emphasis on interoperability and large-scale workflows.

In practice, Harpy offers fast, out-of-core image preprocessing, tiled segmentation, and efficient aggregation workflows to generate `AnnData` tables and compute per-cell features from images, segmentation masks, and transcript coordinates. It also supports deep feature extraction, pixel- and cell-level clustering, and the construction of single-cell representations from highly multiplexed images.

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **Scalable computational building blocks** for segmentation, feature extraction, clustering, and spatial analysis.

For interactive exploration and analysis of `SpatialData` stores, see the [spatiato](https://github.com/vibspatial/spatiato) package, a spatial omics interface for [napari](https://napari.org/stable/), built on `SpatialData` and `Harpy`.

For interactive visualization of `Harpy` outputs using [Vitessce](https://github.com/vitessce/vitessce), see the [harpy_vitessce](https://github.com/vibspatial/harpy_vitessce) package.

> Note for users upgrading to Harpy `0.4.0`: parameters that refer to `SpatialData` elements now use the `*_name` convention instead of the older `*_layer` naming to stay aligned with scverse naming conventions. For example, `img_layer` becomes `image_name`, `labels_layer` becomes `labels_name`, and `table_layer` becomes `table_name`.

---

Explore how to use Harpy with these step-by-step notebooks:

---

- **🎓 Spatial multi-omics training**

  Work through course notebooks on `SpatialData` objects, targeted transcriptomics and spatial proteomics with Harpy and spatiato. 👉 [Training](https://github.com/vibspatial/spatial_multi_omics_training)

---

```{eval-rst}
.. card:: Installation
    :link: installation
    :link-type: doc

    Learn how to install Harpy.

.. card:: Quickstart
    :link: quickstart
    :link-type: doc

    Run a short, end-to-end example.

.. card:: Tutorials
    :link: https://github.com/vibspatial/spatial_multi_omics_training
    :link-type: url

    Tutorials to help you get up to speed with Harpy.

.. card:: Technology-specific advice
    :link: tutorials/general/techno_specific
    :link-type: doc

    Learn which technologies Harpy supports.

.. card:: API
    :link: api
    :link-type: doc

    Find a detailed documentation of Harpy.

.. card:: Computational Benchmark
    :link: tutorials/general/benchmark
    :link-type: doc

    Explore Harpy's benchmark performance.

.. card:: HPC
    :link: tutorials/hpc/index
    :link-type: doc

    Learn how to run Harpy in a High-Performance Computing (HPC) environment.


.. card:: Contributing
    :link: contributing
    :link-type: doc

    Learn how to contribute to Harpy.

```

If you use Harpy for spatial proteomics analysis, please cite:

> Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Evelien Van Hamme, Sofie Van Gassen, Ruth Seurinck, Yvan Saeys. _Scalable analysis of whole slide spatial proteomics with Harpy_. _Bioinformatics_ (2026), btag122. [https://doi.org/10.1093/bioinformatics/btag122](https://doi.org/10.1093/bioinformatics/btag122)

If you use Harpy for spatial transcriptomics analysis, please cite:

> Lotte Pollaris, Bavo Vanneste, Benjamin Rombaut, Arne Defauw, Frank Vernaillen, Julien Mortier, Wout Vanhenden, Liesbet Martens, Tinne Thone, Jean-Francois Hastir, Anna Bujko, Wouter Saelens, Jean-Christophe Marine, Hilde Nelissen, Evelien Van Hamme, Ruth Seurinck, Charlotte L. Scott, Martin Guilliams, Yvan Saeys. _SPArrOW: a flexible, interactive and scalable pipeline for spatial transcriptomics analysis_. [https://doi.org/10.1101/2024.07.04.601829](https://doi.org/10.1101/2024.07.04.601829)

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
