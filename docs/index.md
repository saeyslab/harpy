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

For interactive visualization of `Harpy` outputs using [Vitessce](https://github.com/vitessce/vitessce), see the [harpy_vitessce](https://github.com/vibspatial/harpy_vitessce) package.

Why Harpy?

- **Multi-platform support** for spatial transcriptomics and proteomics data.
- **Interoperable outputs** built on [SpatialData](https://github.com/scverse/spatialdata).
- **Scales to (very) large images**: tiled workflows with [Dask](https://www.dask.org/); optional GPU acceleration with [CuPy](https://cupy.dev/) and [PyTorch](https://pytorch.org/).
- **End-to-end workflows** for segmentation, feature extraction, clustering, and spatial analysis.

---

Explore how to use Harpy for segmentation, shallow and deep feature extraction, clustering, and spatial analysis of gigapixel-scale multiplexed data with these step-by-step notebooks:

---

- **🚀 Basic Usage of Harpy**

  Learn how to read in data, perform **tiled segmentation** using [**Cellpose**](https://github.com/MouseLand/cellpose) and [**Dask-CUDA**](https://docs.rapids.ai/api/dask-cuda/stable/), extract features, and carry out clustering. 👉 [Tutorial](../docs/tutorials/general/Harpy_feature_calculation.ipynb)

---

- **🔧 Technology specific advice**

  Learn which technologies Harpy supports. 👉 [Notebook](../docs/tutorials/general/techno_specific.ipynb)

---

- **🧩 Pixel and Cell Clustering**

  Learn how to perform unsupervised pixel- and cell-level clustering using `Harpy` together with [**FlowSOM**](https://github.com/saeyslab/FlowSOM_Python). 👉 [Tutorial](../docs/tutorials/general/FlowSOM_for_pixel_and_cell_clustering.ipynb)

---

- **✂️ Cell Segmentation**

  Explore segmentation workflows in `Harpy` using different tools:
  - With [**Instanseg**](https://github.com/instanseg/instanseg) 👉 [Tutorial](../docs/tutorials/general//Harpy_instanseg.ipynb)

  - With [**Cellpose**](https://github.com/MouseLand/cellpose) 👉 [Tutorial ](../docs/tutorials/general/Harpy_feature_calculation.ipynb)

  💡 Want us to add support for another segmentation method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

---

- **🧪 Single-cell representations from highly multiplexed images and downstream use with [PyTorch](https://pytorch.org/)**

  Learn how single-cell representations can be generated from highly multiplexed images. These representations can then be used downstream to train classifiers in PyTorch. 👉 [Tutorial](../docs/tutorials/general/generate_single_cell_representations.ipynb)

---

- **🧠 Deep Feature Extraction**

  Discover how `Harpy` enables fast, scalable extraction of deep, cell-level features from highly multiplex imaging data with the [**KRONOS**](https://github.com/mahmoodlab/KRONOS) foundation model for proteomics. 👉 [Tutorial](../docs/tutorials/general/Featurize_with_kronos.ipynb)

  💡 Want us to add support for another deep feature extraction method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

---

- **🔬 Shallow Feature Extraction**

  Learn to extract shallow features—such as **mean**, **median**, and **standard deviation** of intensities—from multiplex imaging data with `Harpy`. 👉 [Tutorial](../docs/tutorials/advanced/Harpy_aggregate_rasters.ipynb)

---

- **🧬 Spatial Transcriptomics**

  Learn how to analyze spatial transcriptomics data with `Harpy`. For detailed information, refer to the [**SPArrOW documentation**](https://sparrow-pipeline.readthedocs.io/en/latest).

  👉 [Tutorial (Mouse Liver, Resolve Molecular Cartography)](../docs/tutorials/advanced/Harpy_transcriptomics.ipynb)

  👉 [Tutorial (Human Ovarian Cancer, Xenium 10x Genomics)](../docs/tutorials/advanced/Harpy_transcriptomics_xenium.ipynb)

---

- **🌐 Multiple samples and coordinate systems**

  Learn how to work with multiple samples, instrinsic and micron coordinates. 👉 [Tutorial](../docs/tutorials/advanced/coordinate_systems.ipynb)

---

- **📐 Unifying Raster and Vector Annotations**

  Learn how to convert a segmentation mask (array) into its vectorized form, and segmentation boundaries (polygons) into their rasterized equivalents. This conversion is useful, for example, when integrating annotations (e.g., from [QuPath](https://qupath.github.io/)) into downstream spatial omics analysis.👉 [Tutorial](../docs/tutorials/advanced/Rasterize_and_vectorize.ipynb)

---

📚 For a complete list of tutorials, visit the [**tutorials section**](https://harpy.readthedocs.io/en/latest/tutorials).

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
    :link: tutorials/index
    :link-type: doc

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
tutorials/index.md
api.md
contributing.md
```
