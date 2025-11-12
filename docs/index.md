# Harpy: single-cell spatial proteomics analysis that makes you happy

> 💫 **If you find Harpy useful, please give us a [⭐ on GitHub](https://github.com/saeyslab/harpy)!** It helps others discover the project and supports continued development.

Harpy is a library for highly-multiplexed microscopy image analysis with single-cell resolution.

Explore how to use Harpy for segmentation, shallow and deep feature extraction, clustering, and spatial analysis of gigapixel-scale multiplexed data with these step-by-step notebooks:

---

- **🚀 Basic Usage of Harpy**

  Learn how to read in data, perform **tiled segmentation** using [**Cellpose**](https://github.com/MouseLand/cellpose) and [**Dask-CUDA**](https://docs.rapids.ai/api/dask-cuda/stable/), extract features, and carry out clustering. 👉 [Tutorial](../docs/tutorials/general/Harpy_feature_calculation.ipynb)

---

- **🧩 Pixel and Cell Clustering**

  Learn how to perform unsupervised pixel- and cell-level clustering using `Harpy` together with [**FlowSOM**](https://github.com/saeyslab/FlowSOM_Python). 👉 [Tutorial](../docs/tutorials/general/FlowSOM_for_pixel_and_cell_clustering.ipynb)

---

- **🔬 Cell Segmentation**

  Explore segmentation workflows in `Harpy` using different tools:

  - With [**Instanseg**](https://github.com/instanseg/instanseg) 👉 [Tutorial](../docs/tutorials/general//Harpy_instanseg.ipynb)

  - With [**Cellpose**](https://github.com/MouseLand/cellpose) 👉 [Tutorial ](../docs/tutorials/general/Harpy_feature_calculation.ipynb)

  💡 Want us to add support for another segmentation method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

---

- **🧠 Deep Feature Extraction**

  Discover how `Harpy` enables fast, scalable extraction of deep, cell-level features from multiplex imaging data with the [**KRONOS**](https://github.com/mahmoodlab/KRONOS) foundation model for proteomics. 👉 [Tutorial](../docs/tutorials/general/Featurize_with_kronos.ipynb)

  💡 Want us to add support for another deep feature extraction method?
  👉 [Open an issue](https://github.com/saeyslab/harpy/issues) and let us know!

---

- **🔬 Shallow Feature Extraction**

  Learn to extract shallow features—such as **mean**, **median**, and **standard deviation** of intensities—from multiplex imaging data with `Harpy`. 👉 [Tutorial](../docs/tutorials/advanced/Harpy_aggregate_rasters.ipynb)

---

- **🧬 Spatial Transcriptomics**

  Learn how to analyze spatial transcriptomics data with `Harpy`. For detailed information, refer to the [**SPArrOW documentation**](https://sparrow-pipeline.readthedocs.io/en/latest). 👉 [Tutorial](../docs/tutorials/advanced/Harpy_transcriptomics.ipynb)

---

📚 For a complete list of tutorials, visit the [**tutorials section**](https://harpy.readthedocs.io/en/latest/tutorials).

```{eval-rst}
.. card:: Installation
    :link: installation
    :link-type: doc

    Learn how to install Harpy.

.. card:: Tutorials
    :link: tutorials/index
    :link-type: doc

    Tutorials to help you get up to speed with Harpy.

.. card:: Usage
    :link: usage
    :link-type: doc

    Learn how Harpy can be integrated into your workflow.

.. card:: API
    :link: api
    :link-type: doc

    Find a detailed documentation of Harpy.


.. card:: HPC
    :link: tutorials/hpc/index
    :link-type: doc

    Learn how to run Harpy in a High-Performance Computing (HPC) environment.


.. card:: Contributing
    :link: contributing
    :link-type: doc

    Learn how to contribute to Harpy.

```

The Harpy Python package described here builds on the spatial transcriptomics analysis tool [SPArrOW](https://github.com/saeyslab/napari-sparrow).

For spatial proteomics analysis, cite [the Harpy GitHub repository](https://github.com/saeyslab/harpy).

For spatial transcriptomics analysis, see the [SPArrOW preprint](https://www.biorxiv.org/content/10.1101/2024.07.04.601829v1) for citation and to learn more.

```{eval-rst}
.. note::
   This library is currently under active development.
```

```{toctree}
:hidden: true
:maxdepth: 2

installation.md
usage.md
tutorials/index.md
api.md
contributing.md
```
