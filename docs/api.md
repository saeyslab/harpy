# API

```{eval-rst}

Import Harpy as::

    import harpy as hp

.. module:: harpy
```

## IO

I/O.

```{eval-rst}

.. module:: harpy.io
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    io.merscope
    io.xenium
    io.visium_hd
    io.read_transcripts
    io.read_resolve_transcripts
    io.read_merscope_transcripts
    io.read_stereoseq_transcripts
    io.create_sdata

```

## Image

Operations on image and labels layers.

```{eval-rst}

.. module:: harpy.im
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    im.add_image_layer
    im.add_labels_layer
    im.map_image
    im.tiling_correction
    im.enhance_contrast
    im.normalize
    im.min_max_filtering
    im.gaussian_filtering
    im.transcript_density
    im.combine
    im.segment
    im.segment_points
    im.cellpose_callable
    im.instanseg_callable
    im.add_grid_labels_layer
    im.expand_labels_layer
    im.align_labels_layers
    im.map_labels
    im.filter_labels_layer
    im.merge_labels_layers
    im.merge_labels_layers_nuclei
    im.rasterize
    im.mask_to_original
    im.pixel_clustering_preprocess
    im.flowsom
```

## Shape

Operations on shapes (polygons) layers.

```{eval-rst}

.. module:: harpy.sh
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    sh.vectorize
    sh.add_shapes_layer
    sh.filter_shapes_layer
    sh.create_voronoi_boundaries
```

## Table

Operations on table (`AnnData` object) layers.

```{eval-rst}

.. module:: harpy.tb
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    tb.add_table_layer
    tb.allocate
    tb.allocate_intensity
    tb.preprocess_transcriptomics
    tb.preprocess_proteomics
    tb.filter_on_size
    tb.leiden
    tb.kmeans
    tb.score_genes
    tb.score_genes_iter
    tb.correct_marker_genes
    tb.cluster_cleanliness
    tb.nhood_enrichment
    tb.add_regionprop_features
    tb.cluster_intensity
    tb.spatial_pixel_neighbors
    tb.cell_clustering_preprocess
    tb.flowsom
    tb.weighted_channel_expression
```

## Points

Operations on points (`Dask` `DataFrame` object) layers.

```{eval-rst}

.. module:: harpy.pt
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    pt.add_points_layer
```

## Plotting

Plotting functions.

### General plots

```{eval-rst}

.. module:: harpy.pl
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    pl.plot
    pl.plot_image
    pl.plot_shapes
    pl.plot_labels
    pl.histogram
    pl.tiling_correction
    pl.flatfield
    pl.segment
```

### Proteomics plots

```{eval-rst}

.. module:: harpy.pl
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    pl.pixel_clusters
    pl.pixel_clusters_heatmap

    pl.snr_ratio
    pl.group_snr_ratio
    pl.snr_clustermap
    pl.signal_clustermap
    pl.clustermap

    pl.segmentation_coverage
    pl.segmentation_size_boxplot
    pl.segments_per_area
```

### Transcriptomics plots

```{eval-rst}

.. module:: harpy.pl
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    pl.sanity_plot_transcripts_matrix
    pl.analyse_genes_left_out
    pl.transcript_density
    pl.preprocess_transcriptomics
    pl.cluster
    pl.score_genes
    pl.cluster_cleanliness
    pl.nhood_enrichment
```

## Utils

Utility functions.

```{eval-rst}

.. module:: harpy.utils
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    utils.bounding_box_query
```

## Datasets

Dataset loaders.

```{eval-rst}

.. module:: harpy.datasets
.. currentmodule:: harpy

.. autosummary::
    :toctree: generated

    datasets.cluster_blobs
    datasets.multisample_blobs
    datasets.pixie_example
    datasets.macsima_example
    datasets.mibi_example
    datasets.vectra_example
    datasets.resolve_example
    datasets.merscope_example
    datasets.xenium_example
    datasets.visium_hd_example
    datasets.get_registry
    datasets.get_spatialdata_registry
```
