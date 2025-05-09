from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image.segmentation._map import map_labels
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def filter_labels_layer(
    sdata: SpatialData,
    labels_layer: str,
    min_size: int = 10,
    max_size: int = 100000,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_layer: str | None = None,
    output_shapes_layer: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Filter labels in labels layer `labels_layer` of Spatialdata object that have a size less than `min_size` or size greater than `max_size`.

    Parameters
    ----------
    sdata
        The spatialdata object containing the labels layer to be filtered.
    labels_layer
        The name of the labels layer to be filtered.
    min_size
        labels in `labels_layer` with size smaller than `min_size` will be set to 0.
    max_size
        labels in `labels_layer` with size larger than `max_size` will be set to 0.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell diameter to avoid chunking effects.
    chunks
        The desired chunk size for the Dask computation, or "auto" to allow the function to
        choose an optimal chunk size based on the data.
    output_labels_layer
        The name of the output labels layer where results will be stored. This must be specified.
    output_shapes_layer
        The name for the new shapes layer generated from the aligned labels layer. If None, no shapes
        layer is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The modified spatial data object with the filtered labels layers.

    Notes
    -----
    The function works with Dask arrays and can handle large datasets that don't fit into memory.


    Examples
    --------
    >>> sdata = expand_labels_layer(
            sdata,
            labels_layer='layer',
            distance=10,
            depth=(100, 100),
            chunks=(1024, 1024),
            output_labels_layer='layer_expanded',
            output_shapes_layer='layer_expanded_boundaries',
            overwrite=True,
        )
    """
    sdata = map_labels(
        sdata,
        labels_layers=[labels_layer],
        func=_filter_labels_block,
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=False,
        trim=True,
        min_size=min_size,
        max_size=max_size,
        _depth=depth,
    )

    return sdata


def _filter_labels_block(
    x_label: NDArray,
    min_size: int,
    max_size: int,
    _depth: tuple[int, ...] | int = 100,
) -> NDArray:
    # input and output is numpy array of shape (z,y,x)
    assert x_label.ndim == 3
    if isinstance(_depth, int):
        _depth = {0: 0, 1: _depth, 2: _depth}
    else:
        assert len(_depth) == x_label.ndim - 1, "Please (only) provide depth for ( 'y', 'x')."
        # set depth for every dimension
        depth2 = {0: 0, 1: _depth[0], 2: _depth[1]}
        _depth = depth2
    # get the labels that are (at least partially) inside the chunk
    labels_inside_original_chunk = np.unique(x_label[:, _depth[1] : -_depth[1], _depth[2] : -_depth[2]])
    for label in labels_inside_original_chunk:
        if label == 0:
            continue
        position = x_label == label
        size = np.sum(position)

        if size < min_size or size > max_size:
            x_label[position] = 0
    return x_label
