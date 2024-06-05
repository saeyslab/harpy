from __future__ import annotations

from numpy.typing import NDArray
from skimage.segmentation import expand_labels
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image.segmentation._merge_masks import apply_labels_layers
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def expand_labels_layer(
    sdata: SpatialData,
    labels_layer: str,
    distance: int = 10,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = "auto",
    output_labels_layer: str | None = None,
    output_shapes_layer: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
) -> SpatialData:
    """
    Expand cells in labels layer `labels_layer` of Spatialdata object with `distance`, using `skimage.segmentation.expand_labels`.

    Parameters
    ----------
    sdata
        The spatialdata object containing the labels layer to be expanded.
    labels_layer
        The name of the labels layer to be expanded.
    distance
        distance passed to skimage.segmentation.expand_labels.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell diameter + distance to avoid chunking effects.
    chunks
        The desired chunk size for the Dask computation, or "auto" to allow the function to
        choose an optimal chunk size based on the data. Default is "auto".
    output_labels_layer
        The name of the output labels layer where results will be stored. This must be specified.
    output_shapes_layer
        The name for the new shapes layer generated from the aligned labels layer. If None, no shapes
        layer is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the output layers if they already exist in `sdata`.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The modified spatial data object with the expanded labels layer.

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
    sdata = apply_labels_layers(
        sdata,
        labels_layers=[labels_layer],
        func=_expand_cells,
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=False,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
        distance=distance,
    )

    return sdata


def _expand_cells(
    x_label: NDArray,
    distance: int,
) -> NDArray:
    # input and output is numpy array of shape (z,y,x)

    assert x_label.ndim == 3
    x_label = expand_labels(x_label, distance=distance)

    return x_label
