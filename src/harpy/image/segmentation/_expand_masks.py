from __future__ import annotations

from numpy.typing import NDArray
from skimage.segmentation import expand_labels as skimage_expand_labels
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image.segmentation._map import map_labels


def expand_labels(
    sdata: SpatialData,
    labels_name: str,
    distance: int = 10,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_name: str | None = None,
    output_shapes_name: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
) -> SpatialData:
    """
    Expand cells in the labels element `labels_name` using `skimage.segmentation.expand_labels`.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels element to be expanded.
    labels_name
        The name of the labels element to be expanded.
    distance
        distance passed to skimage.segmentation.expand_labels.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell diameter + distance to avoid chunking effects.
    chunks
        The desired chunk size for the Dask computation, or "auto" to allow the function to
        choose an optimal chunk size based on the data. Default is "auto".
    output_labels_name
        The name of the output labels element where results will be stored. This must be specified.
    output_shapes_name
        The name for the new shapes element generated from the expanded labels element. If None, no shapes
        element is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites `output_labels_name` and `output_shapes_name` if they already exist in `sdata`.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The modified SpatialData object with the expanded labels element.

    Notes
    -----
    The function works with Dask arrays and can handle large datasets that don't fit into memory.

    Examples
    --------
    >>> sdata = expand_labels(
            sdata,
            labels_name='segmentation_mask',
            distance=10,
            depth=(100, 100),
            chunks=(1024, 1024),
            output_labels_name='segmentation_mask_expanded',
            output_shapes_name='segmentation_mask_expanded_boundaries',
            overwrite=True,
        )
    """
    sdata = map_labels(
        sdata,
        labels_name=[labels_name],
        func=_expand_cells,
        depth=depth,
        chunks=chunks,
        output_labels_name=output_labels_name,
        output_shapes_name=output_shapes_name,
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
    x_label = skimage_expand_labels(x_label, distance=distance)

    return x_label
