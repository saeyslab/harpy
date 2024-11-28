from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
from dask.array import Array
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image.segmentation._map import map_labels
from harpy.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _merge_masks,
    _rechunk_overlap,
    _substract_depth_from_chunks_size,
)
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def align_labels_layers(
    sdata: SpatialData,
    labels_layer_1: str,
    labels_layer_2: str,
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
    Align two labels layers.

    This function aligns two label layers by examining the labels in `labels_layer_1`
    and identifying their maximum overlap with labels in `labels_layer_2`.
    It then updates the labels in `labels_layer_1`, reassigning them to match the corresponding overlapping label values from `labels_layer_2`.
    If there is no overlap with a label from `labels_layer_1` with `label_layer_2`, the label in `labels_layer_1` is set to zero.
    The function can also generate a shapes layer based on the resulting `output_labels_layer`.
    The layers are identified by their names and must exist within the SpatialData object passed.

    Parameters
    ----------
    sdata
        The spatial data object containing the labels layers to be aligned.
    labels_layer_1
        The name of the first labels layer to align.
    labels_layer_2
        The name of the second labels layer to align.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell size to avoid chunking effects.
    chunks
        The desired chunk size for the Dask computation in 'y' and 'x', or "auto" to allow the function to
        choose an optimal chunk size based on the data. Default is "auto".
    output_labels_layer
        The name for the new labels layer generated after alignment. If None and overwrite is False,
        a ValueError is raised. If None and overwrite is True, 'labels_layer_1' will be overwritten
        with the aligned layer. Default is None.
    output_shapes_layer
        The name for the new shapes layer generated from the aligned labels layer. If None, no shapes
        layer is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, allows the function to overwrite the data in `output_labels_layer` and `output_shapes_layer` with the aligned data.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The modified spatial data object with the aligned labels layer.

    Raises
    ------
    AssertionError
        If the shapes of the label arrays or their translations do not match.

    Notes
    -----
    - The function works with Dask arrays and can handle large datasets that don't fit into memory.
    - Only arrays with the same shape are supported for alignment. Misaligned arrays could be due to
      various reasons, including errors in previous processing steps or physical shifts in the samples.
    - The alignment respects the original labelling but ensures that corresponding areas in both layers
      match after the process.

    Examples
    --------
    >>> sdata = align_labels_layers(sdata, 'layer_1', 'layer_2', depth=(50, 50), overwrite=True)
    """
    sdata = map_labels(
        sdata,
        func=_relabel_array_1_to_array_2_per_chunk,
        labels_layers=[labels_layer_1, labels_layer_2],
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=False,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )

    return sdata


def _align_dask_arrays(
    x_label_1: Array,
    x_label_2: Array,
    **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
):
    # we will align labels of x_label_1 with labels of x_labels_2.

    assert x_label_1.shape == x_label_2.shape, "Only arrays with same shape are currently supported."

    chunks = kwargs.pop("chunks", None)
    depth = kwargs.pop("depth", 100)
    boundary = kwargs.pop("boundary", "reflect")

    if isinstance(depth, int):
        depth = {0: 0, 1: depth, 2: depth}
    else:
        assert len(depth) == x_label_1.ndim, f"Please provide depth for each dimension ({x_label_1.ndim})."
        if x_label_1.ndim == 2:
            depth = {0: 0, 1: depth[0], 2: depth[1]}

    assert depth[0] == 0, "Depth not equal to 0 for 'z' dimension is not supported"

    if chunks is None:
        assert (
            x_label_1.chunksize == x_label_2.chunksize
        ), "If chunks is not specified, please ensure Dask arrays have the same chunksize."

    _check_boundary(boundary)

    _to_squeeze = False
    if x_label_1.ndim == 2:
        _to_squeeze = True
        x_label_1 = x_label_1[None, ...]
        x_label_2 = x_label_2[None, ...]

    #  rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
    x_label_1 = _rechunk_overlap(x_label_1, depth=depth, chunks=chunks)
    x_label_2 = _rechunk_overlap(x_label_2, depth=depth, chunks=chunks)

    assert (
        x_label_1.numblocks[0] == 1
    ), f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label_1.numblocks[0]}`."
    assert (
        x_label_2.numblocks[0] == 1
    ), f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label_2.numblocks[0]}`."

    # output_chunks can be derived from either x_label_1 or x_label_2
    output_chunks = _add_depth_to_chunks_size(x_label_1.chunks, depth)

    x_labels = da.map_overlap(
        lambda m, f: _relabel_array_1_to_array_2_per_chunk(m, f),
        x_label_1,
        x_label_2,
        dtype=_SEG_DTYPE,
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
        depth=depth,
        trim=False,
        boundary="reflect",
        # this reflect is useless for this use case, but clean_up_masks and _merge_masks only support
        # results from map_overlap generated with "reflect", "nearest" and "constant"
    )

    x_labels = da.map_blocks(
        _clean_up_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        depth=depth,
    )

    output_chunks = _substract_depth_from_chunks_size(x_labels.chunks, depth=depth)

    x_labels = da.map_overlap(
        _merge_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        num_blocks=x_labels.numblocks,
        trim=False,
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((7,) ,(1024, 1024, 452), (1024, 1024, 452), (1,) ),
        depth=depth,
        boundary="reflect",
        _depth=depth,
    )

    x_labels = x_labels.rechunk(x_labels.chunksize)

    # squeeze if a trivial dimension was added.
    if _to_squeeze:
        x_labels = x_labels.squeeze(0)

    return x_labels


def _relabel_array_1_to_array_2_per_chunk(array_1: NDArray, array_2: NDArray) -> NDArray:
    assert array_1.shape == array_2.shape

    new_array = np.zeros((array_1.shape), dtype=int)

    # Iterate through each unique label in array_1
    for label in np.unique(array_1):
        if label == 0:
            continue  # Skip label 0 as it represents the background

        positions = np.where(array_1 == label)

        overlapping_labels = array_2[positions]

        label_areas = {lbl: np.sum(overlapping_labels == lbl) for lbl in np.unique(overlapping_labels)}

        # Remove the area count for label 0 (background)
        label_areas.pop(0, None)

        # Find the label with the maximum area
        if label_areas:
            max_label = max(label_areas, key=label_areas.get)
        else:
            max_label = 0  # Set to 0 if there's no overlap

        new_array[positions] = max_label

    return new_array
