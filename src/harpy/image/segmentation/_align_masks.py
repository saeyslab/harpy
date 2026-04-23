from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image.segmentation._map import map_labels


def align_labels(
    sdata: SpatialData,
    labels_name_1: str,
    labels_name_2: str,
    threshold: float = 0.0,
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
    Align two labels elements.

    This function aligns two labels elements by examining the labels in `labels_name_1`
    and identifying their maximum overlap with labels in `labels_name_2`.
    It then updates the labels in `labels_name_1`, reassigning them to match the corresponding overlapping label values from `labels_name_2`.
    If there is no overlap with a label from `labels_name_1` with `labels_name_2`, the label in `labels_name_1` is set to zero.
    The function can also generate a shapes element based on the resulting `output_labels_name`.
    The elements are identified by their names and must exist within the passed SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels elements to be aligned.
    labels_name_1
        The name of the first labels element to align.
    labels_name_2
        The name of the second labels element to align.
    threshold
        Minimum required overlap between a label in `labels_name_1` and any label in `labels_name_2`.
        If the overlap fraction is less than this threshold, the label is set to 0 in `output_labels_name`.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell size to avoid chunking effects.
    chunks
        The desired chunk size for the Dask computation in 'y' and 'x', or "auto" to allow the function to
        choose an optimal chunk size based on the data.
    output_labels_name
        The name for the new labels element generated after alignment. If None and overwrite is False,
        a ValueError is raised. If None and overwrite is True, `labels_name_1` will be overwritten
        with the aligned element. Default is None.
    output_shapes_name
        The name for the new shapes element generated from the aligned labels element. If None, no shapes
        element is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, allows the function to overwrite the data in `output_labels_name` and `output_shapes_name` with the aligned data.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The modified SpatialData object with the aligned labels element.

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
    >>> sdata = align_labels(sdata, 'layer_1', 'layer_2', depth=(50, 50), overwrite=True)
    """
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1 (inclusive)."
    sdata = map_labels(
        sdata,
        func=_relabel_array_1_to_array_2_per_chunk,
        threshold=threshold,
        labels_layers=[labels_name_1, labels_name_2],
        depth=depth,
        chunks=chunks,
        output_labels_name=output_labels_name,
        output_shapes_name=output_shapes_name,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=False,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )

    return sdata


def _relabel_array_1_to_array_2_per_chunk(array_1: NDArray, array_2: NDArray, threshold: float = 0.0) -> NDArray:
    assert array_1.shape == array_2.shape

    new_array = np.zeros((array_1.shape), dtype=int)

    # Iterate through each unique label in array_1
    for label in np.unique(array_1):
        if label == 0:
            continue  # Skip label 0 as it represents the background

        mask_label_array_1 = array_1 == label
        positions = np.where(mask_label_array_1)

        overlapping_labels = array_2[positions]

        label_areas = {lbl: np.sum(overlapping_labels == lbl) for lbl in np.unique(overlapping_labels)}

        # Remove the area count for label 0 (background)
        label_areas.pop(0, None)

        # Find the label with the maximum area
        if label_areas:
            max_label = max(label_areas, key=label_areas.get)
            max_area = label_areas[max_label]
            # Only set new label if the max_label covers more than threshold of overlap
            if max_area / mask_label_array_1.sum() >= threshold:
                new_array[positions] = max_label
            else:
                new_array[positions] = 0
        else:
            new_array[positions] = 0  # set to zero if there is no overlap
    return new_array
