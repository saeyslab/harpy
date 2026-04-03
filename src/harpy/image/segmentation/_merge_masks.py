from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.segmentation import relabel_sequential
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image._image import _get_spatial_element
from harpy.image.segmentation._map import map_labels
from harpy.image.segmentation._utils import _SEG_DTYPE


def merge_labels_layers(
    sdata: SpatialData,
    labels_layer_1: str,
    labels_layer_2: str,
    threshold: float = 0.5,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_layer: str | None = None,
    output_shapes_layer: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
) -> SpatialData:
    """
    Merges two labels layers within a SpatialData object based on a specified threshold.

    This function applies a merge operation between two specified labels layers (`labels_layer_1` and `labels_layer_2`)
    in a SpatialData object. The function will copy all labels from `labels_layer_1` to `output_labels_layer`, and for all labels
    in `labels_layer_2` it will check if they have less than `threshold` overlap with labels from `labels_layer_1`, if so,
    label in `labels_layer_2` will be copied to `output_labels_layer` at locations where 'labels_layer_1' is 0.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels layers to be merged.
    labels_layer_1
        The name of the first labels layer. This layer will get priority.
    labels_layer_2
        The name of the second labels layer to be merged in `labels_layer_1`.
    threshold
        The threshold value to control the merging of labels. This value determines how the merge operation is
        conducted based on the overlap between the labels in `labels_layer_1` and `labels_layer_2`.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Please set depth>cell diameter + distance to avoid chunking effects.
    chunks
        Specification for rechunking the data before applying the merge operation. This parameter defines how the data
        is divided into chunks for processing.
    output_labels_layer
        The name of the output labels layer where the merged results will be stored.
    output_shapes_layer
        The name of the output shapes layer where results will be stored if shape data is produced from the merge operation.
    scale_factors
        Scale factors to apply for multiscale processing.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The `sdata` object with the merged labels layer added to the specified output layer. If `output_shapes_layer` is
    provided, a shapes layer will be created corresponding to this labels layer.

    Raises
    ------
    ValueError
        If any of the specified labels layers cannot be found in `sdata`.

    Notes
    -----
    This function leverages dask for potential parallelism and out-of-core computation, enabling the processing of large
    datasets that may not fit entirely in memory. It is particularly useful in scenarios where two segmentation results
    need to be combined to achieve a more accurate or comprehensive segmentation outcome.
    """
    sdata = map_labels(
        sdata,
        func=_merge_masks_block,
        labels_layers=[labels_layer_1, labels_layer_2],
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=True,
        threshold=threshold,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )
    return sdata


def merge_labels_layers_nuclei(
    sdata: SpatialData,
    labels_layer: str,
    labels_layer_nuclei_expanded: str,
    labels_layer_nuclei: str,
    threshold: float = 0.5,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_layer: str | None = None,
    output_shapes_layer: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
) -> SpatialData:
    """
    Merge labels layers using nuclei segmentation.

    Given a labels layer obtained from nuclei segmentation (`labels_layer_nuclei`),
    and corresponding expanded nuclei (`labels_layer_nuclei_expanded`), e.g. obtained through `harpy.im.expand_labels_layer`,
    this function merges labels in labels layer `labels_layer_nuclei_expanded` with `labels_layer` in the SpatialData object,
    if corresponding nuclei in `labels_layer_nuclei` have less than `threshold` overlap with labels from `labels_layer`.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels layers.
    labels_layer
        The name of the labels layer to merge with nuclei labels.
    labels_layer_nuclei_expanded
        The name of the expanded nuclei labels layer.
    labels_layer_nuclei
        The name of the nuclei labels layer.
    threshold
        The threshold value to control the merging of labels. This value determines how the merge operation is
        conducted based on the overlap between the labels in `labels_layer_nuclei` and `labels_layer`.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Please set depth>cell diameter + distance to avoid chunking effects.
    chunks
        Specification for rechunking the data before applying the merge operation. This parameter defines how the data
        is divided into chunks for processing. If 'auto', the chunking strategy is determined automatically.
    output_labels_layer
        The name of the output labels layer where the merged results will be stored.
    output_shapes_layer
        The name of the output shapes layer where results will be stored if shape data is produced from the merge operation.
    scale_factors
        Scale factors to apply for multiscale processing.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The `sdata` object with the merged labels layer added to the specified output layer.
    If `output_shapes_layer` is provided, a shapes layer will be created corresponding to this labels layer.

    Raises
    ------
    ValueError
        If any of the specified labels layers cannot be found in `sdata`.
    ValueError
        If the labels in `labels_layer_nuclei_expanded` do not match the labels in `labels_layer_nuclei`.

    Notes
    -----
    This function is designed to facilitate the merging of expanded nuclei labels with other label layers within a SpatialData
    object.
    It leverages dask for potential parallelism and out-of-core computation.
    """
    labels_layers = [labels_layer, labels_layer_nuclei_expanded, labels_layer_nuclei]
    for layer in labels_layers:
        if layer not in [*sdata.labels]:
            raise ValueError(f"Layer '{layer}' not found in available label layers '{[*sdata.labels]}' of sdata.")

    se_nuclei_expanded = _get_spatial_element(sdata, labels_layer_nuclei_expanded)
    se_nuclei = _get_spatial_element(sdata, labels_layer_nuclei)

    (
        np.array_equal(da.unique(se_nuclei_expanded.data), da.unique(se_nuclei.data)),
        f"Labels layer '{labels_layer_nuclei_expanded}' should contain same labels as '{labels_layer_nuclei}'.",
    )

    sdata = map_labels(
        sdata,
        func=_merge_masks_nuclei_block,
        labels_layers=labels_layers,
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=True,
        threshold=threshold,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )
    return sdata


def _merge_masks_block(
    array_1: NDArray,  # array_1 gets priority
    array_2: NDArray,
    threshold: float = 0.5,
) -> NDArray:
    # this is func for merging of arrays.
    # we need to relabel to avoid collisions in the merged_masks.
    array_1, _, _ = relabel_sequential(array_1)
    array_2, _, _ = relabel_sequential(array_2, offset=array_1.max() + 1)

    merged_masks = array_1
    unique_labels_2 = np.unique(array_2[array_2 > 0])  # Get unique labels from array_2, excluding zero.
    for label in unique_labels_2:
        area_2 = np.sum(array_2 == label)
        # Calculate the overlap area of array_2's label with array_1
        overlap_area = np.sum((array_2 == label) & (array_1 > 0))
        # Check if more than thresh of the overlap area ( e.g., if thresh==0.5 ->half of the area ) is not in array_1
        if overlap_area <= area_2 * threshold:
            # Find the corresponding area in array_2 and merge it into array_1 (only at places where array_1==0)
            merge_condition = (array_2 == label) & (array_1 == 0)
            array_1[merge_condition] = label
    return merged_masks


def _merge_masks_nuclei_block(array_1: NDArray, array_2: NDArray, array_3: NDArray, threshold: float = 0.5):
    # array_1 is priority segmentation
    # array_2 is expanded_nucleus
    # array_3 is nucleus

    # labels in expanded_nucleus are added to priority_segmentation,
    # if corresponding nucleus does not overlap for more than half with labels in priority_segmentation.

    def _relabel_array(arr, original_values, new_values):
        relabeled_array = np.zeros_like(arr)
        assert original_values.shape == new_values.shape
        for new_label, old_label in zip(new_values, original_values, strict=True):
            relabeled_array[arr == old_label] = new_label
        return relabeled_array

    # array_2 and array_3 can contain different labels due to chunking
    # (i.e. array_3 contains nuclei, which are smaller than expanded nuclei from array_2), but they need to be
    # relabeled in the same way.
    original_values = np.unique(np.concatenate([np.unique(array_2), np.unique(array_3)]))

    new_values = np.arange(original_values.size)
    array_2 = _relabel_array(array_2, original_values=original_values, new_values=new_values)
    array_3 = _relabel_array(array_3, original_values=original_values, new_values=new_values)

    # relabel array_1 to avoid collisions
    array_1, _, _ = relabel_sequential(
        array_1, offset=max(array_2.max(), array_3.max()) + 1
    )  # necessary, to avoid collisions

    unique_labels_3 = np.unique(array_3[array_3 > 0])  # Get unique labels from array_3, excluding zero

    for label in unique_labels_3:
        # Determine the area of the label in array_3
        area_3 = np.sum(array_3 == label)

        # Calculate the overlap area of array_3's label with array_1
        overlap_area = np.sum((array_3 == label) & (array_1 > 0))
        # Check if more than threshold of the overlap area ( e.g., if thresh==0.5 ->half of the area ) is not in array_1
        if overlap_area <= area_3 * threshold:
            # Find the corresponding area in array_2 and merge it into array_1 (only at places where array_1==0)
            merge_condition = (array_2 == label) & (array_1 == 0)
            array_1[merge_condition] = label

    return array_1


def _accumulate_mask_to_original_overlap_counts_chunk(
    mask_block: NDArray,
    original_block: NDArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce one chunk pair to a dense local overlap table.

    The rows correspond to mask labels present in this chunk, the columns to
    original labels present in this chunk, and each entry counts the number of
    overlapping pixels. Background label `0` from the original mask is ignored,
    so labels with only background overlap are omitted from the result.
    """
    mask_block = np.asarray(mask_block)
    original_block = np.asarray(original_block)

    if mask_block.shape != original_block.shape:
        raise ValueError(f"Chunk shape mismatch: {mask_block.shape} != {original_block.shape}.")

    foreground = mask_block > 0
    if not np.any(foreground):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.uint64),
        )

    mask_values = mask_block[foreground].astype(np.int64, copy=False)
    original_values = original_block[foreground].astype(np.int64, copy=False)

    nonzero_original = original_values > 0
    if not np.any(nonzero_original):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.uint64),
        )

    mask_values = mask_values[nonzero_original]
    original_values = original_values[nonzero_original]

    mask_ids, mask_dense = np.unique(mask_values, return_inverse=True)
    original_ids, original_dense = np.unique(original_values, return_inverse=True)

    counts = np.bincount(
        mask_dense * original_ids.size + original_dense,
        minlength=mask_ids.size * original_ids.size,
    ).reshape(mask_ids.size, original_ids.size)

    return mask_ids, original_ids, counts.astype(np.uint64, copy=False)


def _map_mask_ids_to_original_labels(
    mask: np.ndarray | da.Array,
    original: np.ndarray | da.Array,
    mask_ids: np.ndarray | None = None,
    original_ids: np.ndarray | None = None,
) -> dict[int, int]:
    """
    Map each non-zero mask id to the non-zero original label with maximum overlap.

    Overlaps are accumulated chunk by chunk. Each chunk returns a local dense
    overlap table of shape `(n_mask_ids_in_chunk, n_original_ids_in_chunk)`,
    and these local tables are merged into a sparse Python accumulator keyed by
    actual `(mask_id, original_id)` pairs, avoiding a huge dense global matrix.
    """
    if not isinstance(mask, da.Array):
        mask = np.asarray(mask)
    if not isinstance(original, da.Array):
        original = np.asarray(original)

    if mask.shape != original.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match original mask shape {original.shape}.")

    if isinstance(mask, da.Array):
        mask_da = mask
    else:
        mask_da = da.from_array(mask, chunks=original.chunks if isinstance(original, da.Array) else mask.shape)

    if isinstance(original, da.Array):
        original_da = original
    else:
        original_da = da.from_array(original, chunks=mask_da.chunks)

    if mask_da.chunks != original_da.chunks:
        original_da = original_da.rechunk(mask_da.chunks)

    mask_ids_allowed: set[int] | None = None
    if mask_ids is not None:
        mask_ids_array = np.asarray(mask_ids, dtype=np.int64)
        mask_ids_array = np.unique(mask_ids_array)
        mask_ids_array = mask_ids_array[mask_ids_array != 0]
        if mask_ids_array.size == 0:
            return {}
        mask_ids_allowed = {int(mask_id) for mask_id in mask_ids_array}

    original_ids_allowed: set[int] | None = None
    if original_ids is not None:
        original_ids_array = np.asarray(original_ids, dtype=np.int64)
        original_ids_array = np.unique(original_ids_array)
        original_ids_array = original_ids_array[original_ids_array != 0]
        if original_ids_array.size == 0:
            return {}
        original_ids_allowed = {int(original_id) for original_id in original_ids_array}

    mask_blocks = mask_da.to_delayed().ravel()
    original_blocks = original_da.to_delayed().ravel()
    chunk_tasks = [
        dask.delayed(_accumulate_mask_to_original_overlap_counts_chunk)(
            mask_block=mask_block_delayed,
            original_block=original_block_delayed,
        )
        for mask_block_delayed, original_block_delayed in zip(mask_blocks, original_blocks, strict=True)
    ]

    # Sparse global overlap accumulator:
    # `global_counts[mask_id][original_id] = total_overlap_pixels`
    # This avoids allocating a dense `(n_mask_ids_global, n_original_ids_global)` matrix.
    global_counts: dict[int, dict[int, int]] = {}

    for local_mask_ids, local_original_ids, local_counts in dask.compute(*chunk_tasks):
        if local_mask_ids.size == 0 or local_original_ids.size == 0:
            continue

        nonzero_rows, nonzero_cols = np.nonzero(local_counts)
        if nonzero_rows.size == 0:
            continue

        for row_idx, col_idx in zip(nonzero_rows, nonzero_cols, strict=True):
            mask_id = int(local_mask_ids[row_idx])
            original_id = int(local_original_ids[col_idx])
            if mask_ids_allowed is not None and mask_id not in mask_ids_allowed:
                continue
            if original_ids_allowed is not None and original_id not in original_ids_allowed:
                continue

            if mask_id not in global_counts:
                global_counts[mask_id] = {}

            if original_id not in global_counts[mask_id]:
                global_counts[mask_id][original_id] = 0

            global_counts[mask_id][original_id] += int(local_counts[row_idx, col_idx])

    if not global_counts:
        return {}

    return {
        int(mask_id): int(min(overlap_counts.items(), key=lambda item: (-item[1], item[0]))[0])
        for mask_id, overlap_counts in sorted(global_counts.items())
    }


def mask_to_original(
    sdata: SpatialData,
    labels_layer: str,
    original_labels_layers: list[str],
    depth: tuple[int, int] | int = 400,
    chunks: str | int | tuple[int, int] | None = None,
) -> DataFrame:
    """
    Map to original.

    Maps labels from a labels layer (`labels_layer`) to their corresponding labels in original labels layers within a SpatialData object.
    The labels in `labels_layer` will be mapped to the label of the labels layers in `original_labels_layers`
    with which it has maximum overlap.

    Parameters
    ----------
    sdata
        Spatialdata object containing the mask and original labels layers.
    labels_layer
        The name of the labels layer used as a mask for mapping.
    original_labels_layers
        The names of the original labels layers to which the mask labels are mapped.
    depth
        Kept for backward compatibility. The optimized implementation aggregates
        per-chunk overlap counts directly and therefore does not require overlap halos.
    chunks
        Specification for rechunking the data before applying the function. If chunks is a Tuple, they should contain
        desired chunk size for 'y', 'x'. 'auto' allows the function to determine optimal chunking. Setting chunks to a
        relative small size (~1000) will significantly speed up the computations.

    Returns
    -------
        A pandas DataFrame where each row corresponds to a unique cell id from the mask layer, and columns correspond
        to the original labels layers. Each cell in the DataFrame contains the label from the original layer that
        overlaps most with the mask label.

    Raises
    ------
    AssertionError
        If arrays from different labels layers do not have the same shape.
    AssertionError
        If depth is provided as a Tuple but does not match (y, x) dimensions.
    AssertionError
        If chunks is a Tuple, and does not match (y, x) dimensions.
    AssertionError
        If the number of blocks in the z-dimension is not equal to 1.

    Notes
    -----
    This function is designed to facilitate the comparison or integration of segmentation results by mapping mask
    labels back to their original labels.
    """
    labels_arrays = [sdata.labels[labels_layer].data]

    for _labels_layer in original_labels_layers:
        labels_arrays.append(sdata.labels[_labels_layer].data)

    # Check for consistent shapes
    first_shape = labels_arrays[0].shape
    for x_label in labels_arrays:
        assert x_label.shape == first_shape, "Only arrays with same shape are currently supported."

    # First make dimension uniform (z,y,x) and keep z unchunked.
    _labels_arrays = []
    for x_label in labels_arrays:
        if x_label.ndim == 2:
            _labels_arrays.append(x_label[None, ...])
        else:
            _labels_arrays.append(x_label)

    if not isinstance(depth, int):
        assert len(depth) == _labels_arrays[0].ndim - 1, "Please (only) provide depth for ( 'y', 'x')."

    if chunks is not None:
        if not isinstance(chunks, int | str):
            assert len(chunks) == _labels_arrays[0].ndim - 1, "Please (only) provide chunks for ( 'y', 'x')."
            chunks = (_labels_arrays[0].shape[0], chunks[0], chunks[1])

    rechunked_arrays = []
    for x_label in _labels_arrays:
        if chunks is not None:
            x_label = x_label.rechunk(chunks)
        elif x_label.numblocks[0] != 1:
            x_label = x_label.rechunk((x_label.shape[0], *x_label.chunksize[1:]))
        assert x_label.numblocks[0] == 1, (
            f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label.numblocks[0]}`."
        )
        rechunked_arrays.append(x_label)

    cell_ids = np.asarray(da.unique(rechunked_arrays[0]).compute(), dtype=np.int64)
    cell_ids = cell_ids[cell_ids != 0]

    result = np.zeros((cell_ids.size, len(original_labels_layers)), dtype=_SEG_DTYPE)

    for index, original_array in enumerate(rechunked_arrays[1:]):
        mapping = _map_mask_ids_to_original_labels(
            mask=rechunked_arrays[0],
            original=original_array,
            mask_ids=cell_ids,
        )
        result[:, index] = np.asarray([mapping.get(int(cell_id), 0) for cell_id in cell_ids], dtype=_SEG_DTYPE)

    return pd.DataFrame(result, index=cell_ids.astype(str), columns=original_labels_layers)
