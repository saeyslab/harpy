from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
from dask.array import unique
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.segmentation import relabel_sequential
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from harpy.image._image import _get_spatial_element
from harpy.image.segmentation._map import map_labels
from harpy.image.segmentation._utils import _SEG_DTYPE, _rechunk_overlap
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


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
    The labels in `labels_layers` will be mapped to the label of the labels layers in `original_labels_layers`
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
        The depth around the boundary of each block to load when the array is split into blocks. This ensures
        that the split doesn't cause misalignment along the edges. Default is 400. Set depth larger than the maximum
        cell diameter to avoid chunking effects.
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

    cell_ids = unique(labels_arrays[0]).compute()

    for _labels_layer in original_labels_layers:
        labels_arrays.append(sdata.labels[_labels_layer].data)

    # Check for consistent shapes
    first_shape = labels_arrays[0].shape
    for x_label in labels_arrays:
        assert x_label.shape == first_shape, "Only arrays with same shape are currently supported."

    # First make dimension uniform (z,y,x).
    _labels_arrays = []
    for x_label in labels_arrays:
        if x_label.ndim == 2:
            _labels_arrays.append(x_label[None, ...])
        else:
            _labels_arrays.append(x_label)

    _x_label = _labels_arrays[0]

    if isinstance(depth, int):
        depth = {0: 0, 1: depth, 2: depth}
    else:
        assert len(depth) == _x_label.ndim - 1, "Please (only) provide depth for ( 'y', 'x')."
        # set depth for every dimension
        depth2 = {0: 0, 1: depth[0], 2: depth[1]}
        depth = depth2

    if chunks is not None:
        if not isinstance(chunks, int | str):
            assert len(chunks) == _x_label.ndim - 1, "Please (only) provide chunks for ( 'y', 'x')."
            chunks = (_x_label.shape[0], chunks[0], chunks[1])

    rechunked_arrays = []
    for x_label in _labels_arrays:
        #  rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x_label = _rechunk_overlap(x_label, depth=depth, chunks=chunks)
        assert x_label.numblocks[0] == 1, (
            f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label.numblocks[0]}`."
        )
        rechunked_arrays.append(x_label)

    def _mask_to_original_chunks(
        x_labels_gs: NDArray,
        *arrays: tuple[NDArray],
        block_info,
        _depth: dict[int, int],
    ):
        def _zero_non_max(list_1, list_2):
            if not list_1 or not list_2 or len(list_1) != len(list_2):
                raise ValueError("Lists should be non-empty and of the same length.")
            max_value = max(list_1)
            for i in range(len(list_1)):
                if list_1[i] != max_value:
                    list_2[i] = 0
            return list_2

        total_blocks = block_info[0]["num-chunks"]
        assert total_blocks[0] == 1, (
            "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
        )
        assert depth[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
        assert len(depth) == 3, "Please provide depth values for z,y and x."

        # x_labels_gs, gold standard labels
        labels_gs = np.unique(x_labels_gs[:, _depth[1] : -_depth[1], _depth[2] : -_depth[2]])

        result = np.zeros((cell_ids.shape[0], len(arrays)), dtype=_SEG_DTYPE)

        for label in labels_gs:
            if label == 0:
                continue
            max_label_list = []
            max_area_list = []

            for _array in arrays:
                positions = np.where(x_labels_gs == label)
                overlapping_labels = _array[positions]

                label_areas = {lbl: np.sum(overlapping_labels == lbl) for lbl in np.unique(overlapping_labels)}
                label_areas.pop(0, None)

                # Find the label with the maximum area
                if label_areas:
                    max_label = max(label_areas, key=label_areas.get)
                    max_area = label_areas[max_label]
                else:
                    max_label = 0  # Set to 0 if there's no overlap
                    max_area = 0

                max_label_list.append(max_label)
                max_area_list.append(max_area)

            max_overlap = _zero_non_max(max_area_list, max_label_list)

            index = np.where(cell_ids == label)[0][0]
            result[index] = max_overlap

        return result

    result = da.map_overlap(
        lambda *arrays, block_info=None, _depth=depth: _mask_to_original_chunks(
            *arrays, block_info=block_info, _depth=_depth
        ),  # Unpack and pass all arrays to _mask_to_original_chunks
        *rechunked_arrays,  # Unpack the list of Dask arrays as individual arguments
        dtype=_SEG_DTYPE,
        drop_axis=0,
        depth=depth,
        trim=False,
        boundary=0,
        chunks=(len(cell_ids), len(original_labels_layers)),
    )

    result_of_chunks = da.zeros((len(cell_ids), len(original_labels_layers)), dtype=result.dtype)

    num_chunks = result.numblocks

    # sum the result for each chunk
    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            current_chunk = result.blocks[i, j]
            condition = result_of_chunks == 0
            result_of_chunks = da.where(
                condition,
                current_chunk,  # if equal to zero, overwrite, else, keep old value
                result_of_chunks,
            )

    result_computed = result_of_chunks.compute()
    df = pd.DataFrame(
        result_computed,
        index=cell_ids,
        columns=original_labels_layers,
    )

    df.drop(0, inplace=True, axis=0)
    df.index = df.index.astype(str)

    return df
