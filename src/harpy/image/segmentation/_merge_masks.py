from __future__ import annotations

from typing import Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.segmentation import relabel_sequential
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from harpy.image._image import add_labels, get_dataarray
from harpy.image.segmentation._map import map_labels
from harpy.image.segmentation._utils import _SEG_DTYPE
from harpy.shape._shape import add_shapes
from harpy.utils._aggregate import get_instance_size


def merge_labels(
    sdata: SpatialData,
    candidate_labels_name: str,
    priority_labels_name: str,
    threshold: float = 0.5,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_name: str | None = None,
    output_shapes_name: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Merge two labels elements using a global object-level overlap rule.

    This function treats `priority_labels_name` as the authoritative segmentation
    and `candidate_labels_name` as a layer that can fill uncovered regions. For
    each non-zero candidate object, overlaps with all non-zero priority objects
    are accumulated globally across the full image. A candidate object is kept if:

    `candidate_fraction = overlap_with_any_priority / area_candidate <= threshold`

    Candidate objects with no overlap with the priority segmentation therefore
    have `candidate_fraction = 0` and are kept. Accepted candidate objects are
    written only at pixels where the priority segmentation is `0`.

    Accepted candidate labels are relabeled above the existing priority labels to
    avoid label-id collisions in the merged result.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels elements to be merged.
    candidate_labels_name
        The name of the labels element containing candidate objects to add to the
        merged result when they satisfy the overlap rule.
    priority_labels_name
        The name of the labels element that takes precedence in the merged result.
    threshold
        Maximum allowed `candidate_fraction` for keeping a candidate object.
        Must lie between 0 and 1. A value of `0` keeps only candidates with no
        overlap with the priority segmentation. A value of `1` keeps all
        candidate objects.
    chunks
        Chunk specification used when rechunking the label arrays before overlap
        accumulation and rendering. If a tuple is provided, it is interpreted as
        the desired `(y, x)` chunk size. If set to `"auto"`, Dask determines the
        chunking.
    output_labels_name
        The name of the output labels element where the merged results will be stored.
    output_shapes_name
        The name of the output shapes element where results will be stored if shape data is produced from the merge operation.
    scale_factors
        Scale factors to apply for multiscale processing.
    overwrite
        If True, overwrites `output_labels_name` or `output_shapes_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the merged labels element added to the specified output element. If `output_shapes_name` is
    provided, a shapes element will be created corresponding to this labels element.

    Raises
    ------
    ValueError
        If the provided labels elements do not have the same shape or transformations.
    ValueError
        If `threshold` is outside the interval `[0, 1]`.
    ValueError
        If candidate label ids can not be relabeled safely into `_SEG_DTYPE`
        without colliding with priority label ids.

    See Also
    --------
    harpy.im.match_labels_to_reference : map labels from a merged or processed labels element back to labels in one or more reference elements.

    Example
    --------
    .. code-block:: python

        sdata = hp.datasets.mibi_example()

        sdata = hp.im.merge_labels(
            sdata,
            candidate_labels_name="masks_whole",
            priority_labels_name="masks_nuclear",
            threshold=0.5,
            chunks=256,
            output_labels_name="masks_merged",
            output_shapes_name="masks_merged_boundaries",
            overwrite=True,
        )

    """
    if output_labels_name is None:
        raise ValueError("Please specify a name for the output labels element.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"'threshold' must be between 0 and 1, found {threshold}.")

    candidate_da = get_dataarray(sdata, layer=candidate_labels_name)
    priority_da = get_dataarray(sdata, layer=priority_labels_name)
    candidate_transformations = get_transformation(candidate_da, get_all=True)
    priority_transformations = get_transformation(priority_da, get_all=True)

    if candidate_da.shape != priority_da.shape:
        raise ValueError(
            "Only arrays with same shape are currently supported, "
            f"but candidate labels element '{candidate_labels_name}' has shape {candidate_da.shape}, "
            f"while priority labels element '{priority_labels_name}' has shape {priority_da.shape}."
        )
    if candidate_transformations != priority_transformations:
        raise ValueError(
            f"Provided labels elements '{candidate_labels_name}' and '{priority_labels_name}' "
            "should have the same transformations defined on them."
        )

    candidate_array = candidate_da.data
    priority_array = priority_da.data

    if not isinstance(candidate_array, da.Array):
        candidate_array = da.from_array(candidate_array, chunks=candidate_array.shape)
    if not isinstance(priority_array, da.Array):
        priority_array = da.from_array(priority_array, chunks=priority_array.shape)

    if chunks is not None:
        if not isinstance(chunks, int | str):
            expected_dims = candidate_array.ndim if candidate_array.ndim == 2 else candidate_array.ndim - 1
            if len(chunks) != expected_dims:
                raise ValueError("Please (only) provide chunks for ('y', 'x').")
            if candidate_array.ndim == 3:
                chunks = (candidate_array.shape[0], chunks[0], chunks[1])
        candidate_array = candidate_array.rechunk(chunks)
        priority_array = priority_array.rechunk(candidate_array.chunks)
    elif candidate_array.chunks != priority_array.chunks:
        priority_array = priority_array.rechunk(candidate_array.chunks)

    candidate_ids = np.asarray(da.unique(candidate_array).compute(), dtype=np.int64)
    candidate_ids = candidate_ids[candidate_ids != 0]

    overlap_counts_by_candidate = _get_source_ids_to_reference_overlap_counts(
        source=candidate_array,
        reference=priority_array,
        source_ids=candidate_ids,
    )

    if candidate_array.ndim == 2:
        candidate_area_array = candidate_array[None, ...]
    else:
        candidate_area_array = candidate_array

    candidate_sizes = get_instance_size(
        mask=candidate_area_array,
        index=candidate_ids,
        instance_key="instance_id",
        instance_size_key="instance_size",
        run_on_gpu=False,
    )
    area_by_candidate_id = {
        int(candidate_id): int(area)
        for candidate_id, area in zip(candidate_sizes["instance_id"], candidate_sizes["instance_size"], strict=True)
    }

    kept_candidate_ids: list[int] = []
    for candidate_id in candidate_ids:
        overlap_with_priority = sum(overlap_counts_by_candidate.get(int(candidate_id), {}).values())
        candidate_fraction = overlap_with_priority / area_by_candidate_id[int(candidate_id)]
        if candidate_fraction <= threshold:
            kept_candidate_ids.append(int(candidate_id))

    kept_candidate_ids_array = np.asarray(kept_candidate_ids, dtype=np.int64)
    priority_max = int(da.max(priority_array).compute())
    max_output_value = priority_max + kept_candidate_ids_array.size
    if max_output_value > np.iinfo(_SEG_DTYPE).max:
        raise ValueError(
            f"Relabeling accepted candidate objects would overflow dtype {_SEG_DTYPE} "
            f"(required max label {max_output_value})."
        )

    if kept_candidate_ids_array.size == 0:
        merged_array = priority_array.astype(_SEG_DTYPE)
    else:
        kept_candidate_output_ids = np.arange(
            priority_max + 1,
            priority_max + 1 + kept_candidate_ids_array.size,
            dtype=np.int64,
        )
        merged_array = da.map_blocks(
            _merge_candidate_into_priority_block,
            priority_array,
            candidate_array,
            dtype=_SEG_DTYPE,
            kept_candidate_ids=kept_candidate_ids_array,
            kept_candidate_output_ids=kept_candidate_output_ids,
        )

    sdata = add_labels(
        sdata,
        merged_array,
        output_labels_name=output_labels_name,
        chunks=chunks,
        transformations=priority_transformations,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    if output_shapes_name is not None:
        se_labels = get_dataarray(sdata, layer=output_labels_name)
        sdata = add_shapes(
            sdata,
            input=se_labels.data,
            output_shapes_name=output_shapes_name,
            transformations=priority_transformations,
            overwrite=overwrite,
        )

    return sdata


def merge_labels_nuclei(
    sdata: SpatialData,
    labels_name: str,
    labels_name_nuclei_expanded: str,
    labels_name_nuclei: str,
    threshold: float = 0.5,
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
    Merge labels elements using nuclei segmentation.

    Given a labels element obtained from nuclei segmentation (`labels_name_nuclei`),
    and corresponding expanded nuclei (`labels_name_nuclei_expanded`), e.g. obtained through `harpy.im.expand_labels`,
    this function merges labels in labels element `labels_name_nuclei_expanded` with `labels_name` in the SpatialData object,
    if corresponding nuclei in `labels_name_nuclei` have less than `threshold` overlap with labels from `labels_name`.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels elements.
    labels_name
        The name of the labels element to merge with nuclei labels.
    labels_name_nuclei_expanded
        The name of the expanded nuclei labels element.
    labels_name_nuclei
        The name of the nuclei labels element.
    threshold
        The threshold value to control the merging of labels. This value determines how the merge operation is
        conducted based on the overlap between the labels in `labels_name_nuclei` and `labels_name`.
    depth
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Please set depth>cell diameter + distance to avoid chunking effects.
    chunks
        Specification for rechunking the data before applying the merge operation. This parameter defines how the data
        is divided into chunks for processing. If 'auto', the chunking strategy is determined automatically.
    output_labels_name
        The name of the output labels element where the merged results will be stored.
    output_shapes_name
        The name of the output shapes element where results will be stored if shape data is produced from the merge operation.
    scale_factors
        Scale factors to apply for multiscale processing.
    overwrite
        If True, overwrites `output_labels_name` or `output_shapes_name` if it already exists in `sdata`.
    iou_depth
        iou depth used for linking labels.
    iou_threshold
        iou threshold used for linking labels.

    Returns
    -------
    The `sdata` object with the merged labels element added to the specified output element.
    If `output_shapes_name` is provided, a shapes element will be created corresponding to this labels element.

    Raises
    ------
    ValueError
        If any of the specified labels elements cannot be found in `sdata`.
    ValueError
        If the labels in `labels_name_nuclei_expanded` do not match the labels in `labels_name_nuclei`.

    Notes
    -----
    This function is designed to facilitate the merging of expanded nuclei labels with other labels elements within a SpatialData
    object.
    It leverages dask for potential parallelism and out-of-core computation.
    """
    labels_layers = [labels_name, labels_name_nuclei_expanded, labels_name_nuclei]
    for layer in labels_layers:
        if layer not in [*sdata.labels]:
            raise ValueError(f"Labels element '{layer}' not found in available labels elements '{[*sdata.labels]}' of sdata.")

    se_nuclei_expanded = get_dataarray(sdata, labels_name_nuclei_expanded)
    se_nuclei = get_dataarray(sdata, labels_name_nuclei)

    (
        np.array_equal(da.unique(se_nuclei_expanded.data), da.unique(se_nuclei.data)),
        f"Labels element '{labels_name_nuclei_expanded}' should contain same labels as '{labels_name_nuclei}'.",
    )

    sdata = map_labels(
        sdata,
        func=_merge_masks_nuclei_block,
        labels_layers=labels_layers,
        depth=depth,
        chunks=chunks,
        output_labels_name=output_labels_name,
        output_shapes_name=output_shapes_name,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=True,
        threshold=threshold,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )
    return sdata


def _merge_candidate_into_priority_block(
    priority_block: NDArray,
    candidate_block: NDArray,
    kept_candidate_ids: np.ndarray,
    kept_candidate_output_ids: np.ndarray,
) -> NDArray:
    """
    Merge one candidate chunk into one priority chunk using globally accepted ids.

    The merge is hierarchical: priority labels are copied first, and accepted
    candidate labels are only written at pixels where the priority chunk is `0`.
    Candidate ids are remapped to `kept_candidate_output_ids` so the merged
    result does not collide with existing priority label ids.

    The `searchsorted` section below works as follows:

    1. Extract candidate ids at writable pixels.
    2. Look up where those ids would appear in the sorted array
       `kept_candidate_ids`.
    3. Make those positions safe for indexing.
    4. Compare the looked-up values to the original candidate ids to determine
       which ids were actually accepted.

    This works because for a sorted array:

    - if a value is present, `searchsorted` points to that value
    - if a value is absent, `searchsorted` points to the insertion position
    - the equality check then distinguishes a real match from an insertion point

    For example, if:

    - `candidate_values = [2, 7, 9, 5]`
    - `kept_candidate_ids = [2, 5, 8]`

    then:

    - `searchsorted(...) -> [0, 2, 3, 1]`
    - safe indexing positions -> `[0, 2, 0, 1]`
    - `kept_candidate_ids[...] -> [2, 8, 2, 5]`
    - comparing back to `candidate_values` gives
      `found_candidates = [True, False, False, True]`
    - if `kept_candidate_output_ids = [6, 7, 8]`, then the accepted positions
      map to `kept_candidate_output_ids[[0, 1]] = [6, 7]`
    - `mapped_values` therefore becomes `[6, 0, 0, 7]`

    So candidate ids `2` and `5` are accepted in this block, while `7` and `9`
    are ignored, and only the accepted ones will be written into the writable
    pixels of the result block.
    """
    priority_block = np.asarray(priority_block)
    candidate_block = np.asarray(candidate_block)

    if priority_block.shape != candidate_block.shape:
        raise ValueError(f"Block shape mismatch: {priority_block.shape} != {candidate_block.shape}.")

    result = priority_block.astype(_SEG_DTYPE, copy=True)
    if kept_candidate_ids.size == 0:
        return result

    write_mask = (result == 0) & (candidate_block > 0)
    if not np.any(write_mask):
        return result

    candidate_values = candidate_block[write_mask].astype(np.int64, copy=False)
    candidate_positions = np.searchsorted(kept_candidate_ids, candidate_values)
    candidate_positions_safe = np.where(
        candidate_positions >= kept_candidate_ids.size,
        0,
        candidate_positions,
    )
    found_candidates = kept_candidate_ids[candidate_positions_safe] == candidate_values

    mapped_values = np.zeros(candidate_values.shape, dtype=_SEG_DTYPE)
    if np.any(found_candidates):
        mapped_values[found_candidates] = kept_candidate_output_ids[candidate_positions_safe[found_candidates]].astype(
            _SEG_DTYPE, copy=False
        )

    result[write_mask] = mapped_values
    return result


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


def _accumulate_source_to_reference_overlap_counts_chunk(
    source_block: NDArray,
    reference_block: NDArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce one chunk pair to a dense local overlap table.

    The rows correspond to source labels present in this chunk, the columns to
    reference labels present in this chunk, and each entry counts the number of
    overlapping pixels. Background label `0` from the reference labels is ignored,
    so labels with only background overlap are omitted from the result.
    """
    source_block = np.asarray(source_block)
    reference_block = np.asarray(reference_block)

    if source_block.shape != reference_block.shape:
        raise ValueError(f"Chunk shape mismatch: {source_block.shape} != {reference_block.shape}.")

    foreground = source_block > 0
    if not np.any(foreground):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.uint64),
        )

    source_values = source_block[foreground].astype(np.int64, copy=False)
    reference_values = reference_block[foreground].astype(np.int64, copy=False)

    nonzero_reference = reference_values > 0
    if not np.any(nonzero_reference):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.uint64),
        )

    source_values = source_values[nonzero_reference]
    reference_values = reference_values[nonzero_reference]

    source_ids, source_dense = np.unique(source_values, return_inverse=True)
    reference_ids, reference_dense = np.unique(reference_values, return_inverse=True)

    counts = np.bincount(
        source_dense * reference_ids.size + reference_dense,
        minlength=source_ids.size * reference_ids.size,
    ).reshape(source_ids.size, reference_ids.size)

    return source_ids, reference_ids, counts.astype(np.uint64, copy=False)


def _get_source_ids_to_reference_overlap_counts(
    source: np.ndarray | da.Array,
    reference: np.ndarray | da.Array,
    source_ids: np.ndarray | None = None,
    reference_ids: np.ndarray | None = None,
) -> dict[int, dict[int, int]]:
    """
    Accumulate sparse overlap counts between non-zero source ids and reference ids.

    Overlaps are accumulated chunk by chunk. Each chunk returns a local dense
    overlap table of shape `(n_source_ids_in_chunk, n_reference_ids_in_chunk)`,
    and these local tables are merged into a sparse Python accumulator keyed by
    actual `(source_id, reference_id)` pairs, avoiding a huge dense global matrix.
    Source ids that overlap only with background label `0` in `reference` are not
    included in the returned mapping.

    The returned object is a nested dictionary of the form
    `overlap_counts_by_source[source_id][reference_id] = overlap_pixels`.

    For example, if source label `1` overlaps reference label `5` in `12` pixels
    and reference label `7` in `3` pixels, and source label `2` overlaps reference
    label `8` in `20` pixels, the function returns:

    `{1: {5: 12, 7: 3}, 2: {8: 20}}`

    This means:

    - source label `1` has two candidate matches, with `5` being the stronger one
    - source label `2` overlaps only reference label `8`
    """
    if not isinstance(source, da.Array):
        source = np.asarray(source)
    if not isinstance(reference, da.Array):
        reference = np.asarray(reference)

    if source.shape != reference.shape:
        raise ValueError(f"Source shape {source.shape} does not match reference shape {reference.shape}.")

    if isinstance(source, da.Array):
        source_da = source
    else:
        source_da = da.from_array(source, chunks=reference.chunks if isinstance(reference, da.Array) else source.shape)

    if isinstance(reference, da.Array):
        reference_da = reference
    else:
        reference_da = da.from_array(reference, chunks=source_da.chunks)

    if source_da.chunks != reference_da.chunks:
        reference_da = reference_da.rechunk(source_da.chunks)

    source_ids_allowed: set[int] | None = None
    if source_ids is not None:
        source_ids_array = np.asarray(source_ids, dtype=np.int64)
        source_ids_array = np.unique(source_ids_array)
        source_ids_array = source_ids_array[source_ids_array != 0]
        if source_ids_array.size == 0:
            return {}
        source_ids_allowed = {int(source_id) for source_id in source_ids_array}

    reference_ids_allowed: set[int] | None = None
    if reference_ids is not None:
        reference_ids_array = np.asarray(reference_ids, dtype=np.int64)
        reference_ids_array = np.unique(reference_ids_array)
        reference_ids_array = reference_ids_array[reference_ids_array != 0]
        if reference_ids_array.size == 0:
            return {}
        reference_ids_allowed = {int(reference_id) for reference_id in reference_ids_array}

    source_blocks = source_da.to_delayed().ravel()
    reference_blocks = reference_da.to_delayed().ravel()
    chunk_tasks = [
        dask.delayed(_accumulate_source_to_reference_overlap_counts_chunk)(
            source_block=source_block_delayed,
            reference_block=reference_block_delayed,
        )
        for source_block_delayed, reference_block_delayed in zip(source_blocks, reference_blocks, strict=True)
    ]

    # Sparse global overlap accumulator:
    # `global_counts[source_id][reference_id] = total_overlap_pixels`
    # This avoids allocating a dense `(n_source_ids_global, n_reference_ids_global)` matrix.
    global_counts: dict[int, dict[int, int]] = {}

    for local_source_ids, local_reference_ids, local_counts in dask.compute(*chunk_tasks):
        if local_source_ids.size == 0 or local_reference_ids.size == 0:
            continue

        nonzero_rows, nonzero_cols = np.nonzero(local_counts)
        if nonzero_rows.size == 0:
            continue

        for row_idx, col_idx in zip(nonzero_rows, nonzero_cols, strict=True):
            source_id = int(local_source_ids[row_idx])
            reference_id = int(local_reference_ids[col_idx])
            if source_ids_allowed is not None and source_id not in source_ids_allowed:
                continue
            if reference_ids_allowed is not None and reference_id not in reference_ids_allowed:
                continue

            if source_id not in global_counts:
                global_counts[source_id] = {}

            if reference_id not in global_counts[source_id]:
                global_counts[source_id][reference_id] = 0

            global_counts[source_id][reference_id] += int(local_counts[row_idx, col_idx])

    return {int(source_id): overlap_counts for source_id, overlap_counts in sorted(global_counts.items())}


def match_labels_to_reference(
    sdata: SpatialData,
    source_labels_name: str,
    reference_labels_layers: list[str],
    chunks: str | int | tuple[int, int] | None = None,
    threshold: float = 0.0,
    overlap_metric: Literal["source_fraction", "reference_fraction", "iou"] = "source_fraction",
) -> DataFrame:
    """
    Match source labels to reference labels based on an overlap score.

    For each non-zero label in `source_labels_name`, this function determines, for
    every labels element in `reference_labels_layers`, which non-zero reference
    label best matches it according to `overlap_metric`. The result is returned as
    a :class:`~pandas.DataFrame` indexed by the source labels, with one column per
    reference labels element.

    With the default parameters `threshold=0` and
    `overlap_metric="source_fraction"`, the function effectively
    assigns each source label to the reference label with the largest
    non-zero overlap.

    Overlap counts are accumulated chunk by chunk using a local dense overlap
    table per chunk pair and a sparse global accumulator across chunks. This
    keeps the implementation suitable for large label images without requiring a
    dense global `(n_source_labels, n_reference_labels)` overlap matrix.

    Parameters
    ----------
    sdata
        The input SpatialData object containing the source labels element and the reference
        labels elements.
    source_labels_name
        Name of the labels element whose non-zero labels are matched to the
        reference labels elements.
    reference_labels_layers
        Names of the reference labels elements against which overlap is computed.
        One output column is produced for each layer in the order provided.
    chunks
        Chunk specification used when rechunking the label arrays before the
        overlap computation. If a tuple is provided, it is interpreted as the
        desired `(y, x)` chunk size. If set to `"auto"`, Dask determines the
        chunking. Smaller spatial chunks can improve performance by reducing the
        size of the per-chunk overlap tables.
    threshold
        Minimum required overlap fraction between a source label and its
        best-matching reference label. The overlap fraction is computed as
        a score controlled by `overlap_metric`. If this score is not strictly
        greater than `threshold`, the mapping is discarded and the output value
        is set to `0`. Must lie between 0 and 1.
    overlap_metric
        Metric used both to select the winning reference label and to apply
        `threshold` to that winning match. Supported values are:

        - `"source_fraction"`: `overlap_pixels / area_source_label`
        - `"reference_fraction"`: `overlap_pixels / area_reference_label`
        - `"iou"`: `overlap_pixels / (area_source_label + area_reference_label - overlap_pixels)`

    Returns
    -------
    A pandas DataFrame where each row corresponds to a non-zero label from
    `source_labels_name` and each column corresponds to one layer in
    `reference_labels_layers`. Every value contains the non-zero reference label
    selected for that source label according to `overlap_metric`. If a source label has no non-zero
    overlap with a given reference labels element, the corresponding output value
    is `0`.

    Raises
    ------
    AssertionError
        If the provided labels elements do not all have the same shape.
    AssertionError
        If `chunks` is provided as a tuple but does not match the `(y, x)`
        dimensions.
    AssertionError
        If any rechunked array has more than one chunk along the z dimension.
    ValueError
        If `threshold` is outside the interval `[0, 1]`.
    ValueError
        If `overlap_metric` is not one of `"source_fraction"`,
        `"reference_fraction"`, or `"iou"`.

    Notes
    -----
    Background label `0` is ignored when computing overlaps. As a result,
    output value `0` indicates that a source label has no non-zero overlap with
    the corresponding reference labels element.

    Example
    --------
    .. code-block:: python

        sdata = hp.datasets.mibi_example()

        matched = hp.im.match_labels_to_reference(
            sdata,
            source_labels_name="masks_whole",
            reference_labels_layers=["masks_nuclear"],
            chunks=256,
        )
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"'threshold' must be between 0 and 1, found {threshold}.")
    if overlap_metric not in {"source_fraction", "reference_fraction", "iou"}:
        raise ValueError(
            f"'overlap_metric' must be one of 'source_fraction', 'reference_fraction', or 'iou', found {overlap_metric!r}."
        )

    label_arrays = [get_dataarray(sdata, layer=source_labels_name).data]

    for _labels_layer in reference_labels_layers:
        label_arrays.append(get_dataarray(sdata, layer=_labels_layer).data)

    # Check for consistent shapes
    first_shape = label_arrays[0].shape
    for x_label in label_arrays:
        assert x_label.shape == first_shape, "Only arrays with same shape are currently supported."

    # First make dimension uniform (z,y,x) and keep z unchunked.
    _labels_arrays = []
    for x_label in label_arrays:
        if x_label.ndim == 2:
            _labels_arrays.append(x_label[None, ...])
        else:
            _labels_arrays.append(x_label)

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

    source_ids = np.asarray(da.unique(rechunked_arrays[0]).compute(), dtype=np.int64)
    source_ids = source_ids[source_ids != 0]

    if source_ids.size == 0:
        return pd.DataFrame(
            np.empty((0, len(reference_labels_layers)), dtype=_SEG_DTYPE),
            index=pd.Index([], dtype=str),
            columns=reference_labels_layers,
        )

    result = np.zeros((source_ids.size, len(reference_labels_layers)), dtype=_SEG_DTYPE)
    if overlap_metric == "iou" or (threshold > 0 and overlap_metric == "source_fraction"):
        log.info(f"Calculating instance sizes for source labels element '{source_labels_name}'.")
        instance_sizes = get_instance_size(
            mask=rechunked_arrays[0],
            index=source_ids,
            instance_key="instance_id",
            instance_size_key="instance_size",
            run_on_gpu=False,
        )
        area_by_source_id = {
            int(source_id): int(area)
            for source_id, area in zip(instance_sizes["instance_id"], instance_sizes["instance_size"], strict=True)
        }
    else:
        area_by_source_id = {}

    for index, reference_array in enumerate(rechunked_arrays[1:]):
        overlap_counts_by_source = _get_source_ids_to_reference_overlap_counts(
            source=rechunked_arrays[0],
            reference=reference_array,
            source_ids=source_ids,
        )
        if overlap_metric in {"reference_fraction", "iou"} and overlap_counts_by_source:
            log.info(f"Calculating instance sizes for reference labels element '{reference_labels_layers[index]}'.")
            candidate_reference_ids = np.unique(
                np.asarray(
                    [
                        reference_id
                        for overlap_counts in overlap_counts_by_source.values()
                        for reference_id in overlap_counts
                    ],
                    dtype=np.int64,
                )
            )
            reference_sizes = get_instance_size(
                mask=reference_array,
                index=candidate_reference_ids,
                instance_key="instance_id",
                instance_size_key="instance_size",
                run_on_gpu=False,
            )
            area_by_reference_id = {
                int(reference_id): int(area)
                for reference_id, area in zip(
                    reference_sizes["instance_id"], reference_sizes["instance_size"], strict=True
                )
            }
        else:
            area_by_reference_id = {}

        mapped_labels = []
        for source_id in source_ids:
            overlap_counts = overlap_counts_by_source.get(int(source_id))
            if overlap_counts is None:
                mapped_labels.append(0)
                continue

            best_reference_id = 0
            best_overlap_pixels = 0
            best_score = -1.0
            for reference_id, overlap_pixels in overlap_counts.items():
                if overlap_metric == "source_fraction":
                    score = overlap_pixels
                elif overlap_metric == "reference_fraction":
                    score = overlap_pixels / area_by_reference_id[int(reference_id)]
                else:
                    source_area = area_by_source_id[int(source_id)]
                    reference_area = area_by_reference_id[int(reference_id)]
                    score = overlap_pixels / (source_area + reference_area - overlap_pixels)

                if score > best_score or (score == best_score and reference_id < best_reference_id):
                    best_reference_id = int(reference_id)
                    best_overlap_pixels = int(overlap_pixels)
                    best_score = float(score)

            if threshold > 0:
                if overlap_metric == "source_fraction":
                    overlap_score = best_overlap_pixels / area_by_source_id[int(source_id)]
                elif overlap_metric == "reference_fraction":
                    overlap_score = best_overlap_pixels / area_by_reference_id[int(best_reference_id)]
                else:
                    source_area = area_by_source_id[int(source_id)]
                    reference_area = area_by_reference_id[int(best_reference_id)]
                    overlap_score = best_overlap_pixels / (source_area + reference_area - best_overlap_pixels)

                if overlap_score <= threshold:
                    mapped_labels.append(0)
                    continue

            mapped_labels.append(best_reference_id)

        result[:, index] = np.asarray(mapped_labels, dtype=_SEG_DTYPE)

    return pd.DataFrame(result, index=source_ids.astype(str), columns=reference_labels_layers)
