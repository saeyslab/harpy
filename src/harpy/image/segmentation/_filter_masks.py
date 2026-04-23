from __future__ import annotations

import dask.array as da
import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from harpy.image._image import add_labels, get_dataarray
from harpy.image.segmentation._utils import _SEG_DTYPE
from harpy.shape._shape import add_shapes
from harpy.utils._aggregate import get_instance_size


def filter_labels(
    sdata: SpatialData,
    labels_name: str,
    min_size: int = 10,
    max_size: int = 100000,
    chunks: str | int | tuple[int, int] | None = None,
    output_labels_name: str | None = None,
    output_shapes_name: str | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Filter labels in a labels element by global object size.

    Labels in `labels_name` whose total size across the full image is smaller
    than `min_size` or larger than `max_size` are set to `0` in the output.
    Size is computed per object globally, so labels that span multiple chunks
    are filtered consistently.

    Parameters
    ----------
    sdata
        The SpatialData object containing the labels element to be filtered.
    labels_name
        The name of the labels element to be filtered.
    min_size
        labels in `labels_name` with size smaller than `min_size` will be set to 0.
    max_size
        labels in `labels_name` with size larger than `max_size` will be set to 0.
    chunks
        The desired chunk size for the Dask computation, or "auto" to allow the function to
        choose an optimal chunk size based on the data.
    output_labels_name
        The name of the output labels element where results will be stored. This must be specified.
    output_shapes_name
        The name for the new shapes element generated from the filtered labels element. If None, no shapes
        element is created. Default is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites `output_labels_name` or `output_shapes_name` if they already exist in `sdata`.

    Returns
    -------
    The modified SpatialData object with the filtered labels element.

    Raises
    ------
    ValueError
        If `output_labels_name` is not provided.
    ValueError
        If `min_size` or `max_size` is negative.
    ValueError
        If `min_size` is larger than `max_size`.

    See Also
    --------
    harpy.utils.get_instance_size : compute global object sizes for a labels mask.

    Notes
    -----
    The function works with Dask arrays and can handle large datasets that do
    not fit into memory.


    Example
    --------
    .. code-block:: python

        sdata = hp.datasets.mibi_example()

        sdata = hp.im.filter_labels(
            sdata,
            labels_name="masks_whole",
            min_size=100,
            max_size=1000,
            chunks=256,
            output_labels_name="masks_whole_filtered",
            output_shapes_name="masks_whole_filtered_boundaries",
            overwrite=True,
        )
    """
    if output_labels_name is None:
        raise ValueError("Please specify a name for the output labels element.")
    if min_size < 0 or max_size < 0:
        raise ValueError(f"'min_size' and 'max_size' must be non-negative, found {min_size} and {max_size}.")
    if min_size > max_size:
        raise ValueError(f"'min_size' must be <= 'max_size', found {min_size} > {max_size}.")

    se_labels = get_dataarray(sdata, element_name=labels_name)
    labels_array = se_labels.data
    transformations = get_transformation(se_labels, get_all=True)

    if not isinstance(labels_array, da.Array):
        labels_array = da.from_array(labels_array, chunks=labels_array.shape)

    if chunks is not None:
        if not isinstance(chunks, int | str):
            expected_dims = labels_array.ndim if labels_array.ndim == 2 else labels_array.ndim - 1
            if len(chunks) != expected_dims:
                raise ValueError("Please (only) provide chunks for ('y', 'x').")
            if labels_array.ndim == 3:
                chunks = (labels_array.shape[0], chunks[0], chunks[1])
        labels_array = labels_array.rechunk(chunks)
    elif labels_array.ndim == 3 and labels_array.numblocks[0] != 1:
        labels_array = labels_array.rechunk((labels_array.shape[0], *labels_array.chunksize[1:]))

    label_ids = np.asarray(da.unique(labels_array).compute(), dtype=np.int64)
    label_ids = label_ids[label_ids != 0]

    if label_ids.size == 0:
        filtered_array = labels_array.astype(_SEG_DTYPE)
    else:
        area_array = labels_array[None, ...] if labels_array.ndim == 2 else labels_array
        instance_sizes = get_instance_size(
            mask=area_array,
            index=label_ids,
            instance_key="instance_id",
            instance_size_key="instance_size",
            run_on_gpu=False,
        )
        kept_label_ids = instance_sizes.loc[
            instance_sizes["instance_size"].between(min_size, max_size, inclusive="both"),
            "instance_id",
        ].to_numpy(dtype=np.int64, copy=False)
        kept_label_ids = np.unique(kept_label_ids)

        filtered_array = da.map_blocks(
            _filter_labels_block_by_size,
            labels_array,
            dtype=_SEG_DTYPE,
            kept_label_ids=kept_label_ids,
        )

    sdata = add_labels(
        sdata,
        filtered_array,
        output_labels_name=output_labels_name,
        chunks=chunks,
        transformations=transformations,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    if output_shapes_name is not None:
        se_result = get_dataarray(sdata, element_name=output_labels_name)
        sdata = add_shapes(
            sdata,
            input=se_result.data,
            output_shapes_name=output_shapes_name,
            transformations=transformations,
            overwrite=overwrite,
        )

    return sdata


def _filter_labels_block_by_size(
    labels_block: NDArray,
    kept_label_ids: np.ndarray,
) -> NDArray:
    labels_block = np.asarray(labels_block)
    result = labels_block.astype(_SEG_DTYPE, copy=True)

    if kept_label_ids.size == 0:
        result[labels_block > 0] = 0
        return result

    foreground = labels_block > 0
    if not np.any(foreground):
        return result

    label_values = labels_block[foreground].astype(np.int64, copy=False)
    label_positions = np.searchsorted(kept_label_ids, label_values)
    label_positions_safe = np.where(label_positions >= kept_label_ids.size, 0, label_positions)
    found_labels = kept_label_ids[label_positions_safe] == label_values

    filtered_values = np.zeros(label_values.shape, dtype=_SEG_DTYPE)
    if np.any(found_labels):
        filtered_values[found_labels] = label_values[found_labels].astype(_SEG_DTYPE, copy=False)

    result[foreground] = filtered_values
    return result
