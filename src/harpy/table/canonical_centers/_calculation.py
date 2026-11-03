from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd

from harpy.image import get_dataarray
from harpy.table.canonical_centers._models import CanonicalRegionMetadata, _CanonicalRegionBinding
from harpy.utils import RasterAggregator
from harpy.utils._keys import _INSTANCE_KEY

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spatialdata import SpatialData


def _calculate_canonical_centers(
    sdata: SpatialData,
    binding: _CanonicalRegionBinding,
    region_metadata: CanonicalRegionMetadata,
) -> NDArray[np.float64]:
    """Calculate intrinsic ``(z, y, x)`` centers of one labels region without mutating a table.

    The labels are read at ``scale0``; 2D ``(y, x)`` labels receive a singleton
    ``z`` axis. Only the instance IDs of ``binding`` are reduced, and the
    returned rows keep their order.
    """
    labels = get_dataarray(sdata, binding.labels_name, scale="scale0")
    mask = da.asarray(labels.data)
    if region_metadata.dims == ("y", "x"):
        mask = mask[None, ...]
    _validate_instance_ids_fit_labels_dtype(binding, mask.dtype)
    return _calculate_centers_with_raster_aggregator(mask, binding)


def _validate_instance_ids_fit_labels_dtype(binding: _CanonicalRegionBinding, labels_dtype: np.dtype) -> None:
    dtype = np.dtype(labels_dtype)
    maximum = int(np.iinfo(dtype).max)
    instance_ids = binding.instance_ids
    too_large = instance_ids > maximum
    if np.any(too_large):
        invalid_ids = instance_ids[too_large]
        raise ValueError(
            f"{len(invalid_ids)} selected instance ID(s) cannot be represented by labels dtype `{dtype.name}`"
            f" ({_format_id_preview(invalid_ids)})."
        )


def _calculate_centers_with_raster_aggregator(
    labels: da.Array,
    binding: _CanonicalRegionBinding,
) -> np.ndarray:
    aggregator = RasterAggregator(
        mask_dask_array=labels,
        image_dask_array=None,
        instance_key=_INSTANCE_KEY,
        run_on_gpu=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        result = aggregator.center_of_mass(index=binding.instance_ids)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("RasterAggregator center_of_mass must return a pandas DataFrame.")
    coordinate_columns = [0, 1, 2]
    expected_columns = [*coordinate_columns, _INSTANCE_KEY]
    if result.columns.tolist() != expected_columns:
        raise ValueError("RasterAggregator center_of_mass must return z, y, x, and instance ID columns in that order.")

    output_ids = result[_INSTANCE_KEY].to_numpy()
    if output_ids.dtype != binding.instance_ids.dtype or not np.array_equal(output_ids, binding.instance_ids):
        raise ValueError("RasterAggregator center_of_mass instance IDs must match the requested IDs in order.")

    try:
        centers = result[coordinate_columns].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("RasterAggregator center_of_mass coordinates must be numeric.") from exc
    finite = np.isfinite(centers).all(axis=1)
    if not finite.all():
        missing_ids = binding.instance_ids[~finite]
        raise ValueError(
            f"Labels element `{binding.labels_name}` has no finite center for {len(missing_ids)} requested instance"
            f" ID(s) ({_format_id_preview(missing_ids)})."
        )
    return centers


def _format_id_preview(instance_ids: np.ndarray, limit: int = 5) -> str:
    preview = ", ".join(str(int(value)) for value in instance_ids[:limit])
    if len(instance_ids) > limit:
        preview += ", ..."
    return preview
