from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd

from harpy.image import get_dataarray
from harpy.table.canonical_centers._models import (
    CanonicalCacheReport,
    CanonicalCacheState,
    CanonicalCacheUpdatePayload,
    CanonicalCentersResult,
    CanonicalRegionBinding,
)
from harpy.table.canonical_centers._schema import (
    CANONICAL_OBSM_KEY,
    build_canonical_cache_update_payload,
    build_canonical_source_signature,
)
from harpy.utils import RasterAggregator

if TYPE_CHECKING:
    from spatialdata import SpatialData


def calculate_canonical_centers(
    sdata: SpatialData,
    report: CanonicalCacheReport,
) -> CanonicalCacheUpdatePayload:
    """Calculate intrinsic ``(z, y, x)`` centers without mutating a table.

    The source labels are read at ``scale0`` and normalized to three dimensions:
    a 2D ``(y, x)`` raster receives a singleton ``z`` axis, while a 3D
    ``(z, y, x)`` raster is used directly. Only the instance IDs captured in
    ``report.binding`` are reduced, and the returned center rows retain that
    exact instance order.
    """
    if not isinstance(report, CanonicalCacheReport):
        raise TypeError("Canonical center calculation requires a CanonicalCacheReport.")

    labels_name = report.labels_name
    current_source = build_canonical_source_signature(sdata, labels_name)
    if current_source != report.source_signature:
        raise ValueError("Labels source changed after canonical cache inspection; calculation was rejected.")

    labels = get_dataarray(sdata, labels_name, scale="scale0")
    mask = da.asarray(labels.data)
    if report.source_signature.dims == ("y", "x"):
        mask = mask[None, ...]
    _validate_instance_ids_fit_labels_dtype(report.binding, mask.dtype)
    centers = _calculate_centers_with_raster_aggregator(mask, report.binding)
    return build_canonical_cache_update_payload(
        binding=report.binding,
        centers=centers,
        source_signature=report.source_signature,
    )


def read_canonical_centers_from_cache(
    sdata: SpatialData,
    report: CanonicalCacheReport,
) -> CanonicalCentersResult:
    """Read one selected region from an already-inspected valid payload."""
    if not isinstance(report, CanonicalCacheReport):
        raise TypeError("Canonical cache reading requires a CanonicalCacheReport.")
    if report.state is not CanonicalCacheState.VALID:
        raise ValueError("Canonical cache reading requires a valid cache report.")

    table = sdata.tables[report.table_name]
    centers = np.asarray(table.obsm[CANONICAL_OBSM_KEY])[report.binding.row_positions]
    return CanonicalCentersResult(
        source_signature=report.source_signature,
        binding=report.binding,
        centers=centers,
        cache_update=None,
    )


def _validate_instance_ids_fit_labels_dtype(binding: CanonicalRegionBinding, labels_dtype: np.dtype) -> None:
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
    binding: CanonicalRegionBinding,
) -> np.ndarray:
    aggregator = RasterAggregator(
        mask_dask_array=labels,
        image_dask_array=None,
        instance_key=binding.instance_key,
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
    expected_columns = [*coordinate_columns, binding.instance_key]
    if result.columns.tolist() != expected_columns:
        raise ValueError("RasterAggregator center_of_mass must return z, y, x, and instance ID columns in that order.")

    output_ids = result[binding.instance_key].to_numpy()
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
