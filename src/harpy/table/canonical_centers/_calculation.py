from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd

from harpy.experimental._moments import MomentAggregator
from harpy.image import get_dataarray
from harpy.table.canonical_centers._models import (
    CanonicalRegionMetadata,
    _CanonicalRegionBinding,
    _CanonicalRegionCenters,
    _format_id_preview,
)
from harpy.table.canonical_centers._schema import (
    _build_canonical_metadata,
    _build_region_metadata,
    _canonical_metadata_to_storage,
)
from harpy.utils._keys import _INSTANCE_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from spatialdata import SpatialData


def _calculate_canonical_centers(sdata: SpatialData, binding: _CanonicalRegionBinding) -> _CanonicalRegionCenters:
    """Calculate intrinsic ``(z, y, x)`` centers of one labels region without mutating a table.

    The labels are read at ``scale0``; 2D ``(y, x)`` labels receive a singleton
    ``z`` axis. Only the instance IDs in ``binding``, those with table rows, get
    a center. Returns the centers, in the order of those IDs, with the table
    rows they belong to (``binding``) and their region metadata.
    """
    metadata = _build_region_metadata(sdata, binding.labels_name)
    labels = get_dataarray(sdata, binding.labels_name, scale="scale0")
    mask = da.asarray(labels.data)
    if metadata.dims == ("y", "x"):
        mask = mask[None, ...]
    _validate_instance_ids_fit_labels_dtype(binding, mask.dtype)
    centers = _calculate_centers(mask, binding)
    return _CanonicalRegionCenters(binding=binding, metadata=metadata, centers=centers)


def _assemble_canonical_components(
    n_obs: int,
    regions: Sequence[_CanonicalRegionCenters],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Place the centers of every labels region in one ``(n_obs, 3)`` matrix and build the stored record.

    Every table row must belong to exactly one region.
    """
    centers = np.full((n_obs, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n_obs, dtype=bool)
    entries: dict[str, CanonicalRegionMetadata] = {}
    for region in regions:
        binding = region.binding
        if binding.labels_name in entries:
            raise ValueError(f"Canonical centers for labels element `{binding.labels_name}` are given more than once.")
        if np.any(binding.row_positions >= n_obs) or covered[binding.row_positions].any():
            raise ValueError("Canonical center bindings overlap or exceed the table rows.")
        centers[binding.row_positions] = region.centers
        covered[binding.row_positions] = True
        entries[binding.labels_name] = region.metadata
    if not covered.all() or not np.isfinite(centers).all():
        raise ValueError("Canonical centers must cover every table row with finite coordinates.")
    return centers, _canonical_metadata_to_storage(_build_canonical_metadata(regions=entries))


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


def _calculate_centers(
    labels: da.Array,
    binding: _CanonicalRegionBinding,
) -> np.ndarray:
    aggregator = MomentAggregator(mask_dask_array=labels, instance_key=_INSTANCE_KEY)
    # an instance ID that does not occur in the labels gets a NaN center; the check below raises for it
    (result,) = aggregator.aggregate_stats("center_of_mass", index=binding.instance_ids)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("The center-of-mass aggregation must return a pandas DataFrame.")
    coordinate_columns = [0, 1, 2]
    expected_columns = [*coordinate_columns, _INSTANCE_KEY]
    if result.columns.tolist() != expected_columns:
        raise ValueError("The center-of-mass aggregation must return z, y, x, and instance ID columns in that order.")

    output_ids = result[_INSTANCE_KEY].to_numpy()
    if output_ids.dtype != binding.instance_ids.dtype or not np.array_equal(output_ids, binding.instance_ids):
        raise ValueError("The center-of-mass aggregation must return the requested instance IDs in order.")

    try:
        centers = result[coordinate_columns].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("The center-of-mass aggregation must return numeric coordinates.") from exc
    finite = np.isfinite(centers).all(axis=1)
    if not finite.all():
        missing_ids = binding.instance_ids[~finite]
        raise ValueError(
            f"Labels element `{binding.labels_name}` has no finite center for {len(missing_ids)} requested instance"
            f" ID(s) ({_format_id_preview(missing_ids)})."
        )
    return centers
