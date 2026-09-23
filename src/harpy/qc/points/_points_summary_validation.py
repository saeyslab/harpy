"""Construction-time spatial contracts shared by point-summary containers."""

import numpy as np
import xarray as xr

from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._points_summary_schema import _FEATURE_KEY


def _validate_spatial_summary(
    grid: xr.DataArray | None,
    *,
    metadata: PointsSummaryMetadata,
    axis: str,
    retained_bin_mask: xr.DataArray | None = None,
) -> None:
    """Check grid/edge consistency and, for feature summaries, the inherited mask.

    Unbinned summaries have no grid, bin edges or stored mask. Checks inspect
    structure and coordinates only: no count reduction, numerical-summary
    validation or source reads. Class masks are derived from the grid and
    therefore need neither computation nor validation here. In-place edits
    after construction are not tracked.

    For example, ``metadata.x_edges = (0, 10, 20, 25)`` and
    ``metadata.y_edges = (0, 10, 20)`` define three columns and two rows.
    A class grid's ``grid.coords`` would look like::

        Coordinates:
          * feature_class  (feature_class)  'Endogenous' 'Negative'
          * y              (y)              5.0  15.0
          * x              (x)              5.0  15.0  22.5

    Class names identify count planes; x/y coordinates identify bin centers.
    Counts live in ``grid.values``, not in these coordinates. X coordinates
    ``(0, 1, 2)`` would be rejected: they are bin indices, not the centers
    implied by these edges. The narrower final x bin has center 22.5.
    """
    if grid is None:
        if any(
            value is not None for value in (metadata.bin_size, metadata.x_edges, metadata.y_edges, retained_bin_mask)
        ):
            raise ValueError("An unbinned summary must have no bin_size, bin edges or retained_bin_mask.")
        return
    if metadata.bin_size is None:
        raise ValueError("A summary with spatial_counts must specify metadata.bin_size.")
    if not isinstance(grid, xr.DataArray) or grid.dims != (axis, "y", "x"):
        raise ValueError(f"spatial_counts must have dimensions ({axis!r}, 'y', 'x').")
    if not isinstance(grid.data, np.ndarray):
        raise ValueError("spatial_counts must contain an in-memory count grid.")
    if axis not in grid.coords or grid[axis].dims != (axis,):
        raise ValueError(f"spatial_counts must identify its {axis!r} planes.")
    names = grid[axis].values.tolist()
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"spatial_counts {axis!r} names must be nonempty and unique strings.")

    for dimension, boundaries in (("x", metadata.x_edges), ("y", metadata.y_edges)):
        boundary = np.asarray(boundaries, dtype=float)
        if (
            boundary.ndim != 1
            or boundary.size != grid.sizes[dimension] + 1
            or boundary.size < 2
            or not np.isfinite(boundary).all()
            or not (np.diff(boundary) > 0).all()
        ):
            raise ValueError(f"metadata.{dimension}_edges must be finite, increasing boundaries matching the grid.")
        centers = boundary[:-1] + np.diff(boundary) / 2
        if (
            dimension not in grid.coords
            or grid[dimension].dims != (dimension,)
            or not np.allclose(grid[dimension].values, centers, rtol=1e-10, atol=1e-10)
        ):
            raise ValueError(f"spatial_counts {dimension!r} coordinates must match the bin centers in metadata.")

    if axis == _FEATURE_KEY and (
        not isinstance(retained_bin_mask, xr.DataArray)
        or retained_bin_mask.dims != ("y", "x")
        or retained_bin_mask.dtype != np.dtype(bool)
        or not isinstance(retained_bin_mask.data, np.ndarray)
        or any(
            dimension not in retained_bin_mask.coords or not retained_bin_mask[dimension].equals(grid[dimension])
            for dimension in ("y", "x")
        )
    ):
        raise ValueError("retained_bin_mask must be a boolean XY grid with the same coordinates as spatial_counts.")
