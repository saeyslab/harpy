"""Partition-local coordinate selection and binning, independent of feature panels."""

import numpy as np
import pandas as pd

from harpy._spatial_bounds import SpatialBounds


def _select_point_coordinates(
    partition: pd.DataFrame,
    *,
    axes: tuple[str, ...],
    matrix: np.ndarray | None,
    crd: SpatialBounds | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Transform points, apply named bounds, then return a source-row mask and selected XY.

    Transform the full source coordinate vector before cropping: a 3D affine
    may mix source x/y/z into any output axis. Every supplied interval applies
    to that transformed axis, never directly to a source column. Intervals are
    half-open. Project to XY only after all bounds, including z, are applied.
    With no spatial operation, ``matrix=None`` avoids reading x/y altogether.
    Neither this helper nor its caller changes the source points coordinates.
    """
    keep = np.ones(len(partition), dtype=bool)
    if matrix is None:
        return keep, None
    source = partition[list(axes)].to_numpy(dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError("Points used for spatial summaries must have finite coordinates.")
    transformed = source @ matrix[:-1, :-1].T + matrix[:-1, -1]
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed points coordinates must be finite.")
    if crd is not None:
        for column, axis in enumerate(axes):
            bounds = getattr(crd, axis)
            if bounds is not None:
                keep &= (transformed[:, column] >= bounds[0]) & (transformed[:, column] < bounds[1])
    return keep, transformed[keep, :2]


def _transformed_point_xy(partition: pd.DataFrame, *, axes: tuple[str, ...], matrix: np.ndarray) -> pd.DataFrame:
    """Return transformed XY coordinates for lazy dataframe extent reductions.

    Validate finite source and transformed coordinates before min/max, which
    would otherwise silently skip NaNs. Keep every row, regardless of class.
    """
    _, xy = _select_point_coordinates(partition, axes=axes, matrix=matrix, crd=None)
    return pd.DataFrame(xy, columns=["x", "y"], index=partition.index)


def _point_bin_edges(
    bounds: tuple[float, ...],
    bin_size: float,
    *,
    explicit_extent: bool,
    class_count: int = 1,
    max_grid_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build common edges; include observed maxima or clip the final bin to an explicit crop.

    Automatic extents use ``floor((max - min) / bin_size) + 1`` bins, ensuring
    the largest observed coordinate is inside the final half-open bin. An
    explicit crop is already half-open and may end with a narrower bin.

    Calculate both axis lengths using scalars before allocating edge arrays.
    If supplied, ``max_grid_bytes`` limits the final dense uint64 array for
    ``class_count`` classes, not total peak memory. This also catches tiny bin
    sizes before they can create oversized edge arrays themselves.
    """
    axis_bounds = (bounds[:2], bounds[2:])
    counts = []
    for minimum, maximum in axis_bounds:
        span = (maximum - minimum) / bin_size
        if not np.isfinite(span):
            raise ValueError("Spatial extent/bin_size must define a finite grid.")
        count = max(1, int(np.ceil(span)) if explicit_extent else int(np.floor(span)) + 1)
        if explicit_extent:
            # Ceil can overestimate by one at a floating-point edge. Do not
            # count a zero-width terminal bin at the exact crop boundary.
            if count > 1 and minimum + (count - 1) * bin_size >= maximum:
                count -= 1
        elif minimum + count * bin_size <= maximum:
            # Subtraction/division can round an exact edge slightly down (e.g.
            # (10.2 - 10) / 0.2). Automatic extents must still include that point.
            count += 1
        counts.append(count)

    grid_shape = (class_count, counts[1], counts[0])
    grid_bytes = class_count * counts[1] * counts[0] * np.dtype(np.uint64).itemsize
    if max_grid_bytes is not None and grid_bytes > max_grid_bytes:
        raise ValueError(
            f"Spatial count grid shape {grid_shape} requires {grid_bytes:,} bytes (uint64), "
            f"exceeding max_grid_bytes={max_grid_bytes:,}. Increase bin_size, restrict crd, "
            "select fewer feature_classes, set bin_size=None, or raise max_grid_bytes "
            "(None disables this limit)."
        )

    edges = []
    for (minimum, maximum), count in zip(axis_bounds, counts, strict=True):
        axis_edges = minimum + np.arange(count + 1, dtype=np.float64) * bin_size
        if explicit_extent:
            axis_edges[-1] = maximum
        if not np.isfinite(axis_edges).all() or not (np.diff(axis_edges) > 0).all():
            raise ValueError("bin_size is not representable at the requested coordinate magnitude.")
        edges.append(axis_edges)
    return edges[0], edges[1]


def _empty_bin_counts() -> pd.Series:
    index = pd.MultiIndex.from_arrays([[], [], []], names=["feature_class", "y_bin", "x_bin"])
    return pd.Series(index=index, dtype=np.uint64, name="n_points")


def _count_point_bins(xy: np.ndarray, classes: np.ndarray, *, edges: tuple[np.ndarray, np.ndarray]) -> pd.Series:
    """Reduce selected XY points to observed (class, y-bin, x-bin) counts.

    For origin (0, 0) and bin_size=200, point (250, 80) increments bin (y=0,
    x=1). No pixel rounding, raster lookup, smoothing, or normalization occurs.
    Only occupied bins are represented here; the final grid fills the rest
    with zeros. ``classes`` are arbitrary group names, not panel metadata.
    """
    if not len(xy):
        return _empty_bin_counts()
    x_edges, y_edges = edges
    # This is floor((coordinate - origin) / bin_size) for regular bins, but
    # comparing actual edges preserves half-open membership despite floating-
    # point cancellation at nonzero origins. Selection excludes the outer edge.
    x_bin = np.searchsorted(x_edges[1:], xy[:, 0], side="right")
    y_bin = np.searchsorted(y_edges[1:], xy[:, 1], side="right")
    frame = pd.DataFrame(
        {
            "feature_class": classes,
            "y_bin": y_bin,
            "x_bin": x_bin,
        }
    )
    return frame.groupby(["feature_class", "y_bin", "x_bin"], observed=True).size().astype(np.uint64).rename("n_points")
