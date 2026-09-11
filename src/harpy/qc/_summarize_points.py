"""Read-only, panel-aware reductions of original points.

``_summarize_point_partition`` validates source feature/class assignments,
selects points, and returns only compact target and occupied-bin counts.
``_merge_point_summaries`` combines these reductions in a bounded-fan-in tree;
no original point rows reach the driver. Class statistics are then derived
from the zero-filled panel counts, without another source scan. If no explicit
extent is supplied, spatial binning first reduces coordinates to four bounds.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import dask
import numpy as np
import pandas as pd
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation

from harpy._feature_panels import (
    _feature_panel_partition_errors,
    _FeaturePanelContract,
    _resolve_points_feature_panel,
    _validate_feature_class_dtype,
)
from harpy._spatial_bounds import SpatialBounds, _normalize_spatial_bounds
from harpy.qc._points_binning import (
    _count_point_bins,
    _point_bin_edges,
    _select_point_coordinates,
    _transformed_point_xy,
)
from harpy.transformations._transformations import _invertible_affine_matrix


@dataclass(frozen=True)
class PointsSummary:
    """Computed summaries of one original points element and its feature panel.

    Attributes
    ----------
    per_target
        One row per selected panel feature, including zero detections. Columns
        are ``feature``, ``feature_class``, ``n_points``, and
        ``within_class_fraction``: this feature's point count divided by the
        total point count of its own class, after selection. Values range
        from 0 to 1; NaN when the class has no points.
    per_class
        One row per selected class: ``n_features``, ``n_zero_features``,
        ``pct_zero_features`` (0–100), ``n_points``,
        ``mean_points_per_feature``, ``median_points_per_feature``,
        ``p95_points_per_feature``, ``top_n`` (requested N), ``n_top_features``
        (min(N, panel size)), and ``pct_points_top_n_features`` (0–100,
        missing for a zero-point class). All target statistics include zeros.
    spatial_counts
        Optional in-memory uint64 DataArray with dimensions
        ``(feature_class, y, x)`` and bin-center x/y coordinates. These
        coordinates and the geometry attributes ``x_edges``, ``y_edges``,
        ``extent``, and ``bin_size`` are expressed in the requested
        ``to_coordinate_system`` and its units, after transforming the source
        points. Coordinates are not rebased to zero or replaced by bin indices.
        The coordinate-system name is stored in
        ``spatial_counts.attrs["to_coordinate_system"]``. Terminal bins may be
        narrower for explicit crops. This is a standalone DataArray, not a
        registered SpatialData image element.
        Counts are raw, unsmoothed, and not normalized by area or panel size.

    Notes
    -----
    Both dataframes contain ``points_name``, ``feature_panel``, and ``sample_id``
    when available. Their ``attrs`` and the grid's ``attrs`` record those same
    identities plus ``to_coordinate_system``, ``crd``, and selected
    ``panel_feature_counts``. No source element name is parsed to guess sample
    identity or physical units.
    ``crd`` is None or normalized ``(xmin, xmax, ymin, ymax)`` bounds, with
    ``zmin, zmax`` appended when supplied, all in ``to_coordinate_system``.

    The container is frozen, but its pandas/xarray contents remain editable.
    Editing them never updates the source SpatialData object or backing store.
    """

    per_target: pd.DataFrame
    per_class: pd.DataFrame
    spatial_counts: xr.DataArray | None


def summarize_points(
    sdata: SpatialData,
    points_name: str,
    *,
    feature_classes: str | Sequence[str] | None = None,
    bin_size: float | None = None,
    max_grid_bytes: int | None = 1024**3,
    to_coordinate_system: str = "global",
    crd: SpatialBounds | tuple[float, ...] | None = None,
    top_n: int = 20,
) -> PointsSummary:
    """Summarize original points by panel feature/class and optional spatial bins.

    Requires the selected points element to reference a feature panel in
    ``sdata.attrs["harpy"]``. Points elements without this metadata are not
    supported, including when ``feature_classes=None``. Register a complete
    assay panel explicitly with ``hp.pt.add_feature_panel()`` before calling
    this function; no panel is inferred or registered here.

    Include every selected panel feature, even with no detected points. This
    is independent of segmentation: points outside cells are included, and no
    labels, images, or AnnData table are required. The operation is read-only.

    Parameters
    ----------
    sdata
        SpatialData containing the points and their Harpy feature-panel metadata.
        Both in-memory and Zarr-backed objects are supported.
    points_name
        One points element to summarize; independent samples are not pooled.
    feature_classes
        Exact panel class name or non-empty sequence of distinct names. None
        selects all classes, including endogenous features. Results retain
        panel class/feature order, regardless of selection order.
    bin_size
        Positive finite bin width in ``to_coordinate_system`` units. None
        skips binning and returns ``spatial_counts=None``. No units are inferred.
    max_grid_bytes
        Maximum bytes for the final dense uint64 count grid; defaults to 1 GiB.
        A positive integer, or None to disable the limit. When binning is
        requested, raise ValueError if ``n_classes * n_y_bins * n_x_bins * 8``
        exceeds this limit, before allocating bin edges or reducing counts.
        Without ``crd``, the coordinate-only extent calculation happens first.
        This bounds only the final count array, not total peak memory: pandas
        summaries, coordinates, and other intermediate objects require more.
        The limit is not enforced when ``bin_size=None``.
    to_coordinate_system
        Registered coordinate system used for crop and bins. Apply the full
        same-dimensional affine transformation before projecting to XY. The
        registration is only required when cropping or binning is requested.
    crd
        Optional :class:`harpy.SpatialBounds`, for example
        ``hp.SpatialBounds(x=(xmin, xmax), y=(ymin, ymax), z=(zmin, zmax))``.
        The optional z interval restricts 3D points; None leaves z unrestricted.
        Tuples ``(xmin, xmax, ymin, ymax)`` and
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` are also accepted.
        All bounds use ``to_coordinate_system`` and apply after the full
        coordinate transformation, before projecting to XY for binning.
        Supplying z bounds for 2D points raises ValueError.
        Both forms require finite, increasing bounds on each axis. The crop
        is half-open: minima included, maxima excluded. Every output uses
        this crop. Last bins may be narrower than bin_size. Result metadata
        stores the normalized tuple, regardless of the input form.
    top_n
        Positive integer used only for the top-N concentration statistic.
        It does not truncate per-target results or other class statistics.

    Returns
    -------
    PointsSummary
        Computed per-target and per-class dataframes and optional raw-count
        spatial grid. Only compact reductions reach driver memory, not the
        original points. Grid memory scales with extent, bin size, and classes.

    Notes
    -----
    The authoritative panel is resolved through::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
            -> sdata.attrs["harpy"]["feature_panels"][panel_name]

    The panel defines source ``feature_key``, categorical ``feature_class_key``,
    classes, and complete feature lists. Every source point's feature must
    occur in the panel and its observed class must match. These checks run
    partition-wise before class, crop, or z filtering, so selection cannot hide
    invalid feature/class assignments. A panel is required even when selecting
    all classes. Panel features without detections contribute explicit zeros.

    Bins are half-open, including on internal edges. Without ``crd``, a
    preliminary coordinate-only reduction discovers shared bounds before class
    filtering. The grid extends beyond the observed
    maxima so every point is included; a constant coordinate needs one bin.
    An empty points element requires explicit ``crd`` to define a grid.
    Without binning, empty selections still yield panel-defined zero counts.
    The optional z interval filters points in the requested coordinate system;
    it does not select an exact plane in the source coordinates. Spatial counts
    remain an XY grid of the selected points, not a volumetric histogram.

    Target and bin counts are computed together from each selected source
    partition. No raster lookup, aggregation table, smoothing, plotting, or
    persistence is performed.

    Examples
    --------
    >>> summary = hp.qc.summarize_points(
    ...     sdata, "transcripts", feature_classes=["Negative", "SystemControl"],
    ...     bin_size=200, to_coordinate_system="sample_micron",
    ... )
    >>> summary.per_target  # includes panel features with zero detections
    >>> summary.spatial_counts  # raw (feature_class, y, x) bin counts
    """
    if points_name not in sdata.points:
        raise ValueError(f"Points element {points_name!r} does not exist.")
    panel_name, panel, points_record = _resolve_points_feature_panel(sdata, points_name)
    points = sdata.points[points_name]
    for key in (panel.feature_key, panel.feature_class_key):
        if key not in points.columns:
            raise ValueError(f"Points element {points_name!r} does not contain panel column {key!r}.")
    _validate_feature_class_dtype(points, points_name=points_name, panel=panel)
    classes = _selected_feature_classes(feature_classes, panel.classes)
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if max_grid_bytes is not None:
        if isinstance(max_grid_bytes, bool) or not isinstance(max_grid_bytes, Integral) or max_grid_bytes < 1:
            raise ValueError("max_grid_bytes must be a positive integer or None.")
        max_grid_bytes = int(max_grid_bytes)
    if bin_size is not None and (not np.isfinite(bin_size) or bin_size <= 0):
        raise ValueError("bin_size must be positive and finite.")
    crd = _normalize_spatial_bounds(crd)

    axes = ()
    matrix = None
    if bin_size is not None or crd is not None:
        axes = tuple(get_axes_names(points))
        if axes not in (("x", "y"), ("x", "y", "z")):
            raise ValueError(f"Spatial summaries require XY or XYZ points, found axes {axes!r}.")
        if crd is not None and crd.z is not None and "z" not in axes:
            raise ValueError("crd z bounds require 3D points; the selected points element has no z axis.")
        try:
            transformation = get_transformation(points, to_coordinate_system=to_coordinate_system)
        except ValueError as e:
            raise ValueError(
                f"Points element {points_name!r} does not define coordinate system {to_coordinate_system!r}."
            ) from e
        matrix = _invertible_affine_matrix(transformation, axes=axes, element_kind="Points")

    edges = None
    if bin_size is not None:
        bounds = None if crd is None else (*crd.x, *crd.y)
        if bounds is None:
            # Reduce coordinates in the requested coordinate system, not the
            # source frame. Both reductions share the lazy transformation.
            transformed_xy = points[list(axes)].map_partitions(
                _transformed_point_xy,
                axes=axes,
                matrix=matrix,
                meta={"x": "float64", "y": "float64"},
            )
            minimum, maximum = dask.compute(transformed_xy.min(), transformed_xy.max())
            bounds = (minimum["x"], maximum["x"], minimum["y"], maximum["y"])
            if not np.isfinite(bounds).all():
                raise ValueError("Cannot infer spatial extent from empty points; supply crd or use bin_size=None.")
        edges = _point_bin_edges(
            bounds,
            bin_size,
            explicit_extent=crd is not None,
            class_count=len(classes),
            max_grid_bytes=max_grid_bytes,
        )

    columns = list(dict.fromkeys([panel.feature_key, panel.feature_class_key, *axes]))
    # Each delayed partition has one consumer that both validates and reduces
    # it. Target and bin counts therefore share the same source read/selection.
    tasks = [
        dask.delayed(_summarize_point_partition)(
            part,
            panel=panel,
            points_name=points_name,
            classes=classes,
            axes=axes,
            matrix=matrix,
            crd=crd,
            edges=edges,
        )
        for part in points[columns].to_delayed()
    ]
    target_counts, bin_counts = dask.compute(_tree_reduce(tasks, _merge_point_summaries))[0]
    per_target, per_class = _summary_frames(target_counts, panel=panel, classes=classes, top_n=int(top_n))
    identity = {"points_name": points_name, "feature_panel": panel_name}
    if points_record.get("sample_id") is not None:
        identity["sample_id"] = points_record["sample_id"]
    metadata = {
        **identity,
        "to_coordinate_system": to_coordinate_system,
        "crd": None if crd is None else crd.as_tuple(),
        "panel_feature_counts": {name: len(panel.features_by_class[name]) for name in classes},
    }
    for frame in (per_target, per_class):
        for name, value in identity.items():
            frame[name] = value
        frame.attrs = deepcopy(metadata)
    grid = (
        None
        if edges is None
        else _spatial_count_array(bin_counts, classes=classes, edges=edges, bin_size=bin_size, metadata=metadata)
    )
    return PointsSummary(per_target=per_target, per_class=per_class, spatial_counts=grid)


def _selected_feature_classes(selection: str | Sequence[str] | None, classes: tuple[str, ...]) -> tuple[str, ...]:
    if selection is None:
        return classes
    requested = (selection,) if isinstance(selection, str) else tuple(selection)
    if not requested or any(not isinstance(value, str) for value in requested):
        raise ValueError("feature_classes must contain one or more exact class names.")
    if len(set(requested)) != len(requested):
        raise ValueError("feature_classes must not contain duplicate class names.")
    unknown = set(requested) - set(classes)
    if unknown:
        raise ValueError(f"Unknown feature classes: {sorted(unknown)}; panel classes are {list(classes)}.")
    return tuple(name for name in classes if name in requested)


def _tree_reduce(tasks, merge, *, fan_in: int = 8):
    """Build a lazy merge tree over compact partition results.

    Each merge receives at most ``fan_in`` partial summaries. Repeat until one
    delayed result remains and return that task, without computing it here.
    This avoids collecting every partition's partial result on the driver.

    For example, 16 partition summaries with ``fan_in=8`` form this tree::

        Summaries from partitions 0-7     Summaries from partitions 8-15
                       |                                |
                       v                                v
               Merge -> summary A               Merge -> summary B
                       |                                |
                       +----------------+---------------+
                                        |
                                        v
                               Merge -> final summary

    In that example, ``level`` is replaced after each round::

        # Initially: 16 partition-summary tasks
        level = [partition_0, ..., partition_15]

        # After the first round: 2 merge tasks
        level = [merge_A, merge_B]

        # After the second round: 1 merge task
        level = [final_merge_task]

    The loop stops when only one task remains. ``return level[0]`` extracts
    that final task from the list; it does not select the first partition's
    summary or the target counts from the eventual result.

    The arrows represent task dependencies, not computation performed by
    this function. Dask executes the graph when the returned task is computed.
    For point summaries, every partition and merged result has the same
    ``(target_counts, bin_counts)`` structure, so the same merge function works
    at each level. The final task therefore depends on all input partitions.
    """
    level = list(tasks)
    while len(level) > 1:
        next_level = []
        for start in range(0, len(level), fan_in):
            group = level[start : start + fan_in]
            merged_task = dask.delayed(merge)(*group)
            next_level.append(merged_task)
        level = next_level
    return level[0]


def _summarize_point_partition(
    partition: pd.DataFrame,
    *,
    panel: _FeaturePanelContract,
    points_name: str,
    classes: tuple[str, ...],
    axes: tuple[str, ...],
    matrix: np.ndarray | None,
    crd: SpatialBounds | None,
    edges: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[pd.Series, pd.Series | None]:
    """Validate all source assignments, then jointly reduce selected targets and bins.

    For example, three selected rows for feature A become ``A: 3``. If those
    rows occupy two bins, the same call also returns the two occupied-bin
    counts. The feature-to-class mapping is fixed by the validated panel, so
    grouping targets by feature alone cannot mix classes. Non-selected classes
    and out-of-crop points still undergo source feature/class validation.
    """
    errors = _feature_panel_partition_errors(
        partition,
        feature_key=panel.feature_key,
        feature_class_key=panel.feature_class_key,
        class_by_feature=panel.class_by_feature,
    )
    if len(errors):
        raise ValueError(f"Points element {points_name!r}: {errors.iloc[0]}")
    keep, xy = _select_point_coordinates(partition, axes=axes, matrix=matrix, crd=crd)
    selected = partition.loc[keep]
    selected_classes = selected[panel.feature_class_key].isin(classes).to_numpy()
    selected = selected.loc[selected_classes]
    # Count observed strings only: categorical value_counts would otherwise
    # include unused source categories independently of the authoritative panel.
    targets = selected[panel.feature_key].astype(object).value_counts(sort=False).astype(np.uint64)
    targets.index.name = "feature"
    targets.name = "n_points"
    bins = None
    if edges is not None:
        bins = _count_point_bins(
            xy[selected_classes],
            selected[panel.feature_class_key].to_numpy(dtype=object),
            edges=edges,
        )
    return targets, bins


def _merge_point_summaries(*summaries) -> tuple[pd.Series, pd.Series | None]:
    """Sum repeated target/bin keys across compact summaries, retaining uint64 counts."""
    targets = pd.concat([summary[0] for summary in summaries]).groupby(level=0, sort=False).sum()
    bins = None
    if summaries[0][1] is not None:
        bins = pd.concat([summary[1] for summary in summaries]).groupby(level=[0, 1, 2], sort=False).sum()
    return targets, bins


def _summary_frames(
    counts: pd.Series, *, panel: _FeaturePanelContract, classes: tuple[str, ...], top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-fill against the selected panel, then derive statistics without rescanning points."""
    target_frames, class_rows = [], []
    for name in classes:
        features = panel.features_by_class[name]
        values = counts.reindex(features, fill_value=0).to_numpy(dtype=np.uint64)
        total = values.sum(dtype=np.uint64)
        target_frames.append(
            pd.DataFrame(
                {
                    "feature": features,
                    "feature_class": name,
                    "n_points": values,
                    "within_class_fraction": values / total if total else np.full(len(values), np.nan),
                }
            )
        )
        n_top = min(top_n, len(values))
        n_zero = int((values == 0).sum())
        class_rows.append(
            {
                "feature_class": name,
                "n_features": len(features),
                "n_zero_features": n_zero,
                "pct_zero_features": 100 * n_zero / len(features),
                "n_points": total,
                "mean_points_per_feature": float(values.mean()),
                "median_points_per_feature": float(np.median(values)),
                "p95_points_per_feature": float(np.percentile(values, 95)),
                "top_n": top_n,
                "n_top_features": n_top,
                "pct_points_top_n_features": 100 * (np.sort(values)[-n_top:].sum() / total) if total else np.nan,
            }
        )
    return pd.concat(target_frames, ignore_index=True), pd.DataFrame(class_rows)


def _spatial_count_array(counts: pd.Series, *, classes, edges, bin_size, metadata) -> xr.DataArray:
    """Expand merged occupied-bin counts once into a zero-filled, coordinate-aware grid."""
    x_edges, y_edges = edges
    grid = np.zeros((len(classes), len(y_edges) - 1, len(x_edges) - 1), dtype=np.uint64)
    if len(counts):
        class_codes = pd.Index(classes).get_indexer(counts.index.get_level_values("feature_class"))
        grid[class_codes, counts.index.get_level_values("y_bin"), counts.index.get_level_values("x_bin")] = (
            counts.to_numpy()
        )
    return xr.DataArray(
        grid,
        dims=("feature_class", "y", "x"),
        name="n_points",
        coords={
            "feature_class": list(classes),
            "x": x_edges[:-1] + np.diff(x_edges) / 2,
            "y": y_edges[:-1] + np.diff(y_edges) / 2,
        },
        attrs={
            **deepcopy(metadata),
            "bin_size": bin_size,
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "extent": (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])),
        },
    )
