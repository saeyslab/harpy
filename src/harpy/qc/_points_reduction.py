"""Shared validation and reductions for class- and feature-level point summaries."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

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
from harpy.qc._points_summary_metadata import PointsSummaryMetadata
from harpy.transformations._transformations import _invertible_affine_matrix


@dataclass(frozen=True)
class _PointSummaryReduction:
    """Compact source reductions shared by class- and feature-level summaries.

    ``selected_names`` contains the selected feature/class names in panel order,
    including names without detections. ``feature_counts`` contains only
    observed selected features' totals; class summaries zero-fill these against the panel.
    No original points are retained here.
    """

    panel: _FeaturePanelContract
    selected_names: tuple[str, ...]
    feature_counts: pd.Series
    spatial_counts: xr.DataArray | None
    metadata: PointsSummaryMetadata


def _reduce_points(
    sdata: SpatialData,
    points_name: str,
    *,
    selected_names: str | Sequence[str] | None,
    summary_axis: Literal["feature_class", "feature"],
    bin_size: float | None,
    max_grid_bytes: int | None,
    to_coordinate_system: str,
    microns_per_unit: float | None,
    crd: SpatialBounds | tuple[float, ...] | None,
) -> _PointSummaryReduction:
    """Validate and jointly count selected features, grouped spatially by feature or class.

    Normalize and validate ``selected_names`` as a tuple in panel order before
    counting, without modifying the caller's input sequence.
    Resolve source columns from the panel. All selected features/classes share
    one grid and one partition-wise validation/count pass; automatic extent
    discovery needs a preliminary coordinate reduction. Selection determines extent and counts,
    but never hides invalid source feature/class assignments during counting.
    ``summary_axis`` names the output feature/class dimension, not a source column.
    """
    if points_name not in sdata.points:
        raise ValueError(f"Points element {points_name!r} does not exist.")
    panel_name, panel, points_record = _resolve_points_feature_panel(sdata, points_name)
    points = sdata.points[points_name]
    for key in (panel.feature_key, panel.feature_class_key):
        if key not in points.columns:
            raise ValueError(f"Points element {points_name!r} does not contain panel column {key!r}.")
    _validate_feature_class_dtype(points, points_name=points_name, panel=panel)
    if summary_axis == "feature_class":
        selection_parameter = "feature_classes"
        allowed_names = panel.classes
        selection_key = panel.feature_class_key
        if selected_names is None:
            selected_names = allowed_names
    else:
        selection_parameter = "features"
        # The mapping's keys are the panel's feature names.
        allowed_names = tuple(panel.class_by_feature)
        selection_key = panel.feature_key
        if selected_names is None:
            raise ValueError("features must contain one or more exact panel names.")

    # The panel returned by _resolve_points_feature_panel() guarantees sorted
    # classes and sorted features within each class. `allowed_names` inherits
    # that order: class names, or feature names ordered first by class, then
    # within each class—not globally alphabetical across classes.
    # Validate the selection below and preserve this panel order, not request order.
    if not isinstance(selected_names, (str, Sequence)):
        raise ValueError(f"{selection_parameter} must contain one or more exact panel names.")
    selected_names = (selected_names,) if isinstance(selected_names, str) else tuple(selected_names)
    if not selected_names or any(not isinstance(value, str) for value in selected_names):
        raise ValueError(f"{selection_parameter} must contain one or more exact panel names.")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError(f"{selection_parameter} must not contain duplicate names.")
    unknown = set(selected_names) - set(allowed_names)
    if unknown:
        label = "feature classes" if summary_axis == "feature_class" else "features"
        raise ValueError(f"Unknown {label}: {sorted(unknown)}; requested names must belong to the panel.")
    selected_names = tuple(name for name in allowed_names if name in selected_names)

    if max_grid_bytes is not None:
        if isinstance(max_grid_bytes, bool) or not isinstance(max_grid_bytes, Integral) or max_grid_bytes < 1:
            raise ValueError("max_grid_bytes must be a positive integer or None.")
        max_grid_bytes = int(max_grid_bytes)
    if bin_size is not None and (
        isinstance(bin_size, (bool, np.bool_))
        or not isinstance(bin_size, Real)
        or not np.isfinite(bin_size)
        or bin_size <= 0
    ):
        raise ValueError("bin_size must be positive and finite.")
    if microns_per_unit is not None:
        if isinstance(microns_per_unit, (bool, np.bool_)) or not isinstance(microns_per_unit, Real):
            raise ValueError("microns_per_unit must be a positive finite number or None.")
        if not np.isfinite(microns_per_unit) or microns_per_unit <= 0:
            raise ValueError("microns_per_unit must be a positive finite number or None.")
        microns_per_unit = float(microns_per_unit)
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
            # Only selected features/classes determine extent. Min/max share the lazy
            # transformation into the requested coordinate system.
            selected_points = points[points[selection_key].isin(selected_names)]
            transformed_xy = selected_points[list(axes)].map_partitions(
                _transformed_point_xy,
                axes=axes,
                matrix=matrix,
                meta={"x": "float64", "y": "float64"},
            )
            minimum, maximum = dask.compute(transformed_xy.min(), transformed_xy.max())
            bounds = (minimum["x"], maximum["x"], minimum["y"], maximum["y"])
            if not np.isfinite(bounds).all():
                raise ValueError(
                    f"No points remain for the selected {selection_parameter}; cannot infer spatial extent. "
                    "Check the selection and source points."
                )
        edges = _point_bin_edges(
            bounds,
            bin_size,
            explicit_extent=crd is not None,
            group_count=len(selected_names),
            max_grid_bytes=max_grid_bytes,
        )

    columns = list(dict.fromkeys([panel.feature_key, panel.feature_class_key, *axes]))
    # Each delayed partition has one consumer that both validates and reduces
    # it. Feature and bin counts therefore share the same source read/selection.
    tasks = [
        dask.delayed(_summarize_point_partition)(
            part,
            panel=panel,
            points_name=points_name,
            selected_names=selected_names,
            summary_axis=summary_axis,
            axes=axes,
            matrix=matrix,
            crd=crd,
            edges=edges,
        )
        for part in points[columns].to_delayed()
    ]
    feature_counts, bin_counts = dask.compute(_tree_reduce(tasks, _merge_point_summaries))[0]
    # Counts contain only observed selected features, before panel zero-filling.
    if feature_counts.empty:
        raise ValueError(
            f"No points remain after applying {selection_parameter} and crd. "
            "Check the selection, crop bounds, and coordinate system."
        )
    metadata = PointsSummaryMetadata(
        points_name=points_name,
        sample_id=points_record.get("sample_id"),
        feature_panel=panel_name,
        to_coordinate_system=to_coordinate_system,
        crd=crd,
        microns_per_unit=microns_per_unit,
        bin_size=None if bin_size is None else float(bin_size),
        x_edges=None if edges is None else tuple(float(value) for value in edges[0]),
        y_edges=None if edges is None else tuple(float(value) for value in edges[1]),
    )
    grid = (
        None
        if edges is None
        else _spatial_count_array(bin_counts, selected_names=selected_names, summary_axis=summary_axis, edges=edges)
    )
    return _PointSummaryReduction(
        panel=panel,
        selected_names=selected_names,
        feature_counts=feature_counts,
        spatial_counts=grid,
        metadata=metadata,
    )


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
    summary or the feature counts from the eventual result.

    The arrows represent task dependencies, not computation performed by
    this function. Dask executes the graph when the returned task is computed.
    For point summaries, every partition and merged result has the same
    ``(feature_counts, bin_counts)`` structure, so the same
    merge function works at each level. The final task therefore depends on
    all input partitions.
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
    selected_names: tuple[str, ...],
    summary_axis: Literal["feature_class", "feature"],
    axes: tuple[str, ...],
    matrix: np.ndarray | None,
    crd: SpatialBounds | None,
    edges: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[pd.Series, pd.Series | None]:
    """Validate all source assignments, then jointly reduce selected features and bins.

    For example, three selected rows for feature A become ``A: 3``. If those
    rows occupy two bins, the same call also returns the two occupied-bin
    counts. The feature-to-class mapping is fixed by the validated panel, so
    grouping totals by feature alone cannot mix classes. Non-selected features
    or classes and out-of-crop points still undergo source validation.

    After validation, derive the source column from ``panel`` and ``summary_axis``
    and filter it by ``selected_names`` before spatial selection. For example,
    ``gene`` can group bins under the output axis ``feature``; ``code_class``
    can group under ``feature_class``.
    Excluded features/classes contribute neither counts nor bin inclusion.
    """
    errors = _feature_panel_partition_errors(
        partition,
        feature_key=panel.feature_key,
        feature_class_key=panel.feature_class_key,
        class_by_feature=panel.class_by_feature,
    )
    if len(errors):
        raise ValueError(f"Points element {points_name!r}: {errors.iloc[0]}")
    selection_key = panel.feature_class_key if summary_axis == "feature_class" else panel.feature_key
    selected = partition.loc[partition[selection_key].isin(selected_names)]
    keep, xy = _select_point_coordinates(selected, axes=axes, matrix=matrix, crd=crd)
    selected = selected.loc[keep]
    bins = None
    if edges is not None:
        bins = _count_point_bins(
            xy, selected[selection_key].to_numpy(dtype=object), edges=edges, summary_axis=summary_axis
        )
    # Count observed strings only: categorical value_counts would otherwise
    # include unused source categories independently of the authoritative panel.
    features = selected[panel.feature_key].astype(object).value_counts(sort=False).astype(np.uint64)
    features.index.name = "feature"
    features.name = "n_points"
    return features, bins


def _merge_point_summaries(*summaries) -> tuple[pd.Series, pd.Series | None]:
    """Sum repeated feature/bin keys across compact summaries, retaining uint64 counts."""
    features = pd.concat([summary[0] for summary in summaries]).groupby(level=0, sort=False).sum()
    bins = None
    if summaries[0][1] is not None:
        bins = pd.concat([summary[1] for summary in summaries]).groupby(level=[0, 1, 2], sort=False).sum()
    return features, bins


def _spatial_count_array(counts: pd.Series, *, selected_names, summary_axis: str, edges) -> xr.DataArray:
    """Expand merged occupied-bin counts once into a zero-filled, coordinate-aware grid."""
    x_edges, y_edges = edges
    grid = np.zeros((len(selected_names), len(y_edges) - 1, len(x_edges) - 1), dtype=np.uint64)
    if len(counts):
        group_codes = pd.Index(selected_names).get_indexer(counts.index.get_level_values(summary_axis))
        grid[group_codes, counts.index.get_level_values("y_bin"), counts.index.get_level_values("x_bin")] = (
            counts.to_numpy()
        )
    return xr.DataArray(
        grid,
        dims=(summary_axis, "y", "x"),
        name="n_points",
        coords={
            summary_axis: list(selected_names),
            "x": x_edges[:-1] + np.diff(x_edges) / 2,
            "y": y_edges[:-1] + np.diff(y_edges) / 2,
        },
    )
