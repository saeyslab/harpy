"""Shared validation and reductions for class- and feature-level point summaries."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.dataframe import DataFrame as DaskDataFrame
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
from harpy.qc.points._points_binning import (
    _count_point_bins,
    _point_bin_edges,
    _select_point_coordinates,
    _transformed_point_xy,
)
from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._points_summary_schema import _FEATURE_CLASS_KEY, _FEATURE_KEY, _N_POINTS_KEY
from harpy.transformations._transformations import _invertible_affine_matrix


@dataclass(frozen=True)
class _PointSummaryReduction:
    """Compact source reductions used to assemble a class-level points summary.

    ``selected_names`` contains the selected class names in panel order,
    including names without detections. ``feature_counts`` contains totals
    for observed features in those classes; the summary zero-fills against the panel.
    No original points are retained here.
    """

    panel: _FeaturePanelContract
    selected_names: tuple[str, ...]
    feature_counts: pd.Series
    spatial_counts: xr.DataArray | None
    metadata: PointsSummaryMetadata


def _reduce_points_by_class(
    sdata: SpatialData,
    points_name: str,
    *,
    feature_classes: str | Sequence[str] | None,
    bin_size: float | None,
    max_grid_bytes: int | None,
    to_coordinate_system: str,
    microns_per_unit: float | None,
    crd: SpatialBounds | tuple[float, ...] | None,
) -> _PointSummaryReduction:
    """Prepare a new class-level analysis context and reduce its source points.

    Validate class selection and spatial parameters, construct optional bin
    edges (discovering selected-class extent when needed), and build metadata.
    The selected population must be nonempty. Actual partition validation and
    counting use ``_compute_point_reductions()``, also used by feature summaries.
    """
    if points_name not in sdata.points:
        raise ValueError(f"Points element {points_name!r} does not exist.")
    panel_name, panel, points_record = _resolve_points_feature_panel(sdata, points_name)
    points = sdata.points[points_name]
    _validate_point_columns(points, points_name=points_name, panel=panel)
    if feature_classes is None:
        feature_classes = panel.classes
    if not isinstance(feature_classes, (str, Sequence)):
        raise ValueError("feature_classes must contain one or more exact panel names.")
    requested = (feature_classes,) if isinstance(feature_classes, str) else tuple(feature_classes)
    if not requested or any(not isinstance(value, str) for value in requested):
        raise ValueError("feature_classes must contain one or more exact panel names.")
    if len(set(requested)) != len(requested):
        raise ValueError("feature_classes must not contain duplicate names.")
    unknown = set(requested) - set(panel.classes)
    if unknown:
        raise ValueError(f"Unknown feature classes: {sorted(unknown)}; requested names must belong to the panel.")
    # _resolve_points_feature_panel() returns sorted panel classes. Preserve
    # this canonical order rather than the order of the user's request.
    selected_names = tuple(name for name in panel.classes if name in requested)

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
        axes, matrix = _resolve_point_coordinate_transform(
            points, points_name=points_name, to_coordinate_system=to_coordinate_system, crd=crd
        )

    edges = None
    if bin_size is not None:
        bounds = None if crd is None else (*crd.x, *crd.y)
        if bounds is None:
            # Only selected classes determine extent. Min/max share the lazy
            # transformation into the requested coordinate system.
            selected_points = points[points[panel.feature_class_key].isin(selected_names)]
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
                    "No points remain for the selected feature_classes; cannot infer spatial extent. "
                    "Check the selection and source points."
                )
        edges = _point_bin_edges(
            bounds,
            bin_size,
            explicit_extent=crd is not None,
            group_count=len(selected_names),
            max_grid_bytes=max_grid_bytes,
        )

    feature_counts, bin_counts = _compute_point_reductions(
        points,
        panel=panel,
        points_name=points_name,
        selected_names=selected_names,
        summary_axis=_FEATURE_CLASS_KEY,
        axes=axes,
        matrix=matrix,
        crd=crd,
        edges=edges,
    )
    # Counts contain only observed selected features, before panel zero-filling.
    if feature_counts.empty:
        raise ValueError(
            "No points remain after applying feature_classes and crd. "
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
        else _spatial_count_array(
            bin_counts, selected_names=selected_names, summary_axis=_FEATURE_CLASS_KEY, edges=edges
        )
    )
    return _PointSummaryReduction(
        panel=panel,
        selected_names=selected_names,
        feature_counts=feature_counts,
        spatial_counts=grid,
        metadata=metadata,
    )


def _validate_point_columns(points: DaskDataFrame, *, points_name: str, panel: _FeaturePanelContract) -> None:
    """Check source columns and categorical class dtype without reading point rows."""
    for key in (panel.feature_key, panel.feature_class_key):
        if key not in points.columns:
            raise ValueError(f"Points element {points_name!r} does not contain panel column {key!r}.")
    _validate_feature_class_dtype(points, points_name=points_name, panel=panel)


def _resolve_point_coordinate_transform(
    points: DaskDataFrame, *, points_name: str, to_coordinate_system: str, crd: SpatialBounds | None
) -> tuple[tuple[str, ...], np.ndarray]:
    """Resolve the full XY/XYZ affine for spatial selection and binning."""
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
    return axes, _invertible_affine_matrix(transformation, axes=axes, element_kind="Points")


def _compute_point_reductions(
    points: DaskDataFrame,
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
    """Compute compact feature totals and optional bin counts from prepared inputs.

    Parameters
    ----------
    points
        Lazy source points dataframe. The caller checks that the panel's
        feature/class columns exist and the class column has a compatible dtype.
    panel
        Validated feature panel defining source column names and the allowed
        feature-to-class assignments.
    points_name
        Source element name used in validation errors.
    selected_names
        Validated class or feature names to include, according to ``summary_axis``.
    summary_axis
        ``"feature_class"`` selects classes through ``panel.feature_class_key``;
        ``"feature"`` selects features through ``panel.feature_key``. Also names
        the first bin-count index level. Feature totals always count individual
        features, regardless of this choice.
    axes
        Source coordinate columns in matrix order: ``("x", "y")`` or
        ``("x", "y", "z")``. Empty when neither cropping nor binning is requested.
    matrix
        Homogeneous affine matrix from source coordinates to the requested
        coordinate system, with input and output axes ordered as ``axes``.
        None when no spatial operation is requested.
    crd
        Optional crop in transformed coordinates, applied to both outputs.
        Any z bounds are applied before projecting to XY for binning.
    edges
        Prepared ``(x_edges, y_edges)`` arrays in transformed coordinates, or
        None to skip binning. The caller supplies edges covering the selected
        points; this helper does not infer extent or construct a grid.

    Returns
    -------
    feature_counts
        In-memory uint64 Series named ``n_points``, indexed by ``feature``.
        Contains totals for observed features after selection and cropping;
        callers add zero counts for undetected panel features.
    bin_counts
        In-memory uint64 Series named ``n_points``, indexed by
        ``(summary_axis, y_bin, x_bin)`` with zero-based bin indices. Contains
        only occupied bins; None when ``edges`` is None.

    Notes
    -----
    Each partition validates all source feature/class assignments before
    filtering, then computes both counts. A tree reduction merges compact
    results; only the final reduced Series are materialized on the driver,
    never original point rows. Empty results are returned unchanged, leaving
    the empty-population policy to the caller.
    """
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
    return dask.compute(_tree_reduce(tasks, _merge_point_summaries))[0]


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
    Excluded features/classes contribute no counts to this reduction. Callers
    decide which bins enter the resulting statistical summaries.
    """
    errors = _feature_panel_partition_errors(
        partition,
        feature_key=panel.feature_key,
        feature_class_key=panel.feature_class_key,
        class_by_feature=panel.class_by_feature,
    )
    if len(errors):
        raise ValueError(f"Points element {points_name!r}: {errors.iloc[0]}")
    selection_key = panel.feature_class_key if summary_axis == _FEATURE_CLASS_KEY else panel.feature_key
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
    features.index.name = _FEATURE_KEY
    features.name = _N_POINTS_KEY
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
        name=_N_POINTS_KEY,
        coords={
            summary_axis: list(selected_names),
            "x": x_edges[:-1] + np.diff(x_edges) / 2,
            "y": y_edges[:-1] + np.diff(y_edges) / 2,
        },
    )
