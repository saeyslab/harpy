"""Tabular measurements derived from a computed spatial-count grid."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from harpy.qc._points_summary_metadata import PointsSummaryMetadata


@dataclass(frozen=True)
class SpatialBinSummary:
    """Counts and statistics over bins containing points from any selected class.

    Attributes
    ----------
    per_bin
        One row per retained bin and selected ``feature_class`` (categorical),
        in class order then row-major ``(y_bin, x_bin)`` order. ``x`` and ``y``
        are bin centers in the selected coordinate system; ``bin_area`` is
        the actual geometric area in squared coordinate-system units.
        ``n_points`` contains uint64 class counts, including zeros when the bin
        contains points from other selected classes. With physical calibration,
        ``bin_area_um2`` is also present. Counts are not normalized by area;
        cropped terminal bins may have smaller areas than full bins.
    per_class
        One row per selected ``feature_class``, including undetected classes.
        Every row exposes the shared population: ``n_total_bins`` counts all
        grid bins, ``n_retained_bins`` counts bins with points from any selected
        class, and ``n_excluded_bins`` counts bins empty across selected classes.
        ``pct_excluded_bins`` is 100 times excluded / total bins.

        ``n_retained_bins_without_class`` counts retained bins with zero points
        of this row's class; ``pct_retained_bins_without_class`` is 100 times
        that count / retained bins. These bins contain other selected classes' points.
        ``n_points``, ``mean_points_per_bin``, ``median_points_per_bin``,
        ``std_points_per_bin``, and ``p95_points_per_bin`` also use the shared
        retained-bin population. SD uses ``ddof=1``; NaN for fewer than two bins.
        An all-zero class in a nonempty population has zero-valued count
        statistics and ``pct_retained_bins_without_class=100``. Percentiles
        use linear interpolation.
    """

    per_bin: pd.DataFrame
    per_class: pd.DataFrame


@dataclass(frozen=True)
class FeatureSpatialBinSummary:
    """Per-feature counts and statistics over bins with any requested-feature points.

    Attributes
    ----------
    per_bin
        One row per retained bin and selected ``feature`` (categorical), in
        feature order then row-major ``(y_bin, x_bin)`` order. ``n_points`` is
        that feature's uint64 point count, not the number of distinct features.
        Individual zeros are retained. ``x``/``y`` are bin centers;
        ``bin_area`` is the actual area in squared coordinate-system units.
        ``bin_area_um2`` is present with physical calibration. Counts are raw;
        cropped terminal bins may be smaller. Geometry context lives in the
        parent result's metadata, not in dataframe attributes.
    per_feature
        One row per requested feature, including undetected features, with
        its panel-defined ``feature_class``. ``n_total_bins``, ``n_retained_bins``,
        ``n_excluded_bins`` and ``pct_excluded_bins`` describe the shared bin
        population. Excluded bins have no points from any requested feature.
        ``n_retained_bins_without_feature`` and ``pct_retained_bins_without_feature``
        count this feature's zeros within that population (percentage 0–100).
        ``n_points``, ``mean_points_per_bin``, ``median_points_per_bin``,
        ``std_points_per_bin`` and ``p95_points_per_bin`` include those zeros.
        Percentiles use linear interpolation; sample SD uses ``ddof=1`` and
        is NaN for fewer than two bins.
    """

    per_bin: pd.DataFrame
    per_feature: pd.DataFrame


def _summarize_feature_bins(
    grid: xr.DataArray, *, metadata: PointsSummaryMetadata, class_by_feature: Mapping[str, str]
) -> FeatureSpatialBinSummary:
    """Summarize the feature grid and attach each feature's authoritative panel class once."""
    per_bin, per_feature = _spatial_bin_frames(grid, metadata=metadata, group_axis="feature")
    feature_classes = per_feature["feature"].astype(object).map(class_by_feature)
    per_feature.insert(1, "feature_class", pd.Categorical(feature_classes))
    return FeatureSpatialBinSummary(per_bin=per_bin, per_feature=per_feature)


def _summarize_spatial_bins(grid: xr.DataArray, *, metadata: PointsSummaryMetadata) -> SpatialBinSummary:
    """Prepare retained-bin measurements once, then summarize those same rows.

    ``grid`` contains only the selected classes. Derive bin inclusion directly
    from its counts: a bin is retained if any selected class has a point.
    For example, a bin with Endogenous=3 and Negative=0 yields a zero Negative
    row when both classes were selected. If only Negative was selected, that
    bin is excluded. A bin empty across selected classes is excluded.
    Geometry and calibration come from the parent summary's ``metadata``,
    passed explicitly rather than copied onto the grid or result dataframes.
    No source points or panel registry are accessed here, and neither input
    is modified.

    For an independently supplied grid with no retained bins, ``per_bin`` is
    empty, class point counts and ``n_retained_bins_without_class`` are zero,
    and distribution statistics and ``pct_retained_bins_without_class`` are NaN.
    """
    per_bin, per_class = _spatial_bin_frames(grid, metadata=metadata, group_axis="feature_class")
    return SpatialBinSummary(per_bin=per_bin, per_class=per_class)


def _spatial_bin_frames(
    grid: xr.DataArray, *, metadata: PointsSummaryMetadata, group_axis: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce a completed class/feature grid into bin rows and per-group statistics.

    Every selected group uses the same bins: any nonzero plane retains a bin,
    including zeros in the other planes. This calculation uses no source
    points. Rows follow group order, then row-major spatial order.
    """
    groups = tuple(grid.coords[group_axis].values)
    zero_suffix = "class" if group_axis == "feature_class" else "feature"
    retained = grid.values.any(axis=0)
    y_bin, x_bin = np.nonzero(retained)
    n_retained = len(y_bin)
    widths = np.diff(metadata.x_edges)[x_bin]
    heights = np.diff(metadata.y_edges)[y_bin]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        areas = widths * heights
    if not (np.isfinite(areas) & (areas > 0)).all():
        raise ValueError("Spatial bin areas must be positive and finite; rescale the coordinate system or bin size.")

    # Only retained-bin rows are expanded. The categorical codes avoid
    # repeating feature/class strings, and geometry is derived from actual edges.
    point_counts = np.empty(len(groups) * n_retained, dtype=np.uint64)
    for ordinal, plane in enumerate(grid.values):
        point_counts[ordinal * n_retained : (ordinal + 1) * n_retained] = plane[retained]
    per_bin = pd.DataFrame(
        {
            group_axis: pd.Categorical.from_codes(np.repeat(np.arange(len(groups)), n_retained), categories=groups),
            "y_bin": np.tile(y_bin, len(groups)),
            "x_bin": np.tile(x_bin, len(groups)),
            "x": np.tile(grid.x.values[x_bin], len(groups)),
            "y": np.tile(grid.y.values[y_bin], len(groups)),
            "bin_area": np.tile(areas, len(groups)),
            "n_points": point_counts,
        },
        copy=False,
    )
    microns_per_unit = metadata.microns_per_unit
    if microns_per_unit is not None:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            areas_um2 = (widths * microns_per_unit) * (heights * microns_per_unit)
        if not (np.isfinite(areas_um2) & (areas_um2 > 0)).all():
            raise ValueError("Calibrated spatial bin areas must be positive and finite; check microns_per_unit.")
        per_bin["bin_area_um2"] = np.tile(areas_um2, len(groups))

    n_total = retained.size
    population = {
        "n_total_bins": n_total,
        "n_retained_bins": n_retained,
        "n_excluded_bins": n_total - n_retained,
        "pct_excluded_bins": 100 * (n_total - n_retained) / n_total,
    }
    group_rows = []
    for ordinal, group in enumerate(groups):
        # Each group occupies n_retained consecutive rows, so select its block
        # directly instead of grouping or filtering the entire per-bin frame.
        group_bins = per_bin.iloc[ordinal * n_retained : (ordinal + 1) * n_retained]
        counts = group_bins["n_points"].to_numpy(copy=False)
        total = counts.sum(dtype=np.uint64)
        n_zero = int(np.count_nonzero(counts == 0))
        row = {
            group_axis: group,
            **population,
            "n_points": total,
            f"n_retained_bins_without_{zero_suffix}": n_zero,
            f"pct_retained_bins_without_{zero_suffix}": 100 * n_zero / n_retained if n_retained else np.nan,
            "mean_points_per_bin": float(counts.mean()) if n_retained else np.nan,
            "median_points_per_bin": float(np.median(counts)) if n_retained else np.nan,
            "std_points_per_bin": float(counts.std(ddof=1)) if n_retained > 1 else np.nan,
            "p95_points_per_bin": float(np.percentile(counts, 95)) if n_retained else np.nan,
        }
        group_rows.append(row)
    per_group = pd.DataFrame(group_rows)
    per_group[group_axis] = pd.Categorical(per_group[group_axis], categories=groups)

    return per_bin, per_group
