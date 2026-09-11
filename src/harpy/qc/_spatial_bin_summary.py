"""Tabular measurements derived from a computed spatial-count grid."""

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
        ``n_points``, ``mean_points_per_bin``, ``median_points_per_bin``, and
        ``p95_points_per_bin`` also use the shared retained-bin population.
        An all-zero class in a nonempty population has zero-valued count
        statistics and ``pct_retained_bins_without_class=100``. Percentiles
        use linear interpolation.
    """

    per_bin: pd.DataFrame
    per_class: pd.DataFrame


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
    classes = tuple(grid.feature_class.values)
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
    # repeating class strings, and geometry is derived from actual edges.
    point_counts = np.empty(len(classes) * n_retained, dtype=np.uint64)
    for ordinal, plane in enumerate(grid.values):
        point_counts[ordinal * n_retained : (ordinal + 1) * n_retained] = plane[retained]
    per_bin = pd.DataFrame(
        {
            "feature_class": pd.Categorical.from_codes(
                np.repeat(np.arange(len(classes)), n_retained), categories=classes
            ),
            "y_bin": np.tile(y_bin, len(classes)),
            "x_bin": np.tile(x_bin, len(classes)),
            "x": np.tile(grid.x.values[x_bin], len(classes)),
            "y": np.tile(grid.y.values[y_bin], len(classes)),
            "bin_area": np.tile(areas, len(classes)),
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
        per_bin["bin_area_um2"] = np.tile(areas_um2, len(classes))

    n_total = retained.size
    population = {
        "n_total_bins": n_total,
        "n_retained_bins": n_retained,
        "n_excluded_bins": n_total - n_retained,
        "pct_excluded_bins": 100 * (n_total - n_retained) / n_total,
    }
    class_rows = []
    for ordinal, feature_class in enumerate(classes):
        # Each class occupies n_retained consecutive rows, so select its block
        # directly instead of grouping or filtering the entire per-bin frame.
        class_bins = per_bin.iloc[ordinal * n_retained : (ordinal + 1) * n_retained]
        counts = class_bins["n_points"].to_numpy(copy=False)
        total = counts.sum(dtype=np.uint64)
        n_zero = int(np.count_nonzero(counts == 0))
        row = {
            "feature_class": feature_class,
            **population,
            "n_points": total,
            "n_retained_bins_without_class": n_zero,
            "pct_retained_bins_without_class": 100 * n_zero / n_retained if n_retained else np.nan,
            "mean_points_per_bin": float(counts.mean()) if n_retained else np.nan,
            "median_points_per_bin": float(np.median(counts)) if n_retained else np.nan,
            "p95_points_per_bin": float(np.percentile(counts, 95)) if n_retained else np.nan,
        }
        class_rows.append(row)
    per_class = pd.DataFrame(class_rows)
    per_class["feature_class"] = pd.Categorical(per_class["feature_class"], categories=classes)

    return SpatialBinSummary(per_bin=per_bin, per_class=per_class)
