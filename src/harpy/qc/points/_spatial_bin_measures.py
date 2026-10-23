"""The measures per spatial bin that the spatial-bin histogram selects from a points summary, by class."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from harpy.qc._panel_normalization import _panel_feature_count
from harpy.qc._summary_schema import _FEATURE_CLASS_KEY, _N_POINTS_KEY, _PCT_POINTS_KEY
from harpy.qc.points._summarize_points import PointsSummary

SpatialBinMeasure = Literal["points", "pct_points"]

# The column of SpatialBinSummary.per_bin that each measure reads, and its axis label. The
# column names stay internal: callers select a class and a measure.
_MEASURES = {
    "points": (_N_POINTS_KEY, "points per spatial bin"),
    "pct_points": (_PCT_POINTS_KEY, "share of the spatial bin's points (%)"),
}


def _spatial_bin_values(
    summary: PointsSummary,
    *,
    feature_class: str,
    measure: str,
    normalization: Literal["per_panel_feature"] | None,
) -> tuple[pd.Series, str]:
    """Return one class's measure per retained spatial bin, normalized if requested, and its axis label.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.PointsSummary`.
    ValueError
        If the summary was computed without ``bin_size``; if the class or
        measure is unknown, ``"detected_features"`` is requested, or
        ``"pct_points"`` from a summary of one class; or if ``normalization``
        is unknown or applied to a measure other than ``"points"``.
    """
    if not isinstance(summary, PointsSummary):
        raise TypeError(
            "summary must be a PointsSummary; use spatial_bin_histogram_by_feature() for a FeaturePointsSummary."
        )
    spatial_bins = summary.spatial_bins
    if spatial_bins is None:
        raise ValueError("Spatial-bin measurements are missing; call hp.qc.summarize_points() with bin_size.")
    # per_class has a row for every selected class, also when no bin is retained and per_bin is empty.
    classes = [str(name) for name in spatial_bins.per_class[_FEATURE_CLASS_KEY]]
    if feature_class not in classes:
        raise ValueError(f"Unknown feature_class {feature_class!r}; choose from {classes}.")
    if measure == "detected_features":
        raise ValueError(f"measure 'detected_features' is not computed per spatial bin; choose from {list(_MEASURES)}.")
    if measure not in _MEASURES:
        raise ValueError(f"Unknown measure {measure!r}; choose from {list(_MEASURES)}.")
    if normalization not in (None, "per_panel_feature"):
        raise ValueError(f"Unknown normalization: {normalization!r}; use None or 'per_panel_feature'.")
    if measure == "pct_points" and len(summary.feature_classes) < 2:
        raise ValueError(
            "measure 'pct_points' needs a summary of two classes or more; this summary has only "
            f"{list(summary.feature_classes)}."
        )
    column, unit = _MEASURES[measure]
    per_bin = spatial_bins.per_bin
    values = per_bin.loc[per_bin[_FEATURE_CLASS_KEY] == feature_class, column]
    if normalization == "per_panel_feature":
        if measure != "points":
            raise ValueError(
                f"normalization='per_panel_feature' applies to measure 'points' only, found {measure!r}: "
                "a share has no per-panel-feature meaning."
            )
        # A constant divisor: the class's number of panel features, features without points included.
        return values / _panel_feature_count(summary, [feature_class]), (
            f"{feature_class} points per panel feature per spatial bin"
        )
    return values, f"{feature_class} {unit}"
