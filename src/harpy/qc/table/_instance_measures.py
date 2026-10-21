"""The measures per instance that the instance plots select from a table summary, by class."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from harpy.qc._panel_normalization import _panel_feature_count
from harpy.qc._summary_schema import _N_DETECTED_FEATURES_KEY, _N_POINTS_KEY, _PCT_POINTS_KEY, _class_column
from harpy.qc.table._summarize_table import TableSummary

InstanceMeasure = Literal["points", "detected_features", "pct_points"]

# The base name of the column of TableSummary.per_instance that each measure reads, followed per
# class by its label, and its axis label. The column names stay internal: callers select a class
# and a measure.
_MEASURES = {
    "points": (_N_POINTS_KEY, "points per instance"),
    "detected_features": (_N_DETECTED_FEATURES_KEY, "detected features per instance"),
    "pct_points": (_PCT_POINTS_KEY, "share of the instance's points (%)"),
}


def _instance_values(
    summary: TableSummary,
    *,
    feature_class: str,
    measure: str,
    normalization: Literal["per_panel_feature"] | None,
) -> tuple[pd.Series, str]:
    """Return one class's measure per instance, normalized if requested, and its axis label.

    With ``"pct_points"``, the values are missing (NaN) for instances without
    points in any class read.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.TableSummary`.
    ValueError
        If the class or measure is unknown, ``"pct_points"`` is requested from
        a summary of one class, or ``normalization`` is unknown or applied to a
        measure other than ``"points"``.
    """
    if not isinstance(summary, TableSummary):
        raise TypeError("summary must be a TableSummary returned by hp.qc.summarize_table().")
    if feature_class not in summary.feature_classes:
        raise ValueError(f"Unknown feature_class {feature_class!r}; choose from {list(summary.feature_classes)}.")
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
    values = summary.per_instance[_class_column(column, label=feature_class)]
    if normalization == "per_panel_feature":
        if measure != "points":
            raise ValueError(
                f"normalization='per_panel_feature' applies to measure 'points' only, found {measure!r}: "
                "a share or a number of detected features has no per-panel-feature meaning."
            )
        # A constant divisor: the class's number of features in the matrices read.
        return values / _panel_feature_count(summary, [feature_class]), (
            f"{feature_class} points per panel feature per instance"
        )
    return values, f"{feature_class} {unit}"
