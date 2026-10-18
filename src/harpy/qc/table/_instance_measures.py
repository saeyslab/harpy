"""The measures per instance that the instance plots select from a table summary, by class."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from harpy.qc._panel_normalization import _panel_feature_count
from harpy.qc._summary_schema import (
    _N_CLASS_DETECTED_FEATURES_TEMPLATE,
    _N_CLASS_POINTS_TEMPLATE,
    _PCT_CLASS_POINTS_TEMPLATE,
)
from harpy.qc.table._summarize_table import TableSummary

InstanceMeasure = Literal["points", "detected_features", "pct_points"]

# The column of TableSummary.per_instance that each measure reads, and its axis label. The
# column names stay internal: callers select a class and a measure.
_MEASURES = {
    "points": (_N_CLASS_POINTS_TEMPLATE, "points per instance"),
    "detected_features": (_N_CLASS_DETECTED_FEATURES_TEMPLATE, "detected features per instance"),
    "pct_points": (_PCT_CLASS_POINTS_TEMPLATE, "share of the instance's points (%)"),
}


def _instance_values(
    summary: TableSummary,
    *,
    feature_class: str,
    measure: str,
    normalization: Literal["per_panel_feature"] | None,
) -> tuple[pd.Series, str]:
    """Return one class's measure per retained instance, normalized if requested, and its axis label.

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
    column, unit = _MEASURES[measure]
    column = column.format(label=feature_class)
    if column not in summary.per_instance:
        raise ValueError(
            "measure 'pct_points' needs a summary of two classes or more; this summary has only "
            f"{list(summary.feature_classes)}."
        )
    values = summary.per_instance[column]
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
