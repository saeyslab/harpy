"""Panel-feature normalization shared by the plots of a points summary."""

from __future__ import annotations

from collections.abc import Iterable

from harpy.qc.points._summarize_points import PointsSummary


def _panel_feature_count(summary: PointsSummary, feature_classes: Iterable[str]) -> int:
    """Return the summed panel feature count of the given classes, the divisor of ``per_panel_feature``.

    Each class's count includes panel features without detections, as recorded in
    ``summary.panel_feature_counts``.

    Raises
    ------
    ValueError
        If a class has no panel feature count in the summary, or a count is not positive.
    """
    panel_sizes = summary.panel_feature_counts
    feature_classes = list(feature_classes)
    if any(name not in panel_sizes or panel_sizes[name] <= 0 for name in feature_classes):
        raise ValueError("Panel-feature normalization requires a positive panel size for every displayed class.")
    return sum(panel_sizes[name] for name in feature_classes)
