from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TableSummaryMetadata:
    """Immutable context owned by a table summary.

    The fields describe the computation's inputs, taken from the table's
    ``uns["point_aggregation"]`` record and its SpatialData annotation;
    they are not a live metadata lookup. The mappings are copies, so later
    changes to the table do not affect them.

    Attributes
    ----------
    table_name
        Name of the summarized table.
    region
        Selected regions, the annotated elements whose rows were summarized,
        resolved from ``None`` to every region the table annotates, in the
        table's order.
    feature_classes
        Selected classes, resolved from ``None`` to every recorded class, in the
        record's order.
    expression_class
        Class whose features form the table's ``X``, copied from the aggregation
        record. It need not be among the selected classes.
    points_name_by_region
        For each selected region, the name of the points element aggregated into
        it. Through that points element, the class definitions remain traceable
        to its feature panel.
    panel_feature_counts
        For each selected class, the number of unique panel features, features
        without detections included: the denominator of its per-panel-feature
        column.
    """

    table_name: str
    region: tuple[str, ...]
    feature_classes: tuple[str, ...]
    expression_class: str
    points_name_by_region: Mapping[str, str]
    panel_feature_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        # Copy the mappings, so that the record does not share them with its source.
        object.__setattr__(self, "points_name_by_region", dict(self.points_name_by_region))
        object.__setattr__(self, "panel_feature_counts", dict(self.panel_feature_counts))
