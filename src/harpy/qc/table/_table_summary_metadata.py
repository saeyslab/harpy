from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TableSummaryMetadata:
    """Immutable context owned by a table summary.

    The fields describe the computation's inputs: the table, the regions and
    the matrices read, taken from the arguments and the table's SpatialData
    annotation. They are not a live metadata lookup. Results of the
    computation, such as the class labels ``feature_classes`` and the
    denominators ``panel_feature_counts``, live in the summary.

    Attributes
    ----------
    table_name
        Name of the summarized table.
    region
        Selected regions, the annotated elements whose rows were summarized,
        resolved from ``None`` to every region the table annotates, in the
        table's order.
    layer
        The matrix aligned with ``var`` that was read: ``None`` for ``X``, or
        the name of a layer.
    obsm_keys
        The ``.obsm`` matrices read, in the order passed.
    """

    table_name: str
    region: tuple[str, ...]
    layer: str | None
    obsm_keys: tuple[str, ...]
