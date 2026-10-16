"""Instance-level QC summary of a class-aware aggregation table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.qc._summary_schema import (
    _FEATURE_CLASS_KEY,
    _N_FEATURES_KEY,
    _N_POINTS_KEY,
    _PER_PANEL_FEATURE_SUFFIX,
    _REGION_COLUMN_KEY,
)
from harpy.qc._summary_statistics import _points_statistics
from harpy.qc.table._table_summary_metadata import TableSummaryMetadata
from harpy.table._metadata import _POINT_AGGREGATION_KEY


@dataclass(frozen=True)
class TableSummary:
    """Computed instance-level summary of one class-aware aggregation table.

    Returned by :func:`harpy.qc.summarize_table`. The counterpart of
    :class:`~harpy.qc.PointsSummary` for the instances of a
    table: it describes the points that were assigned to instances, not the
    original points.

    Attributes
    ----------
    metadata
        One :class:`TableSummaryMetadata` record: the table, the selected
        regions and classes, the expression class, the source points element of
        each region and the denominators of the per-panel-feature columns.
    per_instance
        One row per retained instance, indexed by the table's ``.obs`` index.
        Wide, one column per class, to stay aligned with ``.obs``. Columns:

        - the region and instance identity, under the table's own
          ``region_key`` and ``instance_key`` names;
        - the count column of each selected class, copied from ``.obs``: the
          class's points in the instance, for example ``n_negative_points`` for
          the class ``Negative``;
        - ``auxiliary_points_fraction``, copied from ``.obs``: the instance's
          points of all auxiliary classes divided by all its points;
        - for each selected class, the class's points per panel feature of the
          same class: the average number of points that each of the class's
          panel features contributes to the instance. For example, for a class
          ``Negative``: ``n_negative_points_per_panel_feature`` =
          ``n_negative_points`` / ``metadata.panel_feature_counts["Negative"]``.

        How the names arise::

            panel class -> count column (.obs)  -> per-panel-feature column
            Negative    -> n_negative_points    -> n_negative_points_per_panel_feature

        The count column names come from the ``count_columns`` of the
        aggregation record that :func:`~harpy.tb.aggregate_points` writes,
        which derives them from the panel's class names.
    per_class
        One row per selected ``feature_class`` (categorical), pooled over the
        selected regions; ``per_region`` holds the same statistics per region.
        ``n_total_instances`` counts the table's rows of the
        selected regions, ``n_retained_instances`` those with at least one point
        across the selected classes, and ``n_excluded_instances`` the others;
        ``pct_excluded_instances`` is 100 times excluded / total. These
        population counts are the same for every class.
        ``n_retained_instances_without_class`` counts retained instances with
        zero points of this row's class, and
        ``pct_retained_instances_without_class`` is 100 times that count /
        retained. ``n_points``, ``mean_points_per_instance``,
        ``median_points_per_instance``, ``std_points_per_instance`` and
        ``p95_points_per_instance`` describe the class's points over the
        retained instances, zeros included. ``n_features`` is the class's
        number of panel features, and ``mean_points_per_panel_feature_per_instance``,
        ``median_…``, ``std_…`` and ``p95_…`` are the same statistics of its
        per-panel-feature column, the count statistics divided by
        ``n_features``. SD uses ``ddof=1`` and is NaN for fewer than two
        instances; the percentile uses linear interpolation.
    per_region
        The statistics of ``per_class``, computed within each selected region,
        with a categorical ``region`` column: one row per selected region and
        selected class. With one selected region, it holds the values of
        ``per_class``.
    """

    metadata: TableSummaryMetadata
    per_instance: pd.DataFrame
    per_class: pd.DataFrame
    per_region: pd.DataFrame


def summarize_table(
    sdata: SpatialData,
    table_name: str,
    *,
    region: str | Sequence[str] | None = None,
    feature_classes: str | Sequence[str] | None = None,
) -> TableSummary:
    """Summarize the instances of a class-aware aggregation table, per class.

    The table-level counterpart of :func:`harpy.qc.summarize_points`: it
    describes the instances of a table created by
    :func:`harpy.tb.aggregate_points` in class-aware mode, from their
    per-class point counts, rather than the original points. It derives, per
    instance, each class's points per panel feature, and summarizes the
    instances per class, pooled and per region. Nothing is written to the
    table.

    Parameters
    ----------
    sdata
        SpatialData object containing the table.
    table_name
        Name of the table in ``sdata.tables``.
    region
        Annotated elements whose instances to summarize: names from the table's
        SpatialData annotation, ``uns["spatialdata_attrs"]["region"]``,
        whatever the type of the annotated elements. ``None`` selects every
        region the table annotates. The annotated elements need not be present
        in ``sdata``.
    feature_classes
        Classes to summarize, from the aggregation record's ``classes``.
        ``None`` selects every recorded class.

    Returns
    -------
    TableSummary
        The instance metrics in ``per_instance``, the class statistics pooled
        over the selected regions in ``per_class`` and per region in
        ``per_region``, and their context in ``metadata``. Column names follow
        the table's ``count_columns``; see :class:`TableSummary`.

    Raises
    ------
    ValueError
        If ``table_name`` is not a table of ``sdata``; if the table has no
        SpatialData annotation or no ``uns["point_aggregation"]``
        record, so is not class-aware; if a selected class's count column, or
        the auxiliary-points-fraction column, is missing from ``.obs``; if a
        selected class has no positive panel feature count; if a selected region
        is missing from the record's ``regions``; or if ``region`` or
        ``feature_classes`` is empty, contains duplicates or unknown names.

    Notes
    -----
    **Population.** Within the selected regions, an instance is retained when it
    has at least one point across the selected classes, as
    :func:`harpy.qc.summarize_points` retains bins. Zeros of an individual class
    within that population are kept, so separate calls for single classes may
    retain different instances. The counts describe only the table's rows: the
    function does not reconstruct instances absent from the table, nor include
    points that were not assigned to an instance.

    **Inputs.** Only ``.obs``, ``.uns`` and the table annotation are read, which
    are in memory for in-memory, lazy and storage-backed tables alike, so the
    function reads no matrix and loads nothing, whatever the table's mode. It
    validates only the aggregation-record fields that it uses, not the complete
    class-aware table contract, which :func:`harpy.tb.validate_table` checks.

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_table(sdata, "table_transcriptomics")
        summary.per_class  # one row per class, pooled over the regions

        # Only the controls, in one region.
        summary = hp.qc.summarize_table(
            sdata, "table_transcriptomics", region="sample_a_labels", feature_classes=["Negative", "SystemControl"]
        )
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} is not in sdata; available tables: {sorted(sdata.tables)}.")
    adata = sdata.tables[table_name]
    attrs = adata.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Table {table_name!r} has no SpatialData annotation (uns[{TableModel.ATTRS_KEY!r}]).")
    record = adata.uns.get(_POINT_AGGREGATION_KEY)
    if not isinstance(record, Mapping):
        raise ValueError(
            f"Table {table_name!r} is not class-aware: it has no uns[{_POINT_AGGREGATION_KEY!r}] record, "
            "which hp.tb.aggregate_points writes when an expression_class is given."
        )

    recorded_classes = _names(record["classes"])
    selected_classes = _selected_names(
        feature_classes, recorded=recorded_classes, parameter="feature_classes", kind="feature classes"
    )
    annotated_regions = _names(attrs[TableModel.REGION_KEY])
    selected_regions = _selected_names(region, recorded=annotated_regions, parameter="region", kind="regions")

    count_columns = {str(name): str(column) for name, column in record["count_columns"].items()}
    fraction_column = str(record["auxiliary_points_fraction_column"])
    panel_feature_counts = {str(name): int(count) for name, count in record["panel_feature_counts"].items()}
    record_regions = record["regions"]
    obs = adata.obs
    missing_columns = [
        column for column in [*(count_columns[name] for name in selected_classes), fraction_column] if column not in obs
    ]
    if missing_columns:
        raise ValueError(f"Table {table_name!r} is missing aggregation columns in .obs: {missing_columns}.")
    invalid_counts = [name for name in selected_classes if panel_feature_counts.get(name, 0) <= 0]
    if invalid_counts:
        raise ValueError(
            f"Table {table_name!r} records no positive panel feature count for the classes {invalid_counts}."
        )
    unrecorded_regions = [name for name in selected_regions if name not in record_regions]
    if unrecorded_regions:
        raise ValueError(f"The aggregation record of table {table_name!r} has no regions {unrecorded_regions}.")

    region_key = attrs[TableModel.REGION_KEY_KEY]
    instance_key = attrs[TableModel.INSTANCE_KEY]
    selected_count_columns = [count_columns[name] for name in selected_classes]
    # Select on .obs only: no matrix is read, and the AnnData itself is never copied.
    in_regions = obs[region_key].isin(selected_regions).to_numpy()
    rows = obs.loc[in_regions, [region_key, instance_key, *selected_count_columns, fraction_column]]
    retained = (rows[selected_count_columns].sum(axis=1) > 0).to_numpy()
    per_instance = rows.loc[retained].copy()
    for name, column in zip(selected_classes, selected_count_columns, strict=True):
        per_instance[f"{column}{_PER_PANEL_FEATURE_SUFFIX}"] = per_instance[column] / panel_feature_counts[name]

    per_class = _class_statistics(
        rows,
        retained,
        selected_classes=selected_classes,
        count_columns=count_columns,
        panel_feature_counts=panel_feature_counts,
    )
    region_frames = []
    for name in selected_regions:
        in_region = (rows[region_key] == name).to_numpy()
        frame = _class_statistics(
            rows.loc[in_region],
            retained[in_region],
            selected_classes=selected_classes,
            count_columns=count_columns,
            panel_feature_counts=panel_feature_counts,
        )
        frame.insert(0, _REGION_COLUMN_KEY, name)
        region_frames.append(frame)
    per_region = pd.concat(region_frames, ignore_index=True)
    per_region[_REGION_COLUMN_KEY] = pd.Categorical(per_region[_REGION_COLUMN_KEY], categories=selected_regions)

    metadata = TableSummaryMetadata(
        table_name=table_name,
        region=selected_regions,
        feature_classes=selected_classes,
        expression_class=str(record["expression_class"]),
        points_name_by_region={name: str(record_regions[name]["points_element"]) for name in selected_regions},
        panel_feature_counts={name: panel_feature_counts[name] for name in selected_classes},
    )
    return TableSummary(metadata=metadata, per_instance=per_instance, per_class=per_class, per_region=per_region)


def _names(value: object) -> tuple[str, ...]:
    """Return a recorded name or sequence of names as a tuple of strings."""
    if isinstance(value, str):
        return (value,)
    return tuple(str(name) for name in value)


def _selected_names(
    requested: str | Sequence[str] | None, *, recorded: tuple[str, ...], parameter: str, kind: str
) -> tuple[str, ...]:
    """Validate a selection of names, as summarize_points validates its classes, and keep the recorded order."""
    if requested is None:
        return recorded
    if not isinstance(requested, (str, Sequence)):
        raise ValueError(f"{parameter} must contain one or more exact names of {kind}.")
    names = (requested,) if isinstance(requested, str) else tuple(requested)
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{parameter} must contain one or more exact names of {kind}.")
    if len(set(names)) != len(names):
        raise ValueError(f"{parameter} must not contain duplicate names.")
    unknown = set(names) - set(recorded)
    if unknown:
        raise ValueError(f"Unknown {kind}: {sorted(unknown)}; choose from {list(recorded)}.")
    return tuple(name for name in recorded if name in names)


def _class_statistics(
    rows: pd.DataFrame,
    retained: np.ndarray,
    *,
    selected_classes: tuple[str, ...],
    count_columns: Mapping[str, str],
    panel_feature_counts: Mapping[str, int],
) -> pd.DataFrame:
    """Return one row of population counts and statistics per selected class."""
    n_total = len(rows)
    n_retained = int(retained.sum())
    population = {
        "n_total_instances": n_total,
        "n_retained_instances": n_retained,
        "n_excluded_instances": n_total - n_retained,
        "pct_excluded_instances": 100 * (n_total - n_retained) / n_total if n_total else np.nan,
    }
    class_rows = []
    for name in selected_classes:
        counts = rows.loc[retained, count_columns[name]].to_numpy()
        n_zero = int(np.count_nonzero(counts == 0))
        n_features = panel_feature_counts[name]
        class_rows.append(
            {
                _FEATURE_CLASS_KEY: name,
                **population,
                "n_retained_instances_without_class": n_zero,
                "pct_retained_instances_without_class": 100 * n_zero / n_retained if n_retained else np.nan,
                _N_POINTS_KEY: int(counts.sum()),
                **_points_statistics(counts, unit="instance"),
                _N_FEATURES_KEY: n_features,
                **_points_statistics(counts / n_features, unit="panel_feature_per_instance"),
            }
        )
    per_class = pd.DataFrame(class_rows)
    per_class[_FEATURE_CLASS_KEY] = pd.Categorical(per_class[_FEATURE_CLASS_KEY], categories=selected_classes)
    return per_class
