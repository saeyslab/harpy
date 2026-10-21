"""Per-instance values and statistics of selected features of a table, over a table summary's instances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.abc import CSRDataset
from numpy.typing import NDArray
from spatialdata.models import TableModel

from harpy.qc._summary_schema import (
    _FEATURE_CLASS_KEY,
    _FEATURE_KEY,
    _N_INSTANCES_KEY,
    _N_POINTS_KEY,
    _REGION_COLUMN_KEY,
)
from harpy.qc._summary_statistics import _points_statistics
from harpy.qc.table._summarize_table import (
    TableSummary,
    _are_counts,
    _collect_class_matrices,
    _FeatureClassMatrix,
    _sliced_blocks,
    _spatialdata_annotation,
    _with_region_categories,
)
from harpy.qc.table._table_summary_metadata import TableSummaryMetadata


@dataclass(frozen=True)
class FeatureTableSummary:
    """Per-instance values and statistics of explicitly selected features of a table.

    Returned by :func:`harpy.qc.summarize_table_by_feature`, over the instances
    of a :class:`~harpy.qc.TableSummary`, as
    :class:`~harpy.qc.FeaturePointsSummary` summarizes selected features over the
    retained bins of a :class:`~harpy.qc.PointsSummary`.

    Attributes
    ----------
    metadata
        The reference summary's :class:`~harpy.qc.TableSummaryMetadata`: the
        selected regions and the matrices read.
    per_instance
        Long: one row per requested feature and instance of the reference
        summary, in feature order, then table row order. Columns: the region
        and instance identity, under the table's own ``region_key`` and
        ``instance_key`` names; ``feature`` (categorical); and ``n_points``
        (uint64), the feature's points in the instance, zeros included. Its size
        is the number of requested features times the number of instances.
    per_feature
        One row per requested feature, pooled over the selected regions, in the
        order of the summary's classes and then of each matrix's columns.
        Columns, in this order: ``feature`` (categorical); ``feature_class``
        (categorical, the summary's classes as categories), the class of the
        matrix that holds the feature; ``n_instances``, the population column
        of the reference summary's ``per_class``; ``n_points`` (uint64);
        ``n_instances_without_feature`` and ``pct_instances_without_feature``;
        and ``mean_points_per_instance``, ``median_…``, ``std_…`` and
        ``p95_…``. All statistics include the zeros of the instances; SD uses
        ``ddof=1`` and is NaN for fewer than two instances; the percentile uses
        linear interpolation. These are the columns of
        :class:`~harpy.qc.FeatureSpatialBinSummary`'s ``per_feature``, instances
        in place of bins, with one population column in place of four. The
        totals, mean and SD equal those of the reference summary's
        ``per_feature``, which has no median or percentile.
    per_region_feature
        The columns of ``per_feature``, computed within each selected region,
        with a categorical ``region`` column first: one row per selected region
        and requested feature.
    """

    metadata: TableSummaryMetadata
    per_instance: pd.DataFrame
    per_feature: pd.DataFrame
    per_region_feature: pd.DataFrame


def summarize_table_by_feature(
    adata: AnnData,
    *,
    summary: TableSummary,
    features: str | Sequence[str],
) -> FeatureTableSummary:
    """Summarize selected features of a table over the instances of a table summary.

    The table-level counterpart of :func:`harpy.qc.summarize_points_by_feature`.
    The reference summary decides the regions, the matrices read and the
    instances; the requested features do not change that population, so
    every instance contributes, with a zero for a feature that it did not
    receive. Only the rows of ``summary.per_instance`` are read: rows added to
    the table since the summary are not, so recompute
    :func:`harpy.qc.summarize_table` to include them. Nothing is written to the
    table.

    Parameters
    ----------
    adata
        The reference summary's table, such as ``sdata.tables["table"]``,
        unchanged since the summary.
    summary
        Result of :func:`harpy.qc.summarize_table` on ``adata``. Its metadata
        names the regions and the matrices read; its ``per_instance`` gives the
        instances, and its ``feature_names`` the names of each matrix's
        columns.
    features
        An exact feature name, or a non-empty sequence of distinct names, among
        ``summary.feature_names``. The results keep the order of the summary's
        classes and then of each matrix's columns, not the request order.

    Returns
    -------
    FeatureTableSummary
        The requested features' points per instance, and their
        statistics, pooled and per region. See :class:`FeatureTableSummary`.

    Raises
    ------
    TypeError
        If ``adata`` is not an :class:`~anndata.AnnData`, or ``summary`` is not
        a :class:`~harpy.qc.TableSummary`.
    ValueError
        If the table has no SpatialData annotation; if ``features`` is empty,
        contains duplicates or names that are not among
        ``summary.feature_names``, or a name that more than one column of the
        matrices read carries; if the table, its matrices, their names or its
        instances changed since the summary, or the table is another one; or if
        a requested column holds values that are not counts.

    Notes
    -----
    Only the matrices with a requested feature are read, in memory, lazy or
    backed, and of each only the requested columns of the summary's rows are
    kept: a lazy matrix is sliced chunk by chunk, and a backed one read in row
    blocks, so memory stays bounded by a chunk or block and the requested
    features' values. The names come from
    the summary, not from the table again, and are checked to still fit the
    matrices from metadata alone, before a value is read. ``X`` or the layer
    must still have the summary's ``var_names``. An ``.obsm`` matrix is checked
    by its number of columns. If the table registers its column names in
    ``uns[feature_matrices_key]``, those names must also still be the
    summary's. A matrix the table does not register got its names from the
    ``feature_names`` argument of :func:`harpy.qc.summarize_table`; the table
    does not store them, so only the number of columns can be checked.

    See Also
    --------
    harpy.qc.summarize_table
    harpy.qc.instance_histogram_by_feature
    harpy.qc.summarize_points_by_feature

    Examples
    --------
    .. code-block:: python

        adata = sdata.tables["table"]
        summary = hp.qc.summarize_table(adata, obsm_keys=["negative"])
        features = hp.qc.summarize_table_by_feature(
            adata, summary=summary, features=["Negative01", "Negative02", "GeneA"]
        )
        features.per_feature  # the median and p95 per instance of each feature
        hp.qc.instance_histogram_by_feature(features, feature="Negative01")

        # Every probe of one control class, named from the summary.
        controls = hp.qc.summarize_table_by_feature(
            adata, summary=summary, features=summary.feature_names["negative"]
        )
    """
    attrs = _spatialdata_annotation(adata, function="summarize_table_by_feature")
    if not isinstance(summary, TableSummary):
        raise TypeError("summary must be a TableSummary returned by hp.qc.summarize_table().")
    metadata = summary.metadata

    region_key = attrs[TableModel.REGION_KEY_KEY]
    # The summary's instances, indexed by their .obs names, with their region and instance id.
    identity = summary.per_instance[[region_key, attrs[TableModel.INSTANCE_KEY]]]
    requested_features = _requested_features(features, feature_names=summary.feature_names)
    classes = _collect_class_matrices(adata, layer=metadata.layer, obsm_keys=metadata.obsm_keys)
    _check_names_still_fit(adata, classes, summary=summary)
    positions = _instance_positions(adata, identity)
    values = _read_columns(classes, requested_features, positions=positions)
    n_instances = len(identity)

    names = tuple(feature.name for feature in requested_features)
    feature_labels = tuple(feature.label for feature in requested_features)
    labels = summary.feature_classes
    per_instance = identity.iloc[np.tile(np.arange(n_instances), len(names))].reset_index(drop=True)
    per_instance[_FEATURE_KEY] = pd.Categorical.from_codes(
        np.repeat(np.arange(len(names)), n_instances), categories=list(names)
    )
    # Feature by feature, each in table row order.
    per_instance[_N_POINTS_KEY] = values.T.reshape(-1)

    per_feature = _feature_statistics(values, features=names, feature_labels=feature_labels, labels=labels)
    regions = metadata.region
    region_codes = pd.Categorical(identity[region_key].astype(str).to_numpy(), categories=regions).codes
    region_frames = []
    for code, name in enumerate(regions):
        frame = _feature_statistics(
            values[region_codes == code],
            features=names,
            feature_labels=feature_labels,
            labels=labels,
        )
        frame.insert(0, _REGION_COLUMN_KEY, name)
        region_frames.append(frame)
    per_region_feature = _with_region_categories(pd.concat(region_frames, ignore_index=True), regions)
    return FeatureTableSummary(
        metadata=metadata,
        per_instance=per_instance,
        per_feature=per_feature,
        per_region_feature=per_region_feature,
    )


@dataclass(frozen=True)
class _RequestedFeature:
    """A requested feature: the class label of the matrix that holds it, its column there, and its name.

    The class label is the matrix's name in snake case, ``"x"`` for ``X``, as
    :class:`_FeatureClassMatrix` derives it.
    """

    label: str
    column: int
    name: str


def _requested_features(
    features: str | Sequence[str], *, feature_names: Mapping[str, tuple[str, ...]]
) -> list[_RequestedFeature]:
    """Return the requested features, in the order of the summary's classes and then of each matrix's columns."""
    if isinstance(features, str):
        requested_names = (features,)
    elif isinstance(features, Sequence) and features and all(isinstance(name, str) for name in features):
        requested_names = tuple(features)
    else:
        raise ValueError("features must contain one or more exact feature names of the summary.")
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("features must not contain duplicate names.")
    locations: dict[str, list[tuple[str, int]]] = {}
    for label, names in feature_names.items():
        for column, name in enumerate(names):
            locations.setdefault(name, []).append((label, column))
    unknown = [name for name in requested_names if name not in locations]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}; choose from the names in summary.feature_names.")
    ambiguous = {name: [label for label, _ in locations[name]] for name in requested_names if len(locations[name]) > 1}
    if ambiguous:
        raise ValueError(f"More than one column of the matrices read carries these names, by class: {ambiguous}.")
    order = {label: position for position, label in enumerate(feature_names)}
    return sorted(
        (_RequestedFeature(*locations[name][0], name=name) for name in requested_names),
        key=lambda feature: (order[feature.label], feature.column),
    )


def _check_names_still_fit(adata: AnnData, classes: tuple[_FeatureClassMatrix, ...], *, summary: TableSummary) -> None:
    """Raise unless the matrices read are still the summary's classes, with its names, from metadata alone.

    ``X`` or the layer must still have the summary's names as ``var_names``. An
    ``.obsm`` matrix is checked by its number of columns. If the table registers
    its column names in ``uns[feature_matrices_key]``, those names must also
    still be the summary's. A matrix the table does not register got its names
    from the ``feature_names`` argument of :func:`harpy.qc.summarize_table`; the
    table does not store them, so only the number of columns can be checked.
    """
    labels = tuple(entry.label for entry in classes)
    if labels != summary.feature_classes:
        raise ValueError(
            f"The matrices read from the table are now the classes {list(labels)}, not the summary's "
            f"{list(summary.feature_classes)}; recompute summarize_table()."
        )
    var_names = tuple(str(name) for name in adata.var_names)
    registry = adata.uns.get(summary.metadata.feature_matrices_key)
    records = registry if isinstance(registry, Mapping) else {}
    for entry in classes:
        names = summary.feature_names[entry.label]
        if entry.attribute != "obsm":
            fits = var_names == names
        else:
            fits = entry.matrix.shape[1] == len(names)
            record = records.get(entry.key)
            if fits and isinstance(record, Mapping) and "feature_columns" in record:
                fits = tuple(str(name) for name in record["feature_columns"]) == names
        if not fits:
            raise ValueError(
                f"adata.{entry.location} no longer has the columns that the summary named; recompute summarize_table()."
            )


def _instance_positions(adata: AnnData, identity: pd.DataFrame) -> NDArray[np.int64]:
    """Return the table row of each of the summary's instances, in the summary's order.

    ``identity`` holds the summary's instances: their ``.obs`` names as index,
    and their region and instance id as columns. Each is located by name, and
    must still have its region and instance id there.
    """
    if not adata.obs_names.is_unique:
        raise ValueError("The .obs names of the table are not unique, so its instances cannot be located.")
    positions = adata.obs_names.get_indexer(identity.index)
    if (positions < 0).any():
        missing = list(identity.index[positions < 0][:3])
        raise ValueError(
            f"Instances of the summary are not in the table, for example {missing}; pass the summary's table, or "
            "recompute summarize_table()."
        )
    # Finding an .obs name is not enough: it may now label another instance. So check that each
    # row found still has the summary's region and instance id.
    current = adata.obs.iloc[positions]
    same = np.ones(len(positions), dtype=bool)
    for column in identity.columns:
        same &= current[column].astype(str).to_numpy() == identity[column].astype(str).to_numpy()
    if not same.all():
        moved = list(identity.index[~same][:3])
        raise ValueError(
            f"The .obs names of the table label other instances than in the summary, for example {moved}; pass the "
            "summary's table, or recompute summarize_table()."
        )
    return positions


def _read_columns(
    classes: tuple[_FeatureClassMatrix, ...],
    requested_features: list[_RequestedFeature],
    *,
    positions: NDArray[np.int64],
) -> NDArray[np.uint64]:
    """Return the requested features' values per instance, one column per feature, in their order.

    ``positions`` gives the table row of each of the summary's instances, in
    its order. An in-memory or lazy matrix is sliced: Dask slices a lazy
    one chunk by chunk, all lazy matrices in one compute. A backed matrix is read
    in row blocks, as :func:`harpy.qc.summarize_table` reads it, as a slice of
    its rows would load them whole, with all their columns.
    """
    values = np.zeros((len(positions), len(requested_features)), dtype=np.uint64)
    lazy = []
    for entry in classes:
        # The columns of this class's features in the results, and in its matrix.
        result_columns = [index for index, feature in enumerate(requested_features) if feature.label == entry.label]
        if not result_columns:
            continue
        columns = np.asarray([requested_features[index].column for index in result_columns])
        if isinstance(entry.matrix, CSRDataset):
            values[:, result_columns] = _backed_columns(
                entry.matrix, positions, columns=columns, location=entry.location
            )
            continue
        selected = entry.matrix[positions][:, columns]
        if isinstance(selected, da.Array):
            lazy.append((result_columns, selected, entry.location))
            continue
        # In memory: the slice already holds the values.
        values[:, result_columns] = _to_dense_counts(selected, location=entry.location)
    if lazy:
        # Lazy: compute all slices at once, then write them as the in-memory path does.
        computed = dask.compute(*(selected for _, selected, _ in lazy))
        for (result_columns, _, location), selected in zip(lazy, computed, strict=True):
            values[:, result_columns] = _to_dense_counts(selected, location=location)
    return values


def _backed_columns(
    matrix: CSRDataset, positions: NDArray[np.int64], *, columns: NDArray[np.int64], location: str
) -> NDArray[np.uint64]:
    """Return the values in ``columns`` of a backed matrix's rows ``positions``, read in row blocks."""
    # Each table row's row in the results, -1 for a row outside the summary's regions.
    output_rows = np.full(matrix.shape[0], -1, dtype=np.int64)
    output_rows[positions] = np.arange(len(positions))
    values = np.zeros((len(positions), len(columns)), dtype=np.uint64)
    for block, rows_in_block in _sliced_blocks(matrix, output_rows):
        wanted = rows_in_block >= 0
        values[rows_in_block[wanted]] = _to_dense_counts(block[wanted][:, columns], location=location)
    return values


def _to_dense_counts(selected: object, *, location: str) -> NDArray[np.uint64]:
    """Return the selected CSR rows and columns as a dense array of counts, raising unless they are counts."""
    if not _are_counts(selected.data):
        raise ValueError(
            f"adata.{location} holds values that are not counts, non-negative whole numbers, in the requested "
            "features; recompute summarize_table() if the matrix changed since the summary."
        )
    # toarray adds up the entries that a non-canonical CSR matrix stores more than once.
    return np.asarray(selected.toarray()).astype(np.uint64)


def _feature_statistics(
    values: NDArray[np.uint64],
    *,
    features: tuple[str, ...],
    feature_labels: tuple[str, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    """Return one row per requested feature from its values per instance, zeros included."""
    n_instances = values.shape[0]
    rows = []
    for column, (name, label) in enumerate(zip(features, feature_labels, strict=True)):
        counts = values[:, column]
        n_zero = int(np.count_nonzero(counts == 0))
        rows.append(
            {
                _FEATURE_KEY: name,
                _FEATURE_CLASS_KEY: label,
                _N_INSTANCES_KEY: n_instances,
                _N_POINTS_KEY: counts.sum(dtype=np.uint64),
                "n_instances_without_feature": n_zero,
                "pct_instances_without_feature": 100 * n_zero / n_instances if n_instances else np.nan,
                **_points_statistics(counts, unit="instance"),
            }
        )
    per_feature = pd.DataFrame(rows)
    per_feature[_N_POINTS_KEY] = per_feature[_N_POINTS_KEY].astype(np.uint64)
    per_feature[_FEATURE_KEY] = pd.Categorical(per_feature[_FEATURE_KEY], categories=list(features))
    per_feature[_FEATURE_CLASS_KEY] = pd.Categorical(per_feature[_FEATURE_CLASS_KEY], categories=labels)
    return per_feature
