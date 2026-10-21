"""Instance-level QC summary of the CSR count matrices of a table."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.abc import CSRDataset
from numpy.typing import NDArray
from scipy import sparse
from spatialdata.models import TableModel

from harpy._feature_panels import _require_mapping
from harpy._storage._anndata import _auto_sparse_block_length
from harpy.qc._summary_reduction import _tree_reduce
from harpy.qc._summary_schema import (
    _FEATURE_CLASS_KEY,
    _FEATURE_KEY,
    _N_CLASS_DETECTED_FEATURES_TEMPLATE,
    _N_CLASS_POINTS_TEMPLATE,
    _N_FEATURES_KEY,
    _N_INSTANCES_KEY,
    _N_INSTANCES_WITHOUT_CLASS_KEY,
    _N_INSTANCES_WITHOUT_FEATURE_KEY,
    _N_POINTS_KEY,
    _PCT_CLASS_POINTS_TEMPLATE,
    _PCT_INSTANCES_WITHOUT_CLASS_KEY,
    _PCT_INSTANCES_WITHOUT_FEATURE_KEY,
    _REGION_COLUMN_KEY,
    _WITHIN_CLASS_FRACTION_KEY,
    _statistic_name,
)
from harpy.qc._summary_statistics import _points_statistics
from harpy.qc.table._table_summary_metadata import TableSummaryMetadata
from harpy.table._utils import _spatialdata_annotation
from harpy.table._validation import _metadata_string_sequence, _validate_feature_matrix_record
from harpy.utils._keys import _FEATURE_MATRICES_KEY
from harpy.utils._names import _names, _selected_names, _snake_case

# The label of the class of X, the name of the matrix in snake case.
_X_LABEL = "x"


@dataclass(frozen=True)
class TableSummary:
    """Computed instance-level summary of the count matrices of one table.

    Returned by :func:`harpy.qc.summarize_table`. The counterpart of
    :class:`~harpy.qc.PointsSummary` for the instances of a table: it describes
    the points that were assigned to instances, not the original points. Each
    matrix read is one class, labeled by the matrix's name in snake case. The
    summary stores counts only, as :class:`~harpy.qc.SpatialBinSummary` does:
    a class's points per panel feature are a count column divided by
    :attr:`panel_feature_counts`.

    Attributes
    ----------
    metadata
        One :class:`TableSummaryMetadata` record: the selected regions and the
        matrices read.
    per_instance
        One row per row of the selected regions, indexed by the table's
        ``.obs`` index. Wide, one column per class, to stay aligned with
        ``.obs``. Columns, in
        this order, each measure for every class in the order of
        :attr:`feature_classes`:

        - the region and instance identity, under the table's own
          ``region_key`` and ``instance_key`` names;
        - ``n_<class>_points``: the class's points in the instance, uint64;
        - ``n_<class>_detected_features``: the class's features with at least
          one point in the instance. Spatial bins have no counterpart, as a
          distinct count of features per bin is expensive;
        - ``pct_<class>_points``, with two classes or more: 100 times the
          class's points / the instance's points across the classes read;
          missing (NaN) for an instance without points in any class read, as
          0 / 0 has no value. The share is not corrected for panel size: a
          class with more features collects more points, also from
          background. Compare it across instances or samples of the same
          panel, or compare classes by their points per feature, with
          ``n_features`` of ``per_class``.

        How the names arise: the class label is the name of the matrix read::

            matrix read          class label     count column
            X                    x               n_x_points
            layers["counts"]     counts          n_counts_points
            obsm["negative"]     negative        n_negative_points

        Compared with :func:`scanpy.pp.calculate_qc_metrics`, for ``X``::

            this summary               scanpy
            n_x_points                 total_counts
            n_x_detected_features      n_genes_by_counts
            n_<class>_points           total_counts_<qc_var>
            pct_<class>_points         pct_counts_<qc_var>

        Scanpy's ``qc_vars`` are boolean groups of ``var`` within ``X``, and
        its percentages are relative to the total counts of ``X``; here each
        class is its own matrix, and the percentages are relative to all
        classes read.
    per_class
        One row per class (categorical ``feature_class``), pooled over the
        selected regions. Columns, in this order: ``n_instances`` counts the
        table's rows of the selected regions, the same for every class.
        ``n_points`` (uint64) is the class's points over these instances;
        ``n_instances_without_class`` counts the instances with zero points of
        the class, and ``pct_instances_without_class`` is 100 times that count /
        ``n_instances``. ``mean_points_per_instance``, ``median_…``, ``std_…``
        and ``p95_…`` describe the class's points over the instances, zeros
        included, and
        ``mean_detected_features_per_instance``, ``median_…``, ``std_…`` and
        ``p95_…`` its detected features. Last, ``n_features`` is the class's
        number of features in the matrices read. SD uses ``ddof=1`` and is NaN
        for fewer than two instances; the percentile uses linear
        interpolation.
    per_region_class
        The statistics of ``per_class``, computed within each selected region,
        with a categorical ``region`` column first: one row per selected
        region and class. With one selected region, it holds the values of
        ``per_class``.
    per_feature
        One row per feature of each class read, features without points
        included, in the order of :attr:`feature_classes` and then of each
        matrix's columns, pooled over the selected regions. Its population is
        that of ``per_class``, zeros included. Columns, in this order:
        ``feature``, the feature's name: from ``var_names`` for ``X`` or a
        layer, and for an ``.obsm`` matrix from its registered
        ``feature_columns`` or the names passed; ``feature_class``
        (categorical), the label of its matrix; ``n_instances``, the
        population column of ``per_class``, the same for every feature;
        ``n_points`` (uint64), the feature's points; ``within_class_fraction``,
        ``n_points`` / the class's points, NaN when the class has none;
        ``n_instances_without_feature`` and ``pct_instances_without_feature``,
        the instances with zero points of the feature, and 100 times that
        count / ``n_instances``; and ``mean_points_per_instance`` and
        ``std_points_per_instance``, the feature's points per instance, SD with
        ``ddof=1`` and NaN for fewer than two instances. Compared with the
        ``var`` columns of :func:`scanpy.pp.calculate_qc_metrics`, which also
        counts over every row of ``X`` or a layer, so that the values agree for
        a summary of every region::

            this summary                                 scanpy
            n_points                                     total_counts
            mean_points_per_instance                     mean_counts
            pct_instances_without_feature                pct_dropout_by_counts
            n_instances - n_instances_without_feature    n_cells_by_counts

        Unlike the ``per_feature`` of :class:`~harpy.qc.PointsSummary`, every
        feature has statistics per instance: they come from the pass that
        sums the classes, while statistics per spatial bin would need the
        count of every panel feature in every bin. Medians and percentiles
        are not computed per feature, as that pass does not keep each column's
        values.
    per_region_feature
        The columns of ``per_feature``, computed within each selected region,
        with a categorical ``region`` column first: one row per selected
        region and feature. Within a region, ``within_class_fraction`` is
        relative to the class's points in that region. With one selected
        region, it holds the values of ``per_feature``.
    """

    metadata: TableSummaryMetadata
    per_instance: pd.DataFrame
    per_class: pd.DataFrame
    per_region_class: pd.DataFrame
    per_feature: pd.DataFrame
    per_region_feature: pd.DataFrame

    @property
    def feature_classes(self) -> tuple[str, ...]:
        """Return the class labels in the order of the results, from ``per_class.feature_class``.

        One label per matrix read, its name in snake case: ``"x"`` for ``X``,
        the layer's name, then each ``.obsm`` key.
        """
        return tuple(str(label) for label in self.per_class[_FEATURE_CLASS_KEY])

    @property
    def feature_names(self) -> dict[str, tuple[str, ...]]:
        """Return each class's feature names in column order, from ``per_feature``.

        ``var_names`` for ``X`` or a layer; for an ``.obsm`` matrix, its
        registered ``feature_columns`` or the names passed to
        :func:`harpy.qc.summarize_table`.
        """
        classes = self.per_feature[_FEATURE_CLASS_KEY].to_numpy()
        features = self.per_feature[_FEATURE_KEY].to_numpy()
        return {label: tuple(str(name) for name in features[classes == label]) for label in self.feature_classes}

    @property
    def panel_feature_counts(self) -> dict[str, int]:
        """Return each class's number of features in the matrices read, from ``per_class.n_features``.

        The denominators of a class's points per panel feature. On a table as
        :func:`harpy.tb.aggregate_points` writes it, these are the panel's
        feature counts, features without detections included; after a subset
        of ``X``, the remaining features.
        """
        return {
            str(label): int(count)
            for label, count in zip(self.per_class[_FEATURE_CLASS_KEY], self.per_class[_N_FEATURES_KEY], strict=True)
        }


@dataclass(frozen=True)
class _FeatureClassMatrix:
    """One matrix read, one class: where it is in the AnnData, the matrix and its column names."""

    # The AnnData attribute that holds the matrix.
    attribute: Literal["X", "layers", "obsm"]
    # Its key in that attribute; None for X.
    key: str | None
    matrix: object
    # Its column names: var_names for X or a layer, the registered or passed names for .obsm.
    feature_names: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Return the class label: the matrix's name in snake case, ``"x"`` for ``X``."""
        return _X_LABEL if self.key is None else _snake_case(self.key)

    @property
    def location(self) -> str:
        """Return where the matrix is in the AnnData, for messages, for example ``"obsm['negative']"``."""
        return "X" if self.key is None else f"{self.attribute}[{self.key!r}]"


def summarize_table(
    adata: AnnData,
    *,
    layer: str | None = None,
    obsm_keys: str | Sequence[str] = (),
    feature_names: Mapping[str, Sequence[str]] | None = None,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
    region: str | Sequence[str] | None = None,
) -> TableSummary:
    """Summarize the instances of a table from its count matrices, per class.

    The table-level counterpart of :func:`harpy.qc.summarize_points`: it
    describes the instances of a table from their counts rather than the
    original points. Each matrix read is one class: ``X``, or a layer, and the
    ``.obsm`` matrices named, for example the class matrices that
    :func:`harpy.tb.aggregate_points` writes in class-aware mode. Per instance,
    it derives each class's points, detected features and share of the
    instance's points, and it summarizes the instances per class, pooled and per
    region. Nothing is written to the table:
    :func:`harpy.tb.pp.calculate_qc_metrics` adds the measures per instance and
    per feature to ``.obs`` and ``.var``.

    Parameters
    ----------
    adata
        The table, such as ``sdata.tables["table"]``, with its SpatialData
        annotation in ``uns["spatialdata_attrs"]``. It need not be part of a
        SpatialData object, and the elements that it annotates need not be
        present.
    layer
        The matrix aligned with ``var`` to read: ``None`` reads ``X``, as in
        scanpy, and a name reads ``adata.layers[layer]``. It is always read when
        present; a table whose ``X`` is ``None`` is summarized from its
        ``.obsm`` matrices alone. After normalization, ``X`` no longer holds
        counts: pass the layer of raw counts.
    obsm_keys
        ``.obsm`` matrices to read as well, each one class, for example
        ``["negative", "system_control"]``. Empty by default, which summarizes
        ``X`` alone. The keys are not inferred from the table: the caller names
        what to summarize.
    feature_names
        For an ``.obsm`` matrix read whose names
        ``adata.uns[feature_matrices_key]`` does not register, its column names:
        a mapping from the ``.obsm`` key to as many unique names as the matrix
        has columns. Each matrix has one source of names: a registered matrix
        takes its ``feature_columns``, and names passed for it raise. ``X`` and
        a layer are named by ``var_names``. The names label the rows of
        ``per_feature``.
    feature_matrices_key
        Key in ``adata.uns`` of the registry that names the columns of the
        ``.obsm`` matrices read; ``"feature_matrices"`` by default. Only the
        entries of the matrices read are checked.
    region
        Annotated elements whose instances to summarize: names from the table's
        SpatialData annotation, ``uns["spatialdata_attrs"]["region"]``,
        whatever the type of the annotated elements. ``None`` selects every
        region the table annotates.

    Returns
    -------
    TableSummary
        The instance metrics in ``per_instance``, the class statistics pooled
        over the selected regions in ``per_class`` and per region in
        ``per_region_class``, the feature statistics in ``per_feature`` and
        ``per_region_feature``, and their context in ``metadata``. Column names
        follow the class labels, the names of the matrices read in snake case:
        ``x`` for ``X``. See :class:`TableSummary`.

    Raises
    ------
    TypeError
        If ``adata`` is not an :class:`~anndata.AnnData`.
    ValueError
        If the table has no SpatialData annotation; if there is nothing to
        read; if ``layer`` or a key of
        ``obsm_keys`` is unknown, or ``obsm_keys`` contains duplicates; if two
        matrices read have the same class label; if a matrix read is not a CSR
        matrix with one row per observation, or holds values that are not
        counts; if ``X`` or the layer has no columns, as every class needs a
        feature; if ``feature_matrices_key`` is not a non-empty string, the
        registration of an ``.obsm`` matrix read is invalid, or an ``.obsm``
        matrix read has no names or two sources of names; if ``feature_names``
        names a key that is not read, or its names are not unique strings as
        many as the matrix's columns; or if ``region`` is empty, contains
        duplicates or unknown names.

    Notes
    -----
    **Population.** Every row of the selected regions is an instance of the
    summary, zeros included, whichever matrices are read: a row without points
    in any class read has 0 points and 0 detected features, and a missing
    share. :func:`harpy.qc.summarize_points` keeps only the bins with points
    instead, as a bin grid also covers empty space, while a row of the table is
    an instance. To summarize part of the rows, subset the table first. The
    counts describe only the table's rows: the function does not reconstruct
    instances absent from the table, nor include points that were not assigned
    to an instance.

    **Inputs.** Every matrix read must be a CSR matrix: in memory, a SciPy
    ``csr_matrix`` or ``csr_array``; lazy, a Dask array with CSR blocks, as
    :func:`harpy.io.read_zarr` reads tables by default; or backed, an AnnData
    CSR dataset, as :func:`harpy.tb.aggregate_points` attaches its table. Its
    values must be counts, non-negative whole numbers. Each matrix is read once,
    in row blocks, so that memory stays bounded: a lazy table runs one Dask
    compute for all its matrices. The table is never loaded as a whole or
    copied. A detected feature is a value greater than zero, so explicitly
    stored zeros do not count. Besides the matrices, the function reads only the
    table's ``.obs`` and SpatialData annotation: no feature panel, no record of
    the function that created the table, and it does not run
    :func:`harpy.tb.validate_table`.

    Examples
    --------
    From a class-aware aggregation to the instance plots:

    .. code-block:: python

        sdata = hp.tb.aggregate_points(
            sdata,
            labels_name="labels",
            points_name="transcripts",
            output_table_name="table",
            to_coordinate_system="global",
            expression_class="Endogenous",
        )
        summary = hp.qc.summarize_table(sdata.tables["table"], obsm_keys=["negative", "system_control"])
        summary.per_class  # one row per class: x, negative and system_control

        # Each instance's share of its points in X, the expression class.
        hp.qc.instance_histogram(summary, feature_class="x", measure="pct_points")
        # Background-dominated instances: a high share of negative-probe points (y) at few gene points (x).
        hp.qc.instance_scatter(summary, x=("x", "points"), y=("negative", "pct_points"))

        # The raw counts of a normalized table, in one region.
        summary = hp.qc.summarize_table(sdata.tables["table"], layer="counts", region="sample_a_labels")
    """
    attrs = _spatialdata_annotation(adata, function="summarize_table")
    if not isinstance(feature_matrices_key, str) or not feature_matrices_key:
        raise ValueError(
            f"Parameter 'feature_matrices_key' must be a non-empty string, found {feature_matrices_key!r}."
        )
    selected_regions = _selected_names(
        region, recorded=_names(attrs[TableModel.REGION_KEY]), parameter="region", kind="regions"
    )
    keys = _obsm_keys(obsm_keys)
    classes = _class_matrices(
        adata,
        layer=layer,
        obsm_keys=keys,
        feature_names=feature_names,
        feature_matrices_key=feature_matrices_key,
    )
    labels = tuple(entry.label for entry in classes)
    n_features = {entry.label: int(entry.matrix.shape[1]) for entry in classes}

    region_key = attrs[TableModel.REGION_KEY_KEY]
    instance_key = attrs[TableModel.INSTANCE_KEY]
    obs = adata.obs
    # Select on .obs only: the AnnData itself is never copied. A row's region code
    # is the position of its region in selected_regions, -1 outside the selection.
    region_codes = pd.Categorical(obs[region_key].astype(str).to_numpy(), categories=selected_regions).codes
    region_codes = region_codes.astype(np.int64)
    positions = np.flatnonzero(region_codes >= 0)
    rows, columns = _count_rows(classes, region_codes=region_codes, n_regions=len(selected_regions))
    # points[label] holds the class's points per selected row, in table order: the
    # row's sum in that class's matrix. detected[label] holds its detected features:
    # the row's positive entries. Rows without points are included. columns[label]
    # holds the column sums of that matrix per selected region.
    points = {label: row_sums.sums for label, row_sums in rows.items()}
    detected = {label: row_sums.positives for label, row_sums in rows.items()}

    per_instance = obs.iloc[positions][[region_key, instance_key]].copy()
    for label in labels:
        per_instance[_N_CLASS_POINTS_TEMPLATE.format(label=label)] = points[label]
    for label in labels:
        per_instance[_N_CLASS_DETECTED_FEATURES_TEMPLATE.format(label=label)] = detected[label]
    if len(labels) > 1:
        totals = np.zeros(len(positions), dtype=np.uint64)
        for label in labels:
            totals += points[label]
        for label in labels:
            # 0 / 0 has no value: the share of a row without points stays missing, without a warning.
            share = np.full(len(positions), np.nan)
            np.divide(100 * points[label], totals, out=share, where=totals > 0)
            per_instance[_PCT_CLASS_POINTS_TEMPLATE.format(label=label)] = share

    per_class = _class_statistics(points, detected, labels=labels, n_features=n_features)
    region_frames = []
    # The region code of each selected row, aligned with the vectors in points and detected.
    codes_of_selected = region_codes[positions]
    for code, name in enumerate(selected_regions):
        in_region = codes_of_selected == code
        frame = _class_statistics(
            {label: values[in_region] for label, values in points.items()},
            {label: values[in_region] for label, values in detected.items()},
            labels=labels,
            n_features=n_features,
        )
        frame.insert(0, _REGION_COLUMN_KEY, name)
        region_frames.append(frame)
    per_region_class = _with_region_categories(pd.concat(region_frames, ignore_index=True), selected_regions)

    names = {entry.label: entry.feature_names for entry in classes}
    per_feature = _feature_statistics(
        columns,
        n_instances=len(positions),
        region=None,
        labels=labels,
        feature_names=names,
    )
    n_instances_by_region = np.bincount(codes_of_selected, minlength=len(selected_regions))
    region_feature_frames = []
    for code, name in enumerate(selected_regions):
        frame = _feature_statistics(
            columns,
            n_instances=int(n_instances_by_region[code]),
            region=code,
            labels=labels,
            feature_names=names,
        )
        frame.insert(0, _REGION_COLUMN_KEY, name)
        region_feature_frames.append(frame)
    per_region_feature = _with_region_categories(pd.concat(region_feature_frames, ignore_index=True), selected_regions)

    metadata = TableSummaryMetadata(
        region=selected_regions,
        layer=layer,
        obsm_keys=keys,
        feature_matrices_key=feature_matrices_key,
    )
    return TableSummary(
        metadata=metadata,
        per_instance=per_instance,
        per_class=per_class,
        per_region_class=per_region_class,
        per_feature=per_feature,
        per_region_feature=per_region_feature,
    )


def _with_region_categories(frame: pd.DataFrame, regions: tuple[str, ...]) -> pd.DataFrame:
    """Make the region column categorical, its categories the selected regions in their order."""
    frame[_REGION_COLUMN_KEY] = pd.Categorical(frame[_REGION_COLUMN_KEY], categories=regions)
    return frame


def _class_matrices(
    adata: AnnData,
    *,
    layer: str | None,
    obsm_keys: tuple[str, ...],
    feature_names: Mapping[str, Sequence[str]] | None,
    feature_matrices_key: str,
) -> tuple[_FeatureClassMatrix, ...]:
    """Return the matrices to read, one per class, with their feature names, after checking them without reading a value."""
    entries = _collect_class_matrices(adata, layer=layer, obsm_keys=obsm_keys)
    obsm_names = _obsm_feature_names(
        adata,
        obsm_keys=obsm_keys,
        feature_names=feature_names,
        feature_matrices_key=feature_matrices_key,
    )
    var_names = tuple(str(name) for name in adata.var_names)
    return tuple(
        replace(entry, feature_names=obsm_names[entry.key] if entry.attribute == "obsm" else var_names)
        for entry in entries
    )


def _collect_class_matrices(
    adata: AnnData, *, layer: str | None, obsm_keys: tuple[str, ...]
) -> tuple[_FeatureClassMatrix, ...]:
    """Return the matrices to read, one per class, without their names, after checking them without reading a value.

    ``X`` or the layer, then the ``.obsm`` matrices, each a CSR matrix with one
    row per observation, at least one column, and a class label of its own.
    """
    entries = []
    if layer is not None:
        if not isinstance(layer, str) or layer not in adata.layers:
            raise ValueError(f"Layer {layer!r} is not a layer of the table; available layers: {sorted(adata.layers)}.")
        entries.append(_FeatureClassMatrix(attribute="layers", key=layer, matrix=adata.layers[layer]))
    elif adata.X is not None:
        entries.append(_FeatureClassMatrix(attribute="X", key=None, matrix=adata.X))
    for key in obsm_keys:
        if key not in adata.obsm:
            raise ValueError(f"{key!r} is not an .obsm key of the table; available keys: {sorted(adata.obsm)}.")
        entries.append(_FeatureClassMatrix(attribute="obsm", key=key, matrix=adata.obsm[key]))
    if not entries:
        raise ValueError("The table has no X, and neither layer nor obsm_keys names a matrix to summarize.")

    locations_by_label: dict[str, list[str]] = {}
    for entry in entries:
        locations_by_label.setdefault(entry.label, []).append(f"adata.{entry.location}")
    coinciding = {label: locations for label, locations in locations_by_label.items() if len(locations) > 1}
    if coinciding:
        raise ValueError(f"Each matrix read needs its own class label, but these labels coincide: {coinciding}.")
    for entry in entries:
        _check_csr(entry, n_obs=adata.n_obs)
        # Every class read needs at least one feature, so that its points per panel feature, its points divided
        # by its number of features, are defined.
        # An .obsm matrix always has one, as its names must be non-empty, so only X or a layer can fail this
        # check: they have no columns when var is empty, for example after filtering away every gene.
        if entry.matrix.shape[1] == 0:
            raise ValueError(
                f"adata.{entry.location} has no columns, so its class would have no features: every matrix read "
                "needs at least one. Pass a layer of counts with columns as layer=."
            )
    return tuple(entries)


def _obsm_feature_names(
    adata: AnnData,
    *,
    obsm_keys: tuple[str, ...],
    feature_names: Mapping[str, Sequence[str]] | None,
    feature_matrices_key: str,
) -> dict[str, tuple[str, ...]]:
    """Return the column names of each ``.obsm`` matrix read, from one source each.

    A matrix registered in ``adata.uns[feature_matrices_key]`` takes its
    ``feature_columns``, checked as :func:`harpy.tb.validate_table` checks a
    registration; an unregistered one takes the names passed in
    ``feature_names``. Only the registrations of the matrices read are checked.
    """
    passed = {} if feature_names is None else feature_names
    if not isinstance(passed, Mapping):
        raise ValueError("feature_names must map .obsm keys of obsm_keys to sequences of column names.")
    unread = [key for key in passed if key not in obsm_keys]
    if unread:
        raise ValueError(f"feature_names names .obsm keys that are not read: {unread}; it applies to obsm_keys only.")
    registry = adata.uns.get(feature_matrices_key)
    records = {} if registry is None else _require_mapping(registry, path=feature_matrices_key)

    names = {}
    for key in obsm_keys:
        location = f"adata.obsm[{key!r}]"
        if key in records:
            if key in passed:
                raise ValueError(
                    f"{location} is registered in uns[{feature_matrices_key!r}], which names its columns; "
                    "do not pass its names in feature_names."
                )
            names[key] = _validate_feature_matrix_record(
                adata, key, records[key], feature_matrices_key=feature_matrices_key
            )
        elif key in passed:
            given = _metadata_string_sequence(passed[key], path=f"feature_names[{key!r}]", subject="Parameter")
            n_columns = adata.obsm[key].shape[1]
            if len(given) != n_columns:
                raise ValueError(
                    f"feature_names[{key!r}] has {len(given)} names, but {location} has {n_columns} columns."
                )
            names[key] = given
        else:
            raise ValueError(
                f"{location} has no column names: it is not registered in uns[{feature_matrices_key!r}]. Pass its "
                f"names with feature_names={{{key!r}: [...]}}, or the key of its registry with feature_matrices_key."
            )
    return names


def _obsm_keys(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return ``obsm_keys`` as a tuple of distinct keys."""
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or any(not isinstance(key, str) for key in value):
        raise ValueError("obsm_keys must be an .obsm key or a sequence of .obsm keys.")
    keys = tuple(value)
    if len(set(keys)) != len(keys):
        raise ValueError("obsm_keys must not contain duplicate keys.")
    return keys


def _check_csr(entry: _FeatureClassMatrix, *, n_obs: int) -> None:
    """Raise unless the matrix is a CSR matrix with one row per observation, from its type alone."""
    matrix = entry.matrix
    if isinstance(matrix, da.Array):
        meta = matrix._meta
        is_csr = sparse.issparse(meta) and meta.format == "csr"
        kind = f"a Dask array of {type(meta).__name__} blocks"
    else:
        is_csr = isinstance(matrix, CSRDataset) or (sparse.issparse(matrix) and matrix.format == "csr")
        kind = f"a {type(matrix).__name__}"
    if not is_csr:
        raise ValueError(
            f"adata.{entry.location} is {kind}, not a CSR matrix: summarize_table reads CSR "
            "count matrices only, in memory, lazy or backed. Convert it first, for example "
            f"adata.{entry.location} = scipy.sparse.csr_matrix(adata.{entry.location}), or read the store with "
            "hp.io.read_zarr."
        )
    if len(matrix.shape) != 2 or matrix.shape[0] != n_obs:
        raise ValueError(f"adata.{entry.location} must have shape (n_obs, n_features), found {matrix.shape}.")


@dataclass(frozen=True)
class _RowSums:
    """Per selected row of one class matrix, in table order: its sum and its positive entries.

    ``sums`` are the row sums and ``positives`` the positive entries per row.
    The row sums of consecutive row blocks join into those of the matrix.
    """

    sums: NDArray[np.uint64]
    positives: NDArray[np.int64]

    @classmethod
    def concatenate(cls, parts: Sequence[_RowSums]) -> _RowSums:
        """Join the row sums of consecutive blocks, in block order; empty without blocks."""
        return cls(
            sums=np.concatenate([np.zeros(0, dtype=np.uint64), *(part.sums for part in parts)]),
            positives=np.concatenate([np.zeros(0, dtype=np.int64), *(part.positives for part in parts)]),
        )


@dataclass(frozen=True)
class _ColumnSums:
    """Per selected region and column of one class matrix, over the selected rows.

    ``sums`` are the column sums, ``positives`` the rows with a positive entry
    and ``squares`` the sums of squares, each of shape ``(n_regions, n_columns)``.
    Partial sums of row blocks add up, whatever the blocks.
    """

    sums: NDArray[np.float64]
    positives: NDArray[np.int64]
    squares: NDArray[np.float64]

    @classmethod
    def zeros(cls, n_regions: int, n_columns: int) -> _ColumnSums:
        shape = (n_regions, n_columns)
        return cls(sums=np.zeros(shape), positives=np.zeros(shape, dtype=np.int64), squares=np.zeros(shape))

    def __add__(self, other: _ColumnSums) -> _ColumnSums:
        return _ColumnSums(
            sums=self.sums + other.sums,
            positives=self.positives + other.positives,
            squares=self.squares + other.squares,
        )


def _add_column_sums(*parts: _ColumnSums) -> _ColumnSums:
    """Add partial column sums: the merge of the tree over a lazy matrix's blocks."""
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def _count_rows(
    classes: tuple[_FeatureClassMatrix, ...], *, region_codes: NDArray[np.int64], n_regions: int
) -> tuple[dict[str, _RowSums], dict[str, _ColumnSums]]:
    """Return, per class, the row sums of the selected rows and the column sums per region.

    ``region_codes`` gives each table row's selected region, ``-1`` outside the
    selection. Each matrix is read once, in CSR row blocks, and only the blocks
    with selected rows. No quantity depends on another matrix, so each lazy
    matrix keeps its own row chunks, and all lazy matrices run in one Dask
    compute; their column sums are added in a merge tree within it. Backed and
    in-memory matrices are read in row slices, their column sums added as they
    are read.
    """
    block_rows: dict[str, list[_RowSums]] = {}
    columns: dict[str, _ColumnSums] = {}
    lazy_tasks = {}
    for entry in classes:
        if isinstance(entry.matrix, da.Array):
            lazy_tasks[entry.label] = _lazy_block_tasks(
                entry.matrix, region_codes, n_regions=n_regions, location=entry.location
            )
            continue
        block_rows[entry.label] = []
        total = _ColumnSums.zeros(n_regions, entry.matrix.shape[1])
        # A block that is not counts raises, so the remaining blocks are not read.
        for block, codes_in_block in _sliced_blocks(entry.matrix, region_codes):
            row_sums, column_sums = _block_counts(block, codes_in_block, n_regions=n_regions, location=entry.location)
            block_rows[entry.label].append(row_sums)
            total = total + column_sums
        columns[entry.label] = total
    if lazy_tasks:
        # A task that raises stops the compute and re-raises its error here.
        (computed,) = dask.compute(lazy_tasks)
        for label, (parts, column_sums) in computed.items():
            block_rows[label] = parts
            columns[label] = column_sums

    rows = {entry.label: _RowSums.concatenate(block_rows[entry.label]) for entry in classes}
    return rows, columns


def _lazy_block_tasks(
    matrix: da.Array, region_codes: NDArray[np.int64], *, n_regions: int, location: str
) -> tuple[list, object]:
    """Return the delayed row results of a lazy CSR matrix's blocks with selected rows, and its delayed column sums."""
    if len(matrix.chunks[1]) > 1:
        # Harpy's lazy reader splits a CSR matrix by rows only: every block keeps
        # all columns (chunks=(rows, -1) in harpy._storage._anndata's
        # _decode_anndata_element). Only a Dask array built otherwise, for example
        # da.from_array(matrix, chunks=(rows, columns)), is also split by columns.
        # A block must hold whole rows to give their sums, so merge the column
        # pieces of each row range into one block.
        matrix = matrix.rechunk({1: -1})
    tasks = []
    start = 0
    for block, length in zip(matrix.to_delayed()[:, 0], matrix.chunks[0], strict=True):
        codes_in_block = region_codes[start : start + length]
        start += length
        if (codes_in_block >= 0).any():
            tasks.append(dask.delayed(_block_counts)(block, codes_in_block, n_regions=n_regions, location=location))
    block_rows = [task[0] for task in tasks]
    if not tasks:
        return block_rows, _ColumnSums.zeros(n_regions, matrix.shape[1])
    # A merge tree adds the blocks' column sums, so that they are not all held at once.
    return block_rows, _tree_reduce([task[1] for task in tasks], _add_column_sums)


def _sliced_blocks(matrix: object, row_codes: NDArray[np.int64]) -> Iterator[tuple[object, NDArray[np.int64]]]:
    """Yield the matrix in row blocks with their row codes, skipping the blocks whose codes are all -1.

    A yielded block holds all rows of its range, also those with code -1: the
    caller keeps the rows with ``codes_in_block >= 0``. The rows per block follow
    the policy of Harpy's lazy reader, about ``array.chunk-size`` bytes per block,
    from the number of stored values.

    Parameters
    ----------
    matrix
        A backed CSR dataset or an in-memory CSR matrix, one row per table row.
    row_codes
        One code per row: ``-1`` for a row the caller does not need, otherwise a
        non-negative code whose meaning is the caller's, for example the row's
        region among the selected regions, or its row in the results.

    Yields
    ------
    block
        The rows of one range, as an in-memory CSR matrix with all columns.
    codes_in_block
        Their codes, the matching piece of ``row_codes``.
    """
    if isinstance(matrix, CSRDataset):
        data, indices, indptr = (matrix.group[name] for name in ("data", "indices", "indptr"))
        nnz = data.shape[0]
    else:
        data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
        nnz = matrix.nnz
    n_rows = matrix.shape[0]
    # Rows per block of about array.chunk-size bytes, to bound the memory of reading
    # a backed matrix. An in-memory matrix is already loaded whole, so it gains little.
    length = _auto_sparse_block_length(
        n_rows,
        int(nnz),
        entry_bytes=data.dtype.itemsize + indices.dtype.itemsize,
        indptr_itemsize=indptr.dtype.itemsize,
    )
    for start in range(0, n_rows, length):
        codes_in_block = row_codes[start : start + length]
        if (codes_in_block >= 0).any():
            yield matrix[start : start + length], codes_in_block


def _block_counts(
    block: object, codes_in_block: NDArray[np.int64], *, n_regions: int, location: str
) -> tuple[_RowSums, _ColumnSums]:
    """Return the row sums of a CSR row block's selected rows, and its column sums per region.

    ``codes_in_block`` gives each row's selected region, ``-1`` for a row
    outside the selection. The caller reads the row sums as a class's points
    per instance, and the positive entries as its detected features; the
    column sums, per region, give the statistics per feature. ``location``
    names the matrix in the error raised when a stored value is not a
    non-negative whole number.
    """
    selected_in_block = codes_in_block >= 0
    if not selected_in_block.all():
        block = block[selected_in_block]
    codes = codes_in_block[selected_in_block]
    if not block.has_canonical_format:
        # An edge case: aggregate_points writes canonical CSR matrices, as SciPy's
        # COO-to-CSR conversion merges duplicates and sorts the indices, so this
        # covers matrices built otherwise. A non-canonical CSR matrix may store one
        # entry of a row more than once. SciPy sums such duplicates, but counting
        # stored values would count that entry twice as positive; merge them
        # first, which also sorts the indices.
        block = block.copy()
        block.sum_duplicates()
    data = block.data
    if not _are_counts(data):
        raise ValueError(
            f"adata.{location} holds values that are not counts, non-negative whole numbers: "
            "summarize_table summarizes raw counts. If the matrix was normalized, pass the layer of raw counts "
            "with layer=."
        )
    # Per-row results from cumulative sums over the stored values, split at the row pointers.
    starts, stops = block.indptr[:-1], block.indptr[1:]
    # Start from typed zeros: an int64 zero would promote the uint64 sums to float64.
    sums = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(data.astype(np.uint64, copy=False))))
    # A stored zero is not a positive entry.
    positives = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(data > 0, dtype=np.int64)))
    row_sums = _RowSums(sums=sums[stops] - sums[starts], positives=positives[stops] - positives[starts])
    return row_sums, _column_sums_per_region(block, codes, n_regions=n_regions)


def _column_sums_per_region(block: object, codes: NDArray[np.int64], *, n_regions: int) -> _ColumnSums:
    """Return a CSR row block's column sums per region.

    ``codes`` gives the region of each row of the block. An indicator matrix,
    one row per region with a 1 for each of its rows, adds each region's rows
    column by column: ``indicator @ block``. For example, with rows 0 and 1 in
    region 0 and row 2 in region 1, where ``0*`` is a stored zero::

        indicator        block            indicator @ block
        [1, 1, 0]        [2, 0, 1]        [[2, 3, 1],
        [0, 0, 1]        [0, 3, 0]         [1, 0, 0]]
                         [1, 0, 0*]

    ``positives`` multiplies with ``block > 0`` instead, so a stored zero is
    not counted, and ``squares`` with the squared values.
    """
    n_rows = block.shape[0]
    indicator = sparse.csr_matrix((np.ones(n_rows), (codes, np.arange(n_rows))), shape=(n_regions, n_rows))
    # A csr_matrix in float64 for both csr_matrix and csr_array blocks, so that the
    # products have one type, and squares of large counts do not overflow.
    values = sparse.csr_matrix(block, dtype=np.float64)
    return _ColumnSums(
        sums=(indicator @ values).toarray(),
        positives=(indicator @ (values > 0)).toarray().astype(np.int64),
        squares=(indicator @ values.multiply(values)).toarray(),
    )


def _are_counts(data: np.ndarray) -> bool:
    """Return whether every stored value is a non-negative whole number."""
    kind = data.dtype.kind
    if kind == "u":
        return True
    if kind == "i":
        return bool((data >= 0).all())
    if kind == "f":
        return bool((np.isfinite(data) & (data >= 0) & (np.floor(data) == data)).all())
    return False


def _class_statistics(
    points: Mapping[str, NDArray[np.uint64]],
    detected: Mapping[str, NDArray[np.int64]],
    *,
    labels: tuple[str, ...],
    n_features: Mapping[str, int],
) -> pd.DataFrame:
    """Return one row of population and statistics per class, in the column order of the bin summary."""
    n_instances = len(points[labels[0]])
    class_rows = []
    for label in labels:
        counts = points[label]
        n_zero = int(np.count_nonzero(counts == 0))
        class_rows.append(
            {
                _FEATURE_CLASS_KEY: label,
                _N_INSTANCES_KEY: n_instances,
                _N_POINTS_KEY: counts.sum(dtype=np.uint64),
                _N_INSTANCES_WITHOUT_CLASS_KEY: n_zero,
                _PCT_INSTANCES_WITHOUT_CLASS_KEY: 100 * n_zero / n_instances if n_instances else np.nan,
                **_points_statistics(counts, unit="instance"),
                **_points_statistics(detected[label], unit="instance", measure="detected_features"),
                _N_FEATURES_KEY: n_features[label],
            }
        )
    per_class = pd.DataFrame(class_rows)
    per_class[_N_POINTS_KEY] = per_class[_N_POINTS_KEY].astype(np.uint64)
    per_class[_FEATURE_CLASS_KEY] = pd.Categorical(per_class[_FEATURE_CLASS_KEY], categories=labels)
    return per_class


def _feature_statistics(
    columns: Mapping[str, _ColumnSums],
    *,
    n_instances: int,
    region: int | None,
    labels: tuple[str, ...],
    feature_names: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Return one row per feature of each class, from its column sums, in the column order of the points side.

    ``region`` selects one region's sums, or ``None`` pools the regions;
    ``n_instances`` is the matching number of table rows. The columns follow
    ``FeatureSpatialBinSummary.per_feature``, instances in place of bins, with
    ``within_class_fraction`` of ``PointsSummary.per_feature`` after
    ``n_points``.
    """
    frames = []
    for label in labels:
        part = columns[label]
        if region is None:
            sums, positives, squares = part.sums.sum(axis=0), part.positives.sum(axis=0), part.squares.sum(axis=0)
        else:
            sums, positives, squares = part.sums[region], part.positives[region], part.squares[region]
        without = n_instances - positives
        missing = np.full(len(sums), np.nan)
        class_points = sums.sum()
        frames.append(
            pd.DataFrame(
                {
                    _FEATURE_KEY: list(feature_names[label]),
                    _FEATURE_CLASS_KEY: label,
                    _N_INSTANCES_KEY: n_instances,
                    # The sums of whole numbers are exact in float64 up to 2**53.
                    _N_POINTS_KEY: np.rint(sums).astype(np.uint64),
                    _WITHIN_CLASS_FRACTION_KEY: sums / class_points if class_points else missing,
                    _N_INSTANCES_WITHOUT_FEATURE_KEY: without.astype(np.int64),
                    _PCT_INSTANCES_WITHOUT_FEATURE_KEY: 100 * without / n_instances if n_instances else missing,
                    _statistic_name("mean", unit="instance"): sums / n_instances if n_instances else missing,
                    _statistic_name("std", unit="instance"): _sample_std(sums, squares, n_instances),
                }
            )
        )
    per_feature = pd.concat(frames, ignore_index=True)
    per_feature[_FEATURE_CLASS_KEY] = pd.Categorical(per_feature[_FEATURE_CLASS_KEY], categories=labels)
    return per_feature


def _sample_std(sums: NDArray[np.float64], squares: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Return the sample SD, ``ddof=1``, of ``n`` values from their sums and sums of squares; NaN below two values."""
    if n < 2:
        return np.full(len(sums), np.nan)
    variance = (squares - sums * sums / n) / (n - 1)
    # Rounding can leave a slightly negative variance for constant values.
    return np.sqrt(np.clip(variance, 0, None))
