"""Instance-level QC summary of the CSR count matrices of a table."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.abc import CSRDataset
from numpy.typing import NDArray
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _auto_sparse_block_length
from harpy.qc._summary_schema import _FEATURE_CLASS_KEY, _N_FEATURES_KEY, _N_POINTS_KEY, _REGION_COLUMN_KEY
from harpy.qc._summary_statistics import _points_statistics
from harpy.qc.table._table_summary_metadata import TableSummaryMetadata
from harpy.utils._names import _snake_case

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
        One :class:`TableSummaryMetadata` record: the table, the selected
        regions and the matrices read.
    per_instance
        One row per retained instance, indexed by the table's ``.obs`` index.
        Wide, one column per class, to stay aligned with ``.obs``. Columns, in
        this order, each measure for every class in the order of
        :attr:`feature_classes`:

        - the region and instance identity, under the table's own
          ``region_key`` and ``instance_key`` names;
        - ``n_<class>_points``: the class's points in the instance, uint64;
        - ``n_<class>_detected_features``: the class's features with at least
          one point in the instance. Spatial bins have no counterpart, as a
          distinct count of features per bin is expensive;
        - ``pct_<class>_points``, with two classes or more: 100 times the
          class's points / the instance's points across the classes read.

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
        selected regions. Columns, in this order: ``n_total_instances`` counts
        the table's rows of the selected regions, ``n_retained_instances`` those with at
        least one point across the classes read, ``n_excluded_instances`` the
        others, and ``pct_excluded_instances`` is 100 times excluded / total;
        these population counts are the same for every class. ``n_points``
        (uint64) is the class's points over the retained instances;
        ``n_retained_instances_without_class`` counts retained instances with
        zero points of the class, and ``pct_retained_instances_without_class``
        is 100 times that count / retained. ``mean_points_per_instance``,
        ``median_…``, ``std_…`` and ``p95_…`` describe the class's points over
        the retained instances, zeros included, and
        ``mean_detected_features_per_instance``, ``median_…``, ``std_…`` and
        ``p95_…`` its detected features. Last, ``n_features`` is the class's
        number of features in the matrices read. SD uses ``ddof=1`` and is NaN
        for fewer than two instances; the percentile uses linear
        interpolation.
    per_region
        The statistics of ``per_class``, computed within each selected region,
        with a categorical ``region`` column first: one row per selected
        region and class. With one selected region, it holds the values of
        ``per_class``.
    """

    metadata: TableSummaryMetadata
    per_instance: pd.DataFrame
    per_class: pd.DataFrame
    per_region: pd.DataFrame

    @property
    def feature_classes(self) -> tuple[str, ...]:
        """Return the class labels in the order of the results, from ``per_class.feature_class``.

        One label per matrix read, its name in snake case: ``"x"`` for ``X``,
        the layer's name, then each ``.obsm`` key.
        """
        return tuple(str(label) for label in self.per_class[_FEATURE_CLASS_KEY])

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
class _ClassMatrix:
    """One matrix read, one class."""

    label: str
    # Where the matrix is in the AnnData, for messages, for example "obsm['negative']".
    location: str
    matrix: object


def summarize_table(
    sdata: SpatialData,
    table_name: str,
    *,
    layer: str | None = None,
    obsm_keys: str | Sequence[str] = (),
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
    region. Nothing is written to the table.

    Parameters
    ----------
    sdata
        SpatialData object containing the table. The elements that the table
        annotates need not be present.
    table_name
        Name of the table in ``sdata.tables``.
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
        ``per_region``, and their context in ``metadata``. Column names follow
        the class labels, the names of the matrices read in snake case: ``x``
        for ``X``. See :class:`TableSummary`.

    Raises
    ------
    ValueError
        If ``table_name`` is not a table of ``sdata`` or has no SpatialData
        annotation; if there is nothing to read; if ``layer`` or a key of
        ``obsm_keys`` is unknown, or ``obsm_keys`` contains duplicates; if two
        matrices read have the same class label; if a matrix read is not a CSR
        matrix with one row per observation, or holds values that are not
        counts; or if ``region`` is empty, contains duplicates or unknown names.

    Notes
    -----
    **Population.** Within the selected regions, an instance is retained when it
    has at least one point across the classes read, as
    :func:`harpy.qc.summarize_points` retains bins. Zeros of an individual class
    within that population are kept, so the population depends on the matrices
    read: ``X`` alone retains the instances with points in ``X``, and class
    matrices add the instances with points of those classes only. The counts
    describe only the table's rows: the function does not reconstruct
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
    .. code-block:: python

        summary = hp.qc.summarize_table(
            sdata, "table_transcriptomics", obsm_keys=["negative", "system_control"]
        )
        summary.per_class  # one row per class: x, negative and system_control

        # The raw counts of a normalized table, in one region.
        summary = hp.qc.summarize_table(
            sdata, "table_transcriptomics", layer="counts", region="sample_a_labels"
        )
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} is not in sdata; available tables: {sorted(sdata.tables)}.")
    adata = sdata.tables[table_name]
    attrs = adata.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Table {table_name!r} has no SpatialData annotation (uns[{TableModel.ATTRS_KEY!r}]).")
    selected_regions = _selected_names(
        region, recorded=_names(attrs[TableModel.REGION_KEY]), parameter="region", kind="regions"
    )
    keys = _obsm_keys(obsm_keys)
    classes = _class_matrices(adata, table_name=table_name, layer=layer, obsm_keys=keys)
    labels = tuple(entry.label for entry in classes)
    n_features = {entry.label: int(entry.matrix.shape[1]) for entry in classes}

    region_key = attrs[TableModel.REGION_KEY_KEY]
    instance_key = attrs[TableModel.INSTANCE_KEY]
    obs = adata.obs
    # Select on .obs only: the AnnData itself is never copied.
    region_values = obs[region_key].astype(str).to_numpy()
    positions = np.flatnonzero(np.isin(region_values, selected_regions))
    # points[label] holds the class's points per selected row, in table order: the
    # row's sum in that class's matrix. detected[label] holds its detected features:
    # the row's positive entries. Rows without points are included.
    points, detected = _count_rows(classes, positions=positions, n_obs=adata.n_obs, table_name=table_name)

    totals = np.zeros(len(positions), dtype=np.uint64)
    for label in labels:
        totals += points[label]
    retained = totals > 0

    per_instance = obs.iloc[positions[retained]][[region_key, instance_key]].copy()
    for label in labels:
        per_instance[f"n_{label}_points"] = points[label][retained]
    for label in labels:
        per_instance[f"n_{label}_detected_features"] = detected[label][retained]
    if len(labels) > 1:
        for label in labels:
            per_instance[f"pct_{label}_points"] = 100 * points[label][retained] / totals[retained]

    per_class = _class_statistics(points, detected, retained, labels=labels, n_features=n_features)
    region_frames = []
    # The region of each selected row, aligned with the vectors in points and detected.
    regions_of_selected = region_values[positions]
    for name in selected_regions:
        in_region = regions_of_selected == name
        frame = _class_statistics(
            {label: values[in_region] for label, values in points.items()},
            {label: values[in_region] for label, values in detected.items()},
            retained[in_region],
            labels=labels,
            n_features=n_features,
        )
        frame.insert(0, _REGION_COLUMN_KEY, name)
        region_frames.append(frame)
    per_region = pd.concat(region_frames, ignore_index=True)
    per_region[_REGION_COLUMN_KEY] = pd.Categorical(per_region[_REGION_COLUMN_KEY], categories=selected_regions)

    metadata = TableSummaryMetadata(
        table_name=table_name,
        region=selected_regions,
        layer=layer,
        obsm_keys=keys,
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


def _class_matrices(
    adata: AnnData, *, table_name: str, layer: str | None, obsm_keys: tuple[str, ...]
) -> tuple[_ClassMatrix, ...]:
    """Return the matrices to read, one per class, after checking them without reading a value."""
    entries = []
    if layer is not None:
        if not isinstance(layer, str) or layer not in adata.layers:
            raise ValueError(
                f"Layer {layer!r} is not a layer of table {table_name!r}; available layers: {sorted(adata.layers)}."
            )
        entries.append(
            _ClassMatrix(label=_snake_case(layer), location=f"layers[{layer!r}]", matrix=adata.layers[layer])
        )
    elif adata.X is not None:
        entries.append(_ClassMatrix(label=_X_LABEL, location="X", matrix=adata.X))
    for key in obsm_keys:
        if key not in adata.obsm:
            raise ValueError(
                f"{key!r} is not an .obsm key of table {table_name!r}; available keys: {sorted(adata.obsm)}."
            )
        entries.append(_ClassMatrix(label=_snake_case(key), location=f"obsm[{key!r}]", matrix=adata.obsm[key]))
    if not entries:
        raise ValueError(f"Table {table_name!r} has no X, and neither layer nor obsm_keys names a matrix to summarize.")

    locations_by_label: dict[str, list[str]] = {}
    for entry in entries:
        locations_by_label.setdefault(entry.label, []).append(f"adata.{entry.location}")
    coinciding = {label: locations for label, locations in locations_by_label.items() if len(locations) > 1}
    if coinciding:
        raise ValueError(f"Each matrix read needs its own class label, but these labels coincide: {coinciding}.")
    for entry in entries:
        _check_csr(entry, n_obs=adata.n_obs, table_name=table_name)
    return tuple(entries)


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


def _check_csr(entry: _ClassMatrix, *, n_obs: int, table_name: str) -> None:
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
            f"adata.{entry.location} of table {table_name!r} is {kind}, not a CSR matrix: summarize_table reads CSR "
            "count matrices only, in memory, lazy or backed. Convert it first, for example "
            f"adata.{entry.location} = scipy.sparse.csr_matrix(adata.{entry.location}), or read the store with "
            "hp.io.read_zarr."
        )
    if len(matrix.shape) != 2 or matrix.shape[0] != n_obs:
        raise ValueError(
            f"adata.{entry.location} of table {table_name!r} must have shape (n_obs, n_features), found {matrix.shape}."
        )


def _count_rows(
    classes: tuple[_ClassMatrix, ...], *, positions: NDArray[np.intp], n_obs: int, table_name: str
) -> tuple[dict[str, NDArray[np.uint64]], dict[str, NDArray[np.int64]]]:
    """Return, per class, the points and detected features of the selected rows, in table order.

    Each matrix is read once, in CSR row blocks, and only the blocks with
    selected rows. No quantity depends on another matrix, so each lazy matrix
    keeps its own row chunks, and all lazy matrices run in one Dask compute.
    Backed and in-memory matrices are read in row slices.
    """
    selected = np.zeros(n_obs, dtype=bool)
    selected[positions] = True
    results: dict[str, list[tuple[NDArray[np.uint64], NDArray[np.int64]]]] = {}
    lazy_tasks = {}
    for entry in classes:
        if isinstance(entry.matrix, da.Array):
            lazy_tasks[entry.label] = _lazy_block_tasks(
                entry.matrix, selected, location=entry.location, table_name=table_name
            )
            continue
        # A block that is not counts raises, so the remaining blocks are not read.
        results[entry.label] = [
            _block_counts(block, selected_in_block, location=entry.location, table_name=table_name)
            for block, selected_in_block in _sliced_blocks(entry.matrix, selected)
        ]
    if lazy_tasks:
        # A task that raises stops the compute and re-raises its error here.
        (computed,) = dask.compute(lazy_tasks)
        results.update(computed)

    points, detected = {}, {}
    for entry in classes:
        blocks = results[entry.label]
        points[entry.label] = np.concatenate([np.zeros(0, dtype=np.uint64), *(block[0] for block in blocks)])
        detected[entry.label] = np.concatenate([np.zeros(0, dtype=np.int64), *(block[1] for block in blocks)])
    return points, detected


def _lazy_block_tasks(matrix: da.Array, selected: NDArray[np.bool_], *, location: str, table_name: str) -> list:
    """Return one delayed block summary per row block of a lazy CSR matrix that has selected rows."""
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
        selected_in_block = selected[start : start + length]
        start += length
        if selected_in_block.any():
            tasks.append(
                dask.delayed(_block_counts)(block, selected_in_block, location=location, table_name=table_name)
            )
    return tasks


def _sliced_blocks(matrix: object, selected: NDArray[np.bool_]) -> Iterator[tuple[object, NDArray[np.bool_]]]:
    """Yield the CSR row blocks of a backed or in-memory matrix that have selected rows, with their selection.

    The rows per block follow the policy of Harpy's lazy reader, about
    ``array.chunk-size`` bytes per block, from the number of stored values.
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
        selected_in_block = selected[start : start + length]
        if selected_in_block.any():
            yield matrix[start : start + length], selected_in_block


def _block_counts(
    block: object, selected_in_block: NDArray[np.bool_], *, location: str, table_name: str
) -> tuple[NDArray[np.uint64], NDArray[np.int64]]:
    """Return the row sums and the positive entries per row of a CSR row block's selected rows.

    The caller reads the row sums as a class's points per instance, and the
    positive entries as its detected features. ``location`` and ``table_name`` name
    the matrix in the error raised when a stored value is not a non-negative
    whole number.
    """
    if not selected_in_block.all():
        block = block[selected_in_block]
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
            f"adata.{location} of table {table_name!r} holds values that are not counts, non-negative whole numbers: "
            "summarize_table summarizes raw counts. If the matrix was normalized, pass the layer of raw counts "
            "with layer=."
        )
    # Per-row results from cumulative sums over the stored values, split at the row pointers.
    starts, stops = block.indptr[:-1], block.indptr[1:]
    # Start from typed zeros: an int64 zero would promote the uint64 sums to float64.
    sums = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(data.astype(np.uint64, copy=False))))
    # A stored zero is not a positive entry.
    positives = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(data > 0, dtype=np.int64)))
    return sums[stops] - sums[starts], positives[stops] - positives[starts]


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
    retained: NDArray[np.bool_],
    *,
    labels: tuple[str, ...],
    n_features: Mapping[str, int],
) -> pd.DataFrame:
    """Return one row of population counts and statistics per class, in the column order of the bin summary."""
    n_total = len(retained)
    n_retained = int(retained.sum())
    population = {
        "n_total_instances": n_total,
        "n_retained_instances": n_retained,
        "n_excluded_instances": n_total - n_retained,
        "pct_excluded_instances": 100 * (n_total - n_retained) / n_total if n_total else np.nan,
    }
    class_rows = []
    for label in labels:
        counts = points[label][retained]
        n_zero = int(np.count_nonzero(counts == 0))
        class_rows.append(
            {
                _FEATURE_CLASS_KEY: label,
                **population,
                _N_POINTS_KEY: counts.sum(dtype=np.uint64),
                "n_retained_instances_without_class": n_zero,
                "pct_retained_instances_without_class": 100 * n_zero / n_retained if n_retained else np.nan,
                **_points_statistics(counts, unit="instance"),
                **_points_statistics(detected[label][retained], unit="instance", measure="detected_features"),
                _N_FEATURES_KEY: n_features[label],
            }
        )
    per_class = pd.DataFrame(class_rows)
    per_class[_N_POINTS_KEY] = per_class[_N_POINTS_KEY].astype(np.uint64)
    per_class[_FEATURE_CLASS_KEY] = pd.Categorical(per_class[_FEATURE_CLASS_KEY], categories=labels)
    return per_class
