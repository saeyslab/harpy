"""QC metrics of a table's count matrices, added in place to its ``.obs`` and ``.var``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from spatialdata.models import TableModel

from harpy.qc._summary_schema import (
    _FEATURE_CLASS_KEY,
    _N_CLASS_POINTS_PER_PANEL_FEATURE_TEMPLATE,
    _N_CLASS_POINTS_TEMPLATE,
    _N_INSTANCES_WITHOUT_FEATURE_KEY,
    _N_POINTS_KEY,
    _PCT_INSTANCES_WITHOUT_FEATURE_KEY,
    _WITHIN_CLASS_FRACTION_KEY,
    _statistic_name,
)
from harpy.qc.table._summarize_table import TableSummary, summarize_table
from harpy.table._utils import _spatialdata_annotation
from harpy.utils._keys import _FEATURE_MATRICES_KEY

# The columns of TableSummary.per_feature written to .var, in this order.
_VAR_COLUMNS = (
    _N_POINTS_KEY,
    _WITHIN_CLASS_FRACTION_KEY,
    _N_INSTANCES_WITHOUT_FEATURE_KEY,
    _PCT_INSTANCES_WITHOUT_FEATURE_KEY,
    _statistic_name("mean", unit="instance"),
    _statistic_name("std", unit="instance"),
)


def calculate_qc_metrics(
    adata: AnnData,
    *,
    layer: str | None = None,
    obsm_keys: str | Sequence[str] = (),
    feature_names: Mapping[str, Sequence[str]] | None = None,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
) -> TableSummary:
    """Add the QC metrics of a table's count matrices, per class, to its ``.obs`` and ``.var``, in place.

    Unlike :func:`scanpy.pp.calculate_qc_metrics`, each matrix read is one
    class, ``X`` or a layer and the ``.obsm`` matrices named, rather than
    boolean ``qc_vars`` within ``X``; the columns have other names, mapped
    below; there is no ``percent_top``, ``log1p`` or ``inplace``; and the
    computed :class:`~harpy.qc.TableSummary` is returned. The metrics are
    those of :func:`harpy.qc.summarize_table`, over every region the table
    annotates. Like :mod:`scanpy.pp`, the function changes ``adata`` in place;
    on a table stored in SpatialData, write the columns back with
    :func:`harpy.tb.io.add_table_updates` and ``overwrite=True``, as ``obs``
    and ``var`` exist in the store.

    Parameters
    ----------
    adata
        The table, such as ``sdata.tables["table"]``, with its SpatialData
        annotation, in memory, lazy or backed: only ``.obs`` and ``.var``
        change. Columns that they already hold under the names to add are
        replaced.
    layer
        The matrix aligned with ``var`` to read, as in
        :func:`harpy.qc.summarize_table`: ``None`` reads ``X``, a name reads
        ``adata.layers[layer]``. After normalization, pass the layer of raw
        counts.
    obsm_keys
        ``.obsm`` matrices to read as well, each one class, for example
        ``["negative", "system_control"]``.
    feature_names
        Column names of an ``.obsm`` matrix read that
        ``adata.uns[feature_matrices_key]`` does not register, as in
        :func:`harpy.qc.summarize_table`.
    feature_matrices_key
        Key in ``adata.uns`` of the registry that names the columns of the
        ``.obsm`` matrices read.

    Returns
    -------
    TableSummary
        The summary computed, for the instance plots and for the statistics
        per class and per region, which are not added to the table.

    Raises
    ------
    TypeError
        If ``adata`` is not an :class:`~anndata.AnnData`.
    ValueError
        As :func:`harpy.qc.summarize_table` raises; or if a row of the table
        has a region that its SpatialData annotation does not list, so that the
        row has no metrics. Nothing is written then.

    Notes
    -----
    **Added to ``.obs``,** for every row, with ``<class>`` the label of a
    matrix read, its name in snake case, ``x`` for ``X``:

    - ``n_<class>_points`` (uint64) and ``n_<class>_detected_features``;
    - ``pct_<class>_points``, with two classes or more: the class's share of
      the row's points across the classes read, missing (NaN) for a row
      without points;
    - ``n_<class>_points_per_panel_feature`` (float64): the class's points
      divided by its number of features in the matrices read,
      ``summary.panel_feature_counts``, the values that the instance plots
      show with ``normalization="per_panel_feature"``.

    Compare rates on aggregates rather than per row: a ratio of two classes'
    rates per row is noisy for sparse classes, such as negative probes, and
    undefined for a row without points. Per region, it is stable:

    .. code-block:: python

        rates = ["n_x_points_per_panel_feature", "n_negative_points_per_panel_feature"]
        per_region = adata.obs.groupby("fov_labels", observed=True)[rates].mean()

    **Added to ``.var``,** when ``X`` or a layer is read, the statistics of
    ``summary.per_feature`` for its features: ``n_points`` (uint64),
    ``within_class_fraction``, ``n_instances_without_feature``,
    ``pct_instances_without_feature``, ``mean_points_per_instance`` and
    ``std_points_per_instance``. The features of the ``.obsm`` matrices have
    no ``.var`` axis, and stay in the summary.

    Compared with scanpy, for ``X``::

        this function                         scanpy
        n_x_points (obs)                      total_counts
        n_x_detected_features (obs)           n_genes_by_counts
        pct_<class>_points (obs)              pct_counts_<qc_var>
        n_points (var)                        total_counts
        mean_points_per_instance (var)        mean_counts
        pct_instances_without_feature (var)   pct_dropout_by_counts

    Examples
    --------
    .. code-block:: python

        adata = sdata.tables["table"]
        summary = hp.tb.pp.calculate_qc_metrics(adata, obsm_keys=["negative", "system_control"])
        hp.tb.pl.scatter(adata, "n_x_points", "pct_negative_points")
        hp.qc.instance_histogram(summary, feature_class="negative", normalization="per_panel_feature")
        hp.tb.io.add_table_updates(sdata, table_name="table", overwrite=True)
    """
    attrs = _spatialdata_annotation(adata, function="calculate_qc_metrics")
    summary = summarize_table(
        adata,
        layer=layer,
        obsm_keys=obsm_keys,
        feature_names=feature_names,
        feature_matrices_key=feature_matrices_key,
    )
    per_instance = summary.per_instance
    region_key = attrs[TableModel.REGION_KEY_KEY]
    # The summary holds the table's rows in table order, from the same table, so the columns are written by
    # position. Rows of a region that the annotation does not list are left out of it, and so have no values.
    if len(per_instance) != adata.n_obs:
        regions = adata.obs[region_key].astype(str)
        unlisted = sorted(set(regions[~regions.isin(summary.metadata.region)]))
        raise ValueError(
            f"{adata.n_obs - len(per_instance)} rows of the table have a region in obs[{region_key!r}] that its "
            f"SpatialData annotation does not list, such as {unlisted[:3]}; every row needs an annotated region "
            "to get QC metrics."
        )
    identity = {region_key, attrs[TableModel.INSTANCE_KEY]}
    # Every column is computed before any is written, so a failure leaves the table unchanged.
    obs_columns: dict[str, NDArray] = {
        name: per_instance[name].to_numpy() for name in per_instance.columns if name not in identity
    }
    for label, n_features in summary.panel_feature_counts.items():
        # n_features is positive: summarize_table raises for a matrix without columns.
        points = per_instance[_N_CLASS_POINTS_TEMPLATE.format(label=label)].to_numpy(dtype=np.float64)
        obs_columns[_N_CLASS_POINTS_PER_PANEL_FEATURE_TEMPLATE.format(label=label)] = points / n_features
    var_columns: dict[str, NDArray] = {}
    # X or the layer, when read, is the first class; the others are the .obsm matrices.
    if len(summary.feature_classes) > len(summary.metadata.obsm_keys):
        per_feature = summary.per_feature
        rows = per_feature[per_feature[_FEATURE_CLASS_KEY] == summary.feature_classes[0]]
        var_columns = {name: rows[name].to_numpy() for name in _VAR_COLUMNS}

    for name, values in obs_columns.items():
        adata.obs[name] = values
    for name, values in var_columns.items():
        adata.var[name] = values
    return summary
