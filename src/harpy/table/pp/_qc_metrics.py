"""QC metrics of a table's count matrices, added in place to its ``.obs`` and ``.var``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from spatialdata.models import TableModel

from harpy.qc._summary_schema import (
    _FEATURE_CLASS_KEY,
    _N_INSTANCES_WITHOUT_FEATURE_KEY,
    _N_POINTS_KEY,
    _N_POINTS_PER_PANEL_FEATURE_KEY,
    _PCT_INSTANCES_WITHOUT_FEATURE_KEY,
    _WITHIN_CLASS_FRACTION_KEY,
    _class_column,
    _statistic_name,
)
from harpy.qc.table._summarize_table import TableSummary, summarize_table
from harpy.table._utils import _spatialdata_annotation
from harpy.utils._keys import _FEATURE_MATRICES_KEY

# The columns of TableSummary.per_feature written to .var, in this order, each followed by the label of the
# class of X or the layer read.
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
        change. Columns are only written, never removed: those that they
        already hold under the names to add are replaced, and those of other
        matrices or classes, from earlier calls, stay. Columns are named by
        class label, so two matrices with the same label, such as ``X`` and a
        layer named ``x``, read in different calls, share their columns: the
        later call replaces them.
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
    Every column specific to a class ends with the class label ``<class>``,
    the name of the matrix read in snake case, ``x`` for ``X``, so that the
    column of a class is ``f"{column}_{label}"`` in ``.obs``, ``.var`` and
    ``summary.per_instance`` alike.

    **Added to ``.obs``,** for every row:

    - ``n_points_<class>`` (uint64) and ``n_detected_features_<class>``;
    - ``pct_points_<class>``, with two classes or more: the class's share of
      the row's points across the classes read in this call, missing (NaN) for
      a row without points;
    - ``n_points_per_panel_feature_<class>`` (float64): the class's points
      divided by its number of features in the matrices read,
      ``summary.panel_feature_counts``, the values that the instance plots
      show with ``normalization="per_panel_feature"``.

    Compare rates on aggregates rather than per row: a ratio of two classes'
    rates per row is noisy for sparse classes, such as negative probes, and
    undefined for a row without points. Per region, it is stable:

    .. code-block:: python

        rates = ["n_points_per_panel_feature_x", "n_points_per_panel_feature_negative"]
        per_region = adata.obs.groupby("fov_labels", observed=True)[rates].mean()

    **Added to ``.var``,** when ``X`` or a layer is read, the statistics of
    ``summary.per_feature`` for its features, labelled with its class:
    ``n_points_<class>`` (uint64), ``within_class_fraction_<class>``,
    ``n_instances_without_feature_<class>``,
    ``pct_instances_without_feature_<class>``,
    ``mean_points_per_instance_<class>`` and ``std_points_per_instance_<class>``.
    The features of the ``.obsm`` matrices have no ``.var`` axis, and stay in
    the summary.

    Compared with scanpy, for ``X``::

        this function                           scanpy
        n_points_x (obs)                        total_counts
        n_detected_features_x (obs)             n_genes_by_counts
        pct_points_<class> (obs)                pct_counts_<qc_var>
        n_points_x (var)                        total_counts
        mean_points_per_instance_x (var)        mean_counts
        pct_instances_without_feature_x (var)   pct_dropout_by_counts

    Examples
    --------
    .. code-block:: python

        adata = sdata.tables["table"]
        summary = hp.tb.pp.calculate_qc_metrics(adata, obsm_keys=["negative", "system_control"])
        hp.tb.pl.scatter(adata, "n_points_x", "pct_points_negative")
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
        points = per_instance[_class_column(_N_POINTS_KEY, label=label)].to_numpy(dtype=np.float64)
        obs_columns[_class_column(_N_POINTS_PER_PANEL_FEATURE_KEY, label=label)] = points / n_features
    var_columns: dict[str, NDArray] = {}
    # The class of the matrix aligned with .var: X, or the layer passed as layer= instead of it; None if only
    # .obsm matrices were read.
    var_class = summary.var_class
    if var_class is not None:
        per_feature = summary.per_feature
        # The statistics of the features of .var, from per_feature's rows of their class, which follow var's order,
        # so that they are written by position.
        statistics = per_feature.loc[per_feature[_FEATURE_CLASS_KEY] == var_class, list(_VAR_COLUMNS)]
        var_columns = {_class_column(name, label=var_class): values.to_numpy() for name, values in statistics.items()}

    for name, values in obs_columns.items():
        adata.obs[name] = values
    for name, values in var_columns.items():
        adata.var[name] = values
    return summary
