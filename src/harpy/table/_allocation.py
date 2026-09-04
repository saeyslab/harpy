"""Point-to-label aggregation orchestration.

An *aggregation pair* is one positional combination of a labels element, a
points element and the coordinate system in which they are joined. For
example::

    pair 0 = (labels_a, points_a, sample_a)
    pair 1 = (labels_b, points_b, sample_b)

Each pair receives a zero-based ordinal. The ordinal remains part of the
checkpoint row key so equal instance IDs from different labels elements do not
collapse into one output row.

The class-aware :func:`aggregate_points` implementation uses the following
private data flow::

    source points (coordinates, feature, class)
            |
            +-- _validate_feature_panel_contents()
            |       `-- _feature_panel_partition_errors()
            |               validate each source feature-to-class assignment
            |
            `-- _assign_points_to_labels()
                    produce lazy (instance, feature) rows
                              |
                              v
                   _local_feature_counts()
                    partition-local reduction
                              |
                              v
                _stage_aggregation_checkpoint()
                    +-- Dask shuffle by (aggregation-pair ordinal, instance ID)
                    +-- _merge_count_partition()
                    +-- write merged counts to temporary Parquet on disk:
                        tables/.harpy-aggregate-<uuid>/merged_counts/
                              |
                              v
                   _write_aggregation_table()
                    +-- _checkpoint_sparse_array(expression axis)
                    |       `-- adata.X
                    +-- _checkpoint_sparse_array(auxiliary axis)
                    |       `-- auxiliary_feature_counts
                    `-- _checkpoint_partition_class_counts()
                            `-- per-class totals in adata.obs
                              |
                              v
                _remove_aggregation_workspace()
                    remove the temporary checkpoint, including after failure

The source class column establishes and validates the feature-to-class mapping;
it is not carried through spatial assignment or stored in the count checkpoint.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import anndata as ad
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from loguru import logger as log
from scipy import sparse
from scipy.sparse import issparse
from spatialdata import SpatialData
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations import Identity
from xarray import DataArray

from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _HARPY_METADATA_KEY,
    _METADATA_VERSION,
    _METADATA_VERSION_KEY,
    _POINTS_METADATA_KEY,
)
from harpy.image._image import _get_spatial_element, _get_translation
from harpy.table._aggregation_checkpoint import (
    _CheckpointPair,
    _local_feature_counts,
    _stage_aggregation_checkpoint,
)
from harpy.table._aggregation_writer import (
    _create_aggregation_workspace,
    _FeatureClassWriteContract,
    _remove_aggregation_workspace,
    _validate_aggregation_destination,
    _write_aggregation_table,
)
from harpy.table._metadata import (
    _AUXILIARY_POINTS_FRACTION_COLUMN,
)
from harpy.table._table import add_table
from harpy.table._utils import _sanity_check_append_region
from harpy.utils._aggregate import RasterAggregator
from harpy.utils._keys import _CELL_INDEX, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from harpy.utils._transformations import _identity_check_transformations_points
from harpy.utils.utils import _make_list

_WARNED_DEPRECATED_ATTRIBUTES: set[str] = set()


@dataclass(frozen=True)
class _AggregationPair:
    labels_name: str
    points_name: str
    coordinate_system: str


@dataclass(frozen=True)
class _FeaturePanelContract:
    """Normalized immutable description of a points feature panel.

    The contract describes source metadata shared by compatible points
    elements. It does not select which feature class contributes to an
    expression matrix and contains no observed point counts.

    Attributes
    ----------
    feature_key
        Points column containing feature identifiers, such as gene names.
    feature_class_key
        Categorical points column assigning each feature to a panel class.
    classes
        Ordered feature classes declared by the panel.
    features_by_class_items
        Immutable ordered ``(feature class, features)`` representation of the
        panel's class-to-features mapping.
    """

    feature_key: str
    feature_class_key: str
    classes: tuple[str, ...]
    features_by_class_items: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def features_by_class(self) -> dict[str, tuple[str, ...]]:
        return dict(self.features_by_class_items)

    @property
    def class_by_feature(self) -> dict[str, str]:
        return {
            feature: feature_class for feature_class, features in self.features_by_class_items for feature in features
        }


@dataclass(frozen=True)
class _FeatureClassAggregationContract:
    """Class-aware aggregation configuration for one compatible feature panel.

    The contract selects one panel class for ``adata.X``. Every remaining
    class is treated as an auxiliary class. Output count-column names,
    auxiliary classes, and per-class feature counts are derived from the panel,
    so they cannot disagree with its metadata. The contract contains no spatial
    assignment results or observed point counts.

    Attributes
    ----------
    panel
        Normalized panel shared by every points element in the aggregation.
    expression_class
        Panel class whose features are retained in ``adata.X``.
    """

    panel: _FeaturePanelContract
    expression_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression_class, str) or not self.expression_class:
            raise ValueError(
                f"Parameter 'expression_class' must be a non-empty string, found {self.expression_class!r}."
            )
        if self.expression_class not in self.panel.classes:
            raise ValueError(
                f"Expression class {self.expression_class!r} is not present in panel classes "
                f"{list(self.panel.classes)!r}."
            )
        if not self.auxiliary_classes:
            raise ValueError(
                "Class-aware aggregation requires at least one non-expression feature class. "
                "Use expression_class=None for a panel containing only the expression class."
            )
        generated = [column for _, column in self.count_columns]
        if len(set(generated)) != len(generated):
            raise ValueError(f"Feature classes produce colliding count-column names: {generated!r}.")

    @property
    def count_columns(self) -> tuple[tuple[str, str], ...]:
        return tuple((feature_class, f"n_{_snake_case(feature_class)}_points") for feature_class in self.panel.classes)

    @property
    def auxiliary_classes(self) -> tuple[str, ...]:
        return tuple(feature_class for feature_class in self.panel.classes if feature_class != self.expression_class)

    @property
    def expression_feature_axis(self) -> tuple[str, ...]:
        """Return the ordered features defining the columns of ``adata.X``.

        The axis contains every feature assigned to ``expression_class`` in
        the authoritative panel's order, including features with no observed
        points. The same ordered values become ``adata.var_names``.
        """
        return self.panel.features_by_class[self.expression_class]

    @property
    def auxiliary_feature_axis(self) -> tuple[str, ...]:
        """Return the ordered features defining the auxiliary matrix columns.

        Features outside ``expression_class`` are concatenated first in the
        panel's class order and then in each class's feature order, including
        features with no observed points. The resulting axis describes
        ``adata.obsm["auxiliary_feature_counts"]`` and is recorded as that
        matrix's ``feature_columns`` metadata.
        """
        features_by_class = self.panel.features_by_class
        return tuple(
            feature for feature_class in self.auxiliary_classes for feature in features_by_class[feature_class]
        )

    @property
    def auxiliary_class_slices(self) -> tuple[tuple[str, slice], ...]:
        """Map each non-expression class to its columns on the auxiliary axis.

        Each slice selects that class's contiguous feature block from
        :attr:`auxiliary_feature_axis`. Because this axis also defines the
        columns of ``adata.obsm["auxiliary_feature_counts"]``, the same slice
        selects the class's columns from the auxiliary matrix for calculating
        per-instance class totals.
        """
        result: list[tuple[str, slice]] = []
        start = 0
        features_by_class = self.panel.features_by_class
        for feature_class in self.auxiliary_classes:
            stop = start + len(features_by_class[feature_class])
            result.append((feature_class, slice(start, stop)))
            start = stop
        return tuple(result)

    @property
    def auxiliary_class_feature_counts(self) -> tuple[tuple[str, int], ...]:
        """Return the panel-defined feature count for each non-expression class.

        Each value is the number of features assigned to that auxiliary class
        in the authoritative panel, including panel features for which the
        selected points elements contain zero detections. The values are
        recorded with the aggregation metadata so downstream QC can normalize
        per-class point counts without inferring panel size from observed data.
        """
        features_by_class = self.panel.features_by_class
        return tuple((feature_class, len(features_by_class[feature_class])) for feature_class in self.auxiliary_classes)


def aggregate_points(
    sdata: SpatialData,
    labels_name: str | list[str],
    points_name: str | list[str] = "transcripts",
    output_table_name: str = "table_transcriptomics",
    to_coordinate_system: str | list[str] = "global",
    chunks: str | tuple[int, ...] | int | None = None,
    feature_key: str = _GENES_KEY,
    expression_class: str | None = None,
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    spatial_key: str = _SPATIAL,
    table_index_name: str | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Aggregate one or more points elements within paired labels elements.

    One aggregation call constructs one complete table. Scalar ``points_name``
    and ``to_coordinate_system`` values are broadcast across ``labels_name``;
    lists are paired positionally. The points must have an identity
    transformation to the selected coordinate system. Labels may have an
    identity transformation, translation, or sequence of translations.

    Each value ``adata.X[i, j]`` is the number of input points carrying feature
    ``j`` that were spatially assigned to instance ``i``. These are point
    counts, not counts of features defined by a panel.

    Aggregation has exactly two mutually exclusive modes, selected solely by
    ``expression_class``:

    - **Ordinary mode** is selected by ``expression_class=None``. Feature-panel
      metadata is ignored even when present. The sorted union of observed
      ``feature_key`` values across the selected points elements defines
      ``adata.X``. No auxiliary feature matrix or per-class summaries are
      created.
    - **Class-aware mode** is selected by a non-empty ``expression_class``.
      Feature-panel metadata is mandatory and validated. Features assigned to
      the selected class define ``adata.X``; features in all other panel classes
      define ``adata.obsm["auxiliary_feature_counts"]`` and the per-class
      summaries in ``adata.obs``.

    In class-aware mode, aggregation resolves each points element's feature
    panel through::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
            -> sdata.attrs["harpy"]["feature_panels"][feature_panel]

    Harpy validates every selected points element against its referenced feature
    panel before spatial assignment. The panel's ``feature_key`` must match the
    requested ``feature_key``; its ``feature_class_key`` column must exist in
    the points, be categorical, and use the panel's ordered classes. Every
    source point must contain a non-null feature and class, its feature must
    occur in the panel, and its class must match that feature's panel
    assignment. All selected points elements must resolve compatible panel
    contracts. Panel features do not need to have observed points. The panel is
    authoritative: the materialized points feature-class column is checked for
    consistency, but matrix axes, feature placement and class summaries are
    derived from the panel rather than from that column.

    The referenced panel supplies ``feature_key``, ``feature_class_key``,
    ``classes``, and ``features_by_class``. Features in ``expression_class``
    define ``adata.X``. Every remaining panel feature is retained in the
    independent sparse matrix ``adata.obsm["auxiliary_feature_counts"]``, whose
    ordered columns are recorded under
    ``adata.uns["feature_matrices"]["auxiliary_feature_counts"]``. Instances
    receiving only non-expression points remain in the table with an all-zero
    expression row.

    In class-aware mode, matrix columns are panel-defined rather than
    observation-derived. Every panel feature is retained even when it is absent
    from all selected points elements. An unobserved expression feature remains
    in ``adata.var_names`` with an all-zero column in ``adata.X``; an unobserved
    auxiliary feature remains in the auxiliary ``feature_columns`` metadata
    with an all-zero auxiliary-matrix column. For example::

        panel                              resulting feature axes
        Endogenous: [GeneA, GeneZero]      adata.var_names:
          observed: GeneA                    [GeneA, GeneZero]
          unobserved: GeneZero               GeneZero column is all zero

        Negative: [Neg1, NegZero]          auxiliary feature_columns:
          observed: Neg1                     [Neg1, NegZero]
          unobserved: NegZero                NegZero column is all zero

    Every panel class is also summarized as ``n_<class>_points`` in
    ``adata.obs``, together with ``auxiliary_points_fraction``. These summaries
    are derived from the persisted expression and auxiliary matrices. Auxiliary
    class feature counts are the lengths of the panel's non-expression
    ``features_by_class`` lists and are recorded in
    ``adata.uns["feature_class_aggregation"]`` for later QC; no per-feature
    rates are persisted in ``adata.obs``.

    Coordinates in ``adata.obsm[spatial_key]`` are geometric centers of mass
    calculated from the paired labels rasters for the retained instance IDs.
    This definition is used in both ordinary and class-aware modes and does not
    depend on point positions or feature classes.

    This operation requires ``sdata`` to be backed by a writable local Zarr
    store. Aggregation remains partitioned and the final sparse matrices are
    written in row blocks without materializing the complete reductions on the
    driver.

    Table construction does not revalidate the payload it has just assembled.
    Call :func:`harpy.tb.validate_table` explicitly to check a table against its
    persisted SpatialData annotation, feature-matrix records, source references,
    and authoritative feature-panel metadata.

    Parameters
    ----------
    sdata
        SpatialData object backed by a writable local Zarr store. Write an
        unbacked object first with ``sdata.write("sdata.zarr")``.
    labels_name
        Labels element or ordered list of labels elements whose instances are
        used to aggregate the points. Duplicate labels names are rejected.
    points_name
        Points element or ordered list of points elements. A scalar is
        broadcast across all labels elements; a list must match their length.
    output_table_name
        Table element in which to store the resulting AnnData object.
    to_coordinate_system
        Coordinate system or ordered list of coordinate systems pairing each
        labels and points element. A scalar is broadcast across all pairs.
    chunks
        Optional labels-array chunk size used during assignment. Rechunking the
        labels element on disk beforehand is generally more efficient.
    feature_key
        Column in each points element containing feature identifiers, such as
        gene names. In class-aware mode it must equal the panel's
        ``feature_key``. It must differ from ``instance_key``.
    expression_class
        Selects the aggregation mode. ``None`` selects ordinary mode, which
        ignores feature-panel metadata and retains all observed features in
        ``adata.X``. A non-empty string selects class-aware mode and names the
        panel class retained in ``adata.X``; all other panel classes are
        retained in the auxiliary feature matrix.
    instance_key
        Column in ``adata.obs`` holding instance identifiers. It must differ
        from ``feature_key`` and, in class-aware mode, from the panel's
        ``feature_class_key``.
    region_key
        Categorical column in ``adata.obs`` holding labels-element names.
    spatial_key
        Key in ``adata.obsm`` holding segmentation-label centers of mass in the
        selected coordinate systems.
    table_index_name
        Name of the resulting ``adata.obs`` index. If ``None``, defaults to
        ``f"{instance_key}_index"``. It must not collide with an ``adata.obs``
        column produced by aggregation.
    overwrite
        Whether an existing ``output_table_name`` may be replaced.

    Returns
    -------
    The updated SpatialData object with one AnnData table at
    ``sdata.tables[output_table_name]``.

    Example
    --------
    .. code-block:: python

        sdata = hp.datasets.resolve_example_multiple_coordinate_systems()

        sdata = hp.tb.aggregate_points(
            sdata,
            labels_name=["labels_a1_1", "labels_a1_2"],
            points_name=["points_a1_1", "points_a1_2"],
            output_table_name="my_table",
            to_coordinate_system=["a1_1", "a1_2"],
            overwrite=True,
        )
    """
    destination = _validate_aggregation_destination(
        sdata,
        output_table_name=output_table_name,
        overwrite=overwrite,
    )
    _validate_aggregation_column_keys(feature_key=feature_key, instance_key=instance_key)
    table_index_name = _resolve_table_index_name(
        table_index_name,
        instance_key=instance_key,
        region_key=region_key,
    )
    pairs = _normalize_aggregation_pairs(
        sdata,
        labels_name=labels_name,
        points_name=points_name,
        to_coordinate_system=to_coordinate_system,
        feature_key=feature_key,
    )
    if expression_class is None:
        contract = None
    else:
        contract = _resolve_feature_class_contract(
            sdata,
            pairs=pairs,
            feature_key=feature_key,
            expression_class=expression_class,
            region_key=region_key,
            instance_key=instance_key,
            table_index_name=table_index_name,
        )
        _validate_feature_panel_contents(sdata, pairs=pairs, panel=contract.panel)

    checkpoint_pairs = tuple(
        _CheckpointPair(
            ordinal=ordinal,
            labels_name=pair.labels_name,
            points_name=pair.points_name,
            coordinate_system=pair.coordinate_system,
            coordinate_columns=("x", "y", "z") if "z" in sdata.points[pair.points_name].columns else ("x", "y"),
        )
        for ordinal, pair in enumerate(pairs)
    )
    workspace = _create_aggregation_workspace(destination)
    try:
        partial_counts: list[DaskDataFrame] = []
        for pair_ordinal, pair in enumerate(pairs):
            points = sdata.points[pair.points_name]
            # Lazily map every point to the nonzero label at its rounded pixel.
            # The result contains one (feature, instance) row per assigned
            # point. The source class column is validated separately against
            # the panel; downstream class totals are derived from the panel's
            # feature-to-class mapping. Coordinates, outside points and
            # background assignments are omitted; aggregation into compact
            # (instance, feature, count) rows happens below.
            assigned_points = _assign_points_to_labels(
                se=_get_spatial_element(sdata, element_name=pair.labels_name),
                ddf=points,
                value_key=feature_key,
                drop_coordinates=True,
                to_coordinate_system=pair.coordinate_system,
                chunks=chunks,
                instance_key=instance_key,
            )
            partial_counts.append(
                _local_feature_counts(
                    assigned_points,
                    pair_ordinal=pair_ordinal,
                    instance_key=instance_key,
                    feature_key=feature_key,
                )
            )

        checkpoint, observed_feature_axis = _stage_aggregation_checkpoint(
            partial_counts,
            path=workspace / "merged_counts",
            pairs=checkpoint_pairs,
            discover_observed_features=contract is None,
        )
        centers_by_pair = {
            pair.ordinal: _label_centers(
                sdata,
                pair=pairs[pair.ordinal],
                instance_ids=checkpoint.instance_ids(pair.ordinal),
                coordinate_columns=pair.coordinate_columns,
                instance_key=instance_key,
            )
            for pair in checkpoint.pairs
        }
        class_write_contract = None
        if contract is None:
            if observed_feature_axis is None:
                raise RuntimeError("Ordinary aggregation did not produce an observed feature axis.")
            expression_axis = observed_feature_axis
        else:
            expression_axis = contract.expression_feature_axis
            class_write_contract = _FeatureClassWriteContract(
                feature_key=contract.panel.feature_key,
                feature_class_key=contract.panel.feature_class_key,
                classes=contract.panel.classes,
                expression_class=contract.expression_class,
                features_by_class_items=contract.panel.features_by_class_items,
                count_columns=contract.count_columns,
            )
        return _write_aggregation_table(
            sdata,
            destination=destination,
            workspace=workspace,
            checkpoint=checkpoint,
            expression_axis=expression_axis,
            centers_by_pair=centers_by_pair,
            output_table_name=output_table_name,
            feature_key=feature_key,
            region_key=region_key,
            instance_key=instance_key,
            spatial_key=spatial_key,
            table_index_name=table_index_name,
            class_contract=class_write_contract,
        )
    finally:
        _remove_aggregation_workspace(workspace)


def __getattr__(name: str) -> object:
    if name == "allocate":
        if name not in _WARNED_DEPRECATED_ATTRIBUTES:
            _WARNED_DEPRECATED_ATTRIBUTES.add(name)
            log.warning("`harpy.tb.allocate` is deprecated. Import and use `harpy.tb.aggregate_points` instead.")
        return aggregate_points
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_aggregation_pairs(
    sdata: SpatialData,
    *,
    labels_name: str | list[str],
    points_name: str | list[str],
    to_coordinate_system: str | list[str],
    feature_key: str,
) -> tuple[_AggregationPair, ...]:
    """Normalize scalar/list inputs into validated positional aggregation pairs."""
    labels_names = _require_name_list(labels_name, parameter_name="labels_name")
    if len(set(labels_names)) != len(labels_names):
        raise ValueError("Duplicate labels elements are not supported in a single aggregation call.")
    points_names = _broadcast_names(points_name, len(labels_names), parameter_name="points_name")
    coordinate_systems = _broadcast_names(
        to_coordinate_system,
        len(labels_names),
        parameter_name="to_coordinate_system",
    )
    missing_labels = sorted(set(labels_names) - set(sdata.labels))
    if missing_labels:
        raise ValueError(f"Labels elements are not present in 'sdata.labels': {missing_labels}.")
    missing_points = sorted(set(points_names) - set(sdata.points))
    if missing_points:
        raise ValueError(f"Points elements are not present in 'sdata.points': {missing_points}.")

    coordinate_columns: tuple[str, ...] | None = None
    for points in dict.fromkeys(points_names):
        columns = sdata.points[points].columns
        if feature_key not in columns:
            raise ValueError(f"Points element {points!r} does not contain feature column {feature_key!r}.")
        current = ("x", "y", "z") if "z" in columns else ("x", "y")
        if not all(column in columns for column in current):
            raise ValueError(f"Points element {points!r} must contain coordinate columns 'x' and 'y'.")
        if coordinate_columns is None:
            coordinate_columns = current
        elif coordinate_columns != current:
            raise ValueError(
                "All selected points elements must use the same coordinate dimensions, "
                f"found {coordinate_columns} and {current}."
            )

    return tuple(
        _AggregationPair(labels_name=labels, points_name=points, coordinate_system=coordinate_system)
        for labels, points, coordinate_system in zip(
            labels_names,
            points_names,
            coordinate_systems,
            strict=True,
        )
    )


def _validate_aggregation_column_keys(*, feature_key: str, instance_key: str) -> None:
    """Validate distinct public columns used for features and assigned instances."""
    for parameter_name, value in (("feature_key", feature_key), ("instance_key", instance_key)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Parameter {parameter_name!r} must be a non-empty string, found {value!r}.")
    if feature_key == instance_key:
        raise ValueError("Parameters 'feature_key' and 'instance_key' must refer to different columns.")


def _require_name_list(value: str | list[str], *, parameter_name: str) -> list[str]:
    values = _make_list(value)
    if not values:
        raise ValueError(f"Parameter {parameter_name!r} must contain at least one name.")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"Parameter {parameter_name!r} must contain only non-empty strings, found {values!r}.")
    return values


def _broadcast_names(value: str | list[str], size: int, *, parameter_name: str) -> list[str]:
    values = _require_name_list(value, parameter_name=parameter_name)
    if len(values) == 1:
        return values * size
    if len(values) != size:
        raise ValueError(
            f"Parameter {parameter_name!r} must have length 1 or match the number of labels elements "
            f"({size}), found {len(values)}."
        )
    return values


def _resolve_table_index_name(
    table_index_name: str | None,
    *,
    instance_key: str,
    region_key: str,
) -> str:
    """Resolve and validate the name of the output AnnData observation index."""
    result = f"{instance_key}_index" if table_index_name is None else table_index_name
    if not isinstance(result, str) or not result:
        raise ValueError(f"Parameter 'table_index_name' must be a non-empty string or None, found {result!r}.")
    reserved = {instance_key, region_key, _AUXILIARY_POINTS_FRACTION_COLUMN}
    if result in reserved:
        raise ValueError(
            f"Parameter 'table_index_name' must not collide with an aggregation output column, found {result!r}."
        )
    return result


def _resolve_feature_class_contract(
    sdata: SpatialData,
    *,
    pairs: tuple[_AggregationPair, ...],
    feature_key: str,
    expression_class: str,
    region_key: str,
    instance_key: str,
    table_index_name: str,
) -> _FeatureClassAggregationContract:
    """Resolve one immutable feature-panel contract for class-aware aggregation."""
    harpy_metadata = _require_mapping(sdata.attrs.get(_HARPY_METADATA_KEY), path=_HARPY_METADATA_KEY)
    version = harpy_metadata.get(_METADATA_VERSION_KEY)
    if isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION:
        raise ValueError(
            f"Harpy metadata {_HARPY_METADATA_KEY}.{_METADATA_VERSION_KEY} must equal "
            f"{_METADATA_VERSION}, found {version!r}."
        )
    points_registry = _require_mapping(
        harpy_metadata.get(_POINTS_METADATA_KEY),
        path=f"{_HARPY_METADATA_KEY}.{_POINTS_METADATA_KEY}",
    )
    panel_registry = _require_mapping(
        harpy_metadata.get(_FEATURE_PANELS_METADATA_KEY),
        path=f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}",
    )

    resolved: list[_FeaturePanelContract] = []
    for points_name in dict.fromkeys(pair.points_name for pair in pairs):
        points_record = _require_mapping(
            points_registry.get(points_name),
            path=f"{_HARPY_METADATA_KEY}.{_POINTS_METADATA_KEY}.{points_name}",
        )
        panel_name = _require_nonempty_string(
            points_record.get("feature_panel"),
            path=f"{_HARPY_METADATA_KEY}.{_POINTS_METADATA_KEY}.{points_name}.feature_panel",
        )
        panel_record = _require_mapping(
            panel_registry.get(panel_name),
            path=f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}.{panel_name}",
        )
        panel = _parse_feature_panel(panel_record, panel_name=panel_name)
        if panel.feature_key != feature_key:
            raise ValueError(
                f"Feature panel {panel_name!r} declares feature_key={panel.feature_key!r}, "
                f"which does not match feature_key={feature_key!r}."
            )
        if panel.feature_class_key == instance_key:
            raise ValueError(
                f"Feature panel {panel_name!r} declares feature_class_key={panel.feature_class_key!r}, "
                "which must differ from instance_key."
            )
        points = sdata.points[points_name]
        if panel.feature_class_key not in points.columns:
            raise ValueError(
                f"Points element {points_name!r} does not contain panel feature-class column "
                f"{panel.feature_class_key!r}."
            )
        _validate_feature_class_dtype(points, points_name=points_name, panel=panel)
        resolved.append(panel)

    panel = resolved[0]
    for candidate in resolved[1:]:
        if candidate != panel:
            raise ValueError(
                "All points elements in one class-aware aggregation call must reference compatible panels."
            )
    contract = _FeatureClassAggregationContract(
        panel=panel,
        expression_class=expression_class,
    )
    generated = {column for _, column in contract.count_columns}
    collisions = sorted(generated & {region_key, instance_key, table_index_name, _AUXILIARY_POINTS_FRACTION_COLUMN})
    if collisions:
        raise ValueError(f"Generated feature-class columns collide with aggregation output columns: {collisions}.")

    return contract


def _parse_feature_panel(record: Mapping[str, object], *, panel_name: str) -> _FeaturePanelContract:
    path = f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}.{panel_name}"
    feature_key = _require_nonempty_string(record.get("feature_key"), path=f"{path}.feature_key")
    feature_class_key = _require_nonempty_string(
        record.get("feature_class_key"),
        path=f"{path}.feature_class_key",
    )
    if feature_key == feature_class_key:
        raise ValueError(f"Feature panel {panel_name!r} must use different feature and feature-class keys.")

    classes_value = record.get("classes")
    if not isinstance(classes_value, list) or not classes_value:
        raise ValueError(f"Harpy metadata {path}.classes must be a non-empty list of strings.")
    classes = tuple(_require_nonempty_string(value, path=f"{path}.classes item") for value in classes_value)
    if len(set(classes)) != len(classes):
        raise ValueError(f"Harpy metadata {path}.classes must contain unique values.")

    grouped = _require_mapping(record.get("features_by_class"), path=f"{path}.features_by_class")
    if set(grouped) != set(classes):
        raise ValueError(f"Harpy metadata {path}.features_by_class must contain exactly the declared classes.")
    features_by_class_items: list[tuple[str, tuple[str, ...]]] = []
    seen_features: dict[str, str] = {}
    for feature_class in classes:
        values = grouped[feature_class]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Harpy metadata {path}.features_by_class[{feature_class!r}] must be non-empty.")
        features = tuple(
            _require_nonempty_string(value, path=f"{path}.features_by_class[{feature_class!r}] item")
            for value in values
        )
        if len(set(features)) != len(features):
            raise ValueError(f"Harpy metadata {path}.features_by_class[{feature_class!r}] must contain unique values.")
        for feature in features:
            previous = seen_features.setdefault(feature, feature_class)
            if previous != feature_class:
                raise ValueError(
                    f"Feature {feature!r} belongs to both {previous!r} and {feature_class!r} in panel {panel_name!r}."
                )
        features_by_class_items.append((feature_class, features))
    return _FeaturePanelContract(
        feature_key=feature_key,
        feature_class_key=feature_class_key,
        classes=classes,
        features_by_class_items=tuple(features_by_class_items),
    )


def _validate_feature_class_dtype(
    points: DaskDataFrame,
    *,
    points_name: str,
    panel: _FeaturePanelContract,
) -> None:
    dtype = points.dtypes[panel.feature_class_key]
    if not isinstance(dtype, pd.CategoricalDtype):
        raise ValueError(
            f"Points element {points_name!r} feature-class column {panel.feature_class_key!r} "
            f"must be categorical, found {dtype}."
        )
    column = points[panel.feature_class_key]
    if column.cat.known:
        categories = tuple(column.cat.categories.tolist())
        if categories != panel.classes:
            raise ValueError(
                f"Points element {points_name!r} feature-class categories {list(categories)!r} "
                f"do not match panel classes {list(panel.classes)!r}."
            )


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_").casefold()
    if not value:
        raise ValueError("Feature-class names must produce a non-empty snake-case output name.")
    return value


def _label_centers(
    sdata: SpatialData,
    *,
    pair: _AggregationPair,
    instance_ids: np.ndarray,
    coordinate_columns: tuple[str, ...],
    instance_key: str,
) -> pd.DataFrame:
    """Calculate indexed label centers in the pair's coordinate system.

    The labels raster is normalized to ``(z, y, x)`` for
    :class:`~harpy.utils.RasterAggregator`. Only ``instance_ids`` are reduced.
    Its local ``(z, y, x)`` results are translated in ``x`` and ``y``, reordered
    to SpatialData coordinate order, and indexed explicitly so table assembly
    never relies on raster traversal order.

    Parameters
    ----------
    sdata
        SpatialData object containing the labels element.
    pair
        Labels element and target coordinate system for this aggregation pair.
    instance_ids
        Sorted nonzero label IDs receiving at least one assigned point.
    coordinate_columns
        Output order, either ``("x", "y")`` or ``("x", "y", "z")``.
    instance_key
        Name used for the returned instance-ID index.

    Returns
    -------
    pandas.DataFrame
        One finite center-of-mass coordinate row per requested instance ID.
    """
    if instance_ids.size == 0:
        return pd.DataFrame(
            index=pd.Index(instance_ids, name=instance_key),
            columns=coordinate_columns,
            dtype=np.float64,
        )

    labels = _get_spatial_element(sdata, element_name=pair.labels_name)
    label_dims = tuple(labels.dims)
    mask = da.asarray(labels.data)
    if label_dims == ("y", "x"):
        mask = mask[None, ...]
    elif label_dims != ("z", "y", "x"):
        raise ValueError(
            f"Labels element {pair.labels_name!r} must use dimensions ('y', 'x') or ('z', 'y', 'x'), "
            f"found {label_dims!r}."
        )

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=None,
        instance_key=instance_key,
        run_on_gpu=False,
    )
    centers = aggregator.center_of_mass(index=instance_ids)
    centers = centers.rename(columns={0: "z", 1: "y", 2: "x"}).set_index(instance_key)
    if not centers.index.is_unique:
        raise ValueError(f"Labels element {pair.labels_name!r} produced duplicate center-of-mass instance IDs.")
    centers = centers.reindex(instance_ids)
    if centers[["z", "y", "x"]].isna().any(axis=None):
        raise ValueError(f"Labels centers of mass are missing for assigned instances in {pair.labels_name!r}.")

    translation_x, translation_y = _get_translation(labels, to_coordinate_system=pair.coordinate_system)
    centers["x"] += translation_x
    centers["y"] += translation_y
    centers.index.name = instance_key
    return centers.loc[:, list(coordinate_columns)]


def _validate_feature_panel_contents(
    sdata: SpatialData,
    *,
    pairs: tuple[_AggregationPair, ...],
    panel: _FeaturePanelContract,
) -> None:
    """Validate each unique source points element against its feature panel.

    Validation is computed before spatial assignment starts. A points element
    reused by multiple aggregation pairs is scanned only once, and every source
    point is checked, including points that would later fall outside the labels
    raster or on background.
    """
    points_names = tuple(dict.fromkeys(pair.points_name for pair in pairs))
    categorical_dtype = pd.CategoricalDtype(categories=panel.classes)
    error_tasks = []
    for points_name in points_names:
        points = sdata.points[points_name]
        normalized_points = points.assign(
            **{panel.feature_class_key: points[panel.feature_class_key].astype(categorical_dtype)}
        )
        partition_errors = normalized_points[[panel.feature_key, panel.feature_class_key]].map_partitions(
            _feature_panel_partition_errors,
            feature_key=panel.feature_key,
            feature_class_key=panel.feature_class_key,
            class_by_feature=panel.class_by_feature,
            meta=pd.Series(name="error", dtype="object"),
        )
        error_tasks.append(dask.delayed(_first_partition_error)(*partition_errors.to_delayed()))

    computed_errors = dask.compute(*error_tasks)
    for points_name, error in zip(points_names, computed_errors, strict=True):
        if error is not None:
            raise ValueError(f"Points element {points_name!r} disagrees with its feature panel: {error}")


def _first_partition_error(*partitions: pd.Series) -> object | None:
    """Return the first compact error emitted by a partition-wise validation."""
    for partition in partitions:
        if len(partition):
            return partition.iloc[0]
    return None


def _feature_panel_partition_errors(
    partition: pd.DataFrame,
    *,
    feature_key: str,
    feature_class_key: str,
    class_by_feature: Mapping[str, str],
) -> pd.Series:
    """Validate one points partition against its feature-panel assignments.

    Each point must contain a non-null feature and feature class. Its feature
    must occur in ``class_by_feature``, and its observed feature class must
    equal the class assigned by that mapping. Validation is partition-wise so
    the complete points element does not need to be materialized.

    Parameters
    ----------
    partition
        One in-memory pandas partition from the source Dask points element.
    feature_key
        Name of the column containing feature identifiers, such as genes.
    feature_class_key
        Name of the column containing feature classes.
    class_by_feature
        Expected feature class for every feature declared by the panel.

    Returns
    -------
    pandas.Series
        An empty series when the partition is valid, or a one-element series
        containing the first validation error found in the partition.

    Examples
    --------
    ``EPCAM`` is declared endogenous, so observing it as negative produces a
    compact error that the caller can collect alongside the Dask reductions:

    >>> partition = pd.DataFrame({"gene": ["EPCAM"], "code_class": ["Negative"]})
    >>> errors = _feature_panel_partition_errors(
    ...     partition,
    ...     feature_key="gene",
    ...     feature_class_key="code_class",
    ...     class_by_feature={"EPCAM": "Endogenous"},
    ... )
    >>> errors.iloc[0]
    "feature 'EPCAM' has class 'Negative'; expected 'Endogenous'."
    """
    features = partition[feature_key]
    feature_classes = partition[feature_class_key]
    invalid = features.isna() | feature_classes.isna()
    if invalid.any():
        return pd.Series(["feature and feature-class values must not be null."], name="error", dtype="object")
    expected = features.astype(object).map(class_by_feature)
    missing = expected.isna()
    if missing.any():
        feature = features.loc[missing].iloc[0]
        return pd.Series([f"feature {feature!r} is absent from the panel."], name="error", dtype="object")
    mismatched = feature_classes.astype(object) != expected
    if mismatched.any():
        position = int(np.flatnonzero(mismatched.to_numpy())[0])
        return pd.Series(
            [
                f"feature {features.iloc[position]!r} has class {feature_classes.iloc[position]!r}; "
                f"expected {expected.iloc[position]!r}."
            ],
            name="error",
            dtype="object",
        )
    return pd.Series(name="error", dtype="object")


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Harpy metadata {path} must be a mapping.")
    return value


def _require_nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Harpy metadata {path} must be a non-empty string.")
    return value


def bin_counts(
    sdata: SpatialData,
    table_name: str,
    labels_name: str,
    output_table_name: str,
    to_coordinate_system: str = "global",
    chunks: str | tuple[int, ...] | int | None = None,
    append: bool = True,
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    spatial_key: str = _SPATIAL,
    cell_index_name: str = _CELL_INDEX,
    overwrite: bool = False,
) -> SpatialData:
    """
    Bins gene counts from barcodes to cells or regions defined in `labels_name` and returns an updated SpatialData object with a table element (`sdata.tables[output_table_name]`) holding an AnnData object with the binned counts per cell or region.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_name
        The table element holding the counts. E.g. obtained using :func:`harpy.io.visium_hd`.
        We assume that `sdata[table_name].obsm[spatial_key]` contains a numpy array holding the barcode coordinates ('x', 'y').
        The relation of `sdata[table_name].obsm[spatial_key]` to `to_coordinate_system` should be an identity transformation.
    labels_name
        The labels element (e.g., segmentation mask, or a grid generated by :func:`harpy.im.add_grid_labels`) in `sdata` used to bin barcodes (as specified via `table_name`) into cells or regions.
    output_table_name
        The table element in `sdata` in which to save the AnnData object with the binned counts per cell or region defined by `labels_name`.
    to_coordinate_system
        The coordinate system that holds `labels_name`.
    chunks
        Chunk sizes for processing. Can be a string, integer, or tuple of integers.
        Consider setting the chunks to a relatively high value to speed up processing,
        taking into account the available memory of your system.
    append
        If set to `True`, and the `labels_name` does not yet exist as a `region_key` in `sdata.tables[output_table_name].obs`,
        the binned counts obtained during the current function call will be appended (along axis=0) to `output_table_name`.
        If `False`, and `overwrite` is set to `True`, any existing data in `sdata.tables[output_table_name]` will be overwritten by the newly binned counts.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids.
    region_key
        Region key. The name of the column in  :class:`~anndata.AnnData` table `.obs` that will hold the name of the elements that is annotated by the resulting table.
    spatial_key
        The key in the :class:`~anndata.AnnData` table `.obsm` that will hold the `x` and `y` center of the instances.
        This center is calculated taking the average x,y coordinate of the assigned spots per bin/cell.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
    overwrite
        If `True`, overwrites the `output_table_name` if it already exists in `sdata`.

    Returns
    -------
    An updated SpatialData object with an AnnData table added to `sdata.tables` at slot `output_table_name`.

    Example
    --------
    .. code-block:: python

        import harpy as hp

        sdata_bin = hp.datasets.visium_hd_example_custom_binning()

        table_name_bins = "square_002um"
        labels_name = (
            "square_labels_32"  # custom grid to bin the counts of table_name_bins; can be any segmentation mask
        )
        table_name = "table_custom_bin_32"
        output_table_name = f"{table_name}_reproduce"

        # Check that barcodes are unique in table_name_bins of sdata_bin
        assert sdata_bin.tables[table_name_bins].obs.index.is_unique

        sdata_bin = hp.tb.bin_counts(
            sdata_bin,
            table_name=table_name_bins,
            labels_name=labels_name,
            output_table_name=output_table_name,
            overwrite=True,
            append=False,
        )
    """
    se = _get_spatial_element(sdata, element_name=labels_name)

    # sdata[table_name].obsm[spatial_key] contains the positions of the barcodes if visium reader is used 'harpy.io.visium_hd'
    name_x = "x"
    name_y = "y"
    df = pd.DataFrame(sdata[table_name].obsm[spatial_key], columns=[name_x, name_y])
    name_barcode_id = "barcode_id"
    df[name_barcode_id] = sdata[table_name].obs.index

    ddf = PointsModel.parse(
        df,
        transformations={to_coordinate_system: Identity()},
    )

    combined_partitions = _assign_points_to_labels(
        se=se,
        ddf=ddf,
        chunks=chunks,
        to_coordinate_system=to_coordinate_system,
        name_x=name_x,
        name_y=name_y,
        drop_coordinates=False,
        value_key=name_barcode_id,
        instance_key=cell_index_name,
    )

    coordinates = combined_partitions.groupby(cell_index_name)[name_x, name_y].mean()

    cell_counts = combined_partitions.groupby([name_barcode_id, cell_index_name]).size()

    cell_counts = cell_counts.map_partitions(lambda x: x.astype(np.uint32))

    coordinates, cell_counts = dask.compute(coordinates, cell_counts)

    # Sanity check that every barcode that could be assigned to a bin is assigned exactly ones to a bin.
    _mask = cell_counts == 1
    assert _mask.all(), (
        f"Some spots, given by 'sdata.tables[{table_name}].obsm[{spatial_key}]', where assigned to more than one cell defined in '{labels_name}'."
    )
    cell_counts = cell_counts.reset_index(level=cell_index_name)
    assert cell_counts.index.is_unique, "Spots should not be assigned to more than one cell."

    value_counts_counter = Counter(cell_counts.groupby(cell_index_name).count()[0])
    value_counts_sorted = sorted(value_counts_counter.items())
    df = pd.DataFrame(value_counts_sorted, columns=["Number of spots per bin", "Frequency"])
    log.info(f"\n{df.to_string(index=False)}")
    # get adata
    adata_in = sdata.tables[table_name].copy()  # should we do a copy here? otherwise in memory adata will be changed
    merged = pd.merge(adata_in.obs, cell_counts[cell_index_name], left_index=True, right_index=True, how="inner")
    assert merged.shape[0] != 0, (
        "Result after merging AnnData object, passed via 'table_name' parameter with aggregated spots is empty."
    )
    adata_in = adata_in[merged.index]
    adata_in.obs = merged

    group_labels = adata_in.obs[cell_index_name].values
    unique_labels, group_indices = np.unique(group_labels, return_inverse=True)
    N_groups = len(unique_labels)

    assert issparse(adata_in.X), "Currently only AnnData objects with a sparse feature matrix are supported."

    # Extract the gene expression counts
    counts = adata_in.X

    rows = group_indices
    cols = np.arange(len(group_indices))
    data = np.ones(len(group_indices))
    group_indicator = sparse.csr_matrix((data, (rows, cols)), shape=(N_groups, counts.shape[0]))

    summed_counts = group_indicator.dot(counts)

    # exclude bins for which sum is zero (i.e. no genes detected)
    row_sums = np.array(summed_counts.sum(axis=1)).flatten()
    nonzero_rows = row_sums != 0
    summed_counts = summed_counts[nonzero_rows, :]
    unique_labels = unique_labels[nonzero_rows]

    adata = AnnData(
        X=summed_counts, obs=pd.DataFrame(unique_labels, columns=[instance_key], index=unique_labels), var=adata_in.var
    )

    adata.obs[region_key] = pd.Categorical([labels_name] * len(adata.obs))

    _uuid_value = str(uuid.uuid4())[:8]
    adata.obs.index = adata.obs.index.map(lambda x: f"{x}_{labels_name}_{_uuid_value}")
    adata.obs.index.name = cell_index_name

    # now add the coordinates
    # coordinates are the average x,y coordinate of the assigned spots per bin/cell
    # adata.obs[ instance_key ] is also sorted. And index of coordinates corresponds to instance_key.
    adata.obsm[spatial_key] = coordinates[coordinates.index.isin(adata.obs[instance_key])].sort_index().values

    if append:
        region = []
        if output_table_name in [*sdata.tables]:
            _sanity_check_append_region(
                adata=sdata.tables[output_table_name],
                region_key=region_key,
                instance_key=instance_key,
                region=labels_name,
            )
            adata = ad.concat([sdata.tables[output_table_name], adata], axis=0)
            # get the regions already in sdata, and append the new one
            region = sdata.tables[output_table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        region.append(labels_name)

    else:
        region = [labels_name]

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    return sdata


def _assign_points_to_labels(
    se: DataArray,
    ddf: DaskDataFrame,
    value_key: str | Sequence[str],
    drop_coordinates: bool = False,  # if set to True, will drop ((z),y,x) in resulting dask dataframe
    chunks: str | tuple[int, ...] | int | None = None,
    to_coordinate_system: str = "global",
    name_x: str = "x",
    name_y: str = "y",
    name_z: str = "z",
    instance_key: str = _INSTANCE_KEY,
) -> DaskDataFrame:
    """Assign points to non-background labels through the labels chunk grid.

    Points are first rounded to integer pixel coordinates and mapped once to a
    temporary row-major labels-block ID. For a two-dimensional raster the
    mapping is:

    ::

                     x chunk
                    0       1
                +-------+-------+
        y chunk 0 | id 0  | id 1  |
                +-------+-------+
        y chunk 1 | id 2  | id 3  |
                +-------+-------+

    Chunk intervals are half-open, so a point exactly on an internal boundary
    belongs to the chunk beginning at that boundary. Points outside the full
    labels extent are removed before the points are redistributed once by
    block ID. Each resulting points partition is paired with the corresponding
    delayed labels chunk and looked up vectorially. Three-dimensional labels
    apply the same rule in row-major ``(z, y, x)`` order.

    Labels may have an integer-pixel translation in ``to_coordinate_system``;
    points must have an identity transformation there. Graph construction does
    not read either source. The returned Dask dataframe contains one row per
    point assigned to a nonzero label, the requested value columns, the
    assigned ``instance_key`` column and, unless ``drop_coordinates=True``,
    the rounded spatial-coordinate columns. Requested categorical dtypes are
    preserved.

    Redistribution does not preserve the input index, row order, or partition
    order. The temporary block ID is absent from the returned dataframe.

    Parameters
    ----------
    se
        Two- or three-dimensional integer labels raster in ``(y, x)`` or
        ``(z, y, x)`` order.
    ddf
        Points dataframe with an identity transformation to
        ``to_coordinate_system``.
    value_key
        Column or columns retained alongside the assigned label ID.
    drop_coordinates
        Whether to omit the rounded point coordinates from the result.
    chunks
        Optional virtual rechunking of the labels raster. ``None`` preserves
        its existing chunks.
    to_coordinate_system
        Coordinate system in which points and translated labels are aligned.
    name_x, name_y, name_z
        Point coordinate-column names.
    instance_key
        Name of the output label-ID column.

    Returns
    -------
    Lazy assigned-points dataframe. Its index and ordering are unspecified.

    Examples
    --------
    With ``value_key="gene"``, the raster lookup conceptually transforms
    individual input points as follows. ``cell_ID`` contains the label value at
    the rounded point coordinate::

        input points                  assigned points
        x     y    gene               gene    cell_ID
        12.1  8.9  EPCAM       -->    EPCAM   42
        14.2  9.1  EPCAM       -->    EPCAM   42
        80.0  4.0  Neg01       -->    omitted: label 0

    With ``drop_coordinates=True``, the output has one column for every
    requested ``value_key`` plus the assigned instance-ID column. The exact row
    count remains unknown until computation and cannot exceed the input point
    count. This function performs assignment only; repeated
    ``(cell_ID, gene)`` rows are aggregated into counts by the caller.
    """
    if not np.issubdtype(se.data.dtype, np.integer):
        raise ValueError(f"Labels must use an integer dtype, found {se.data.dtype}.")
    missing_xy = [key for key in (name_x, name_y) if key not in ddf.columns]
    if missing_xy:
        raise ValueError(f"Points dataframe is missing required coordinate columns: {missing_xy}.")
    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)

    requested_value_keys = list(dict.fromkeys([value_key] if isinstance(value_key, str) else value_key))
    missing_value_keys = [key for key in requested_value_keys if key not in ddf.columns]
    if missing_value_keys:
        raise ValueError(f"Dask DataFrame does not contain requested value columns: {missing_value_keys}.")
    dimensions = tuple(se.dims)
    if dimensions == ("y", "x"):
        if name_z in ddf.columns:
            raise ValueError(
                f"Two-dimensional labels require only '{name_x}' and '{name_y}' point coordinates; "
                f"unexpected column '{name_z}' was found."
            )
        coordinate_keys = [name_x, name_y]
    elif dimensions == ("z", "y", "x"):
        if name_z not in ddf.columns:
            raise ValueError(f"Three-dimensional labels require point coordinate column '{name_z}'.")
        coordinate_keys = [name_x, name_y, name_z]
    else:
        raise ValueError(f"Labels dimensions must be ('y', 'x') or ('z', 'y', 'x'), found {dimensions!r}.")

    translation_x, translation_y = _get_translation(se, to_coordinate_system=to_coordinate_system)
    translations = {
        "x": _normalize_pixel_translation(translation_x, axis="x"),
        "y": _normalize_pixel_translation(translation_y, axis="y"),
        "z": 0,
    }

    value_keys = list(dict.fromkeys([*coordinate_keys, *requested_value_keys]))
    arr = se.data
    if chunks is not None:
        arr = arr.rechunk(chunks)

    log.info("Calculating cell counts.")

    projected = ddf[value_keys]
    block_id_key = "__harpy_block_id"
    while block_id_key in projected.columns:
        block_id_key = f"_{block_id_key}"
    boundaries = tuple(tuple(np.cumsum((0, *axis_chunks), dtype=np.int64).tolist()) for axis_chunks in arr.chunks)
    grid_shape = tuple(len(axis_chunks) for axis_chunks in arr.chunks)
    number_of_blocks = int(np.prod(grid_shape))
    translation_by_dimension = tuple(translations[dimension] for dimension in dimensions)
    coordinate_key_by_dimension = tuple({"x": name_x, "y": name_y, "z": name_z}[dimension] for dimension in dimensions)

    classified_meta = projected._meta.copy()
    for key in coordinate_keys:
        classified_meta[key] = pd.Series(index=classified_meta.index, dtype=np.int64)
    classified_meta[block_id_key] = pd.Series(index=classified_meta.index, dtype=np.int64)
    classified = projected.map_partitions(
        _classify_points_by_label_block,
        coordinate_keys=coordinate_key_by_dimension,
        boundaries=boundaries,
        translations=translation_by_dimension,
        grid_shape=grid_shape,
        block_id_key=block_id_key,
        meta=classified_meta,
    )

    # Explicit divisions prevent Dask from sampling the points to estimate
    # quantiles. They also give one points partition per labels chunk, so both
    # collections can be paired positionally in row-major block order.
    # This remains a full shuffle because arbitrary input partitions may contain
    # points from any block. If points elements later expose a trusted spatial
    # partition index aligned with the labels-block grid, those already aligned
    # partitions can be paired directly and this redistribution can be skipped.
    routed = classified.set_index(
        block_id_key,
        divisions=tuple(range(number_of_blocks + 1)),
    )

    # One division per labels block guarantees spatial alignment, not balanced
    # row counts: a dense block can produce a much larger points partition.
    # This is mainly a concern for pathologically concentrated point
    # distributions; users can normally reduce ``chunks`` to subdivide dense
    # spatial blocks. Multiple point shards per labels block could provide a
    # future safeguard while allowing every shard to reuse the same labels
    # chunk.
    point_blocks = routed.to_delayed()
    label_blocks = arr.to_delayed().ravel()

    returned_columns = [*requested_value_keys, instance_key] if drop_coordinates else [*value_keys, instance_key]
    result_meta = projected._meta[value_keys].copy()
    for key in coordinate_keys:
        result_meta[key] = pd.Series(index=result_meta.index, dtype=np.int64)
    result_meta[instance_key] = pd.Series(index=result_meta.index, dtype=arr.dtype)
    result_meta = result_meta[returned_columns]
    result_meta.index = pd.RangeIndex(0)

    starts_by_dimension = tuple(
        tuple(np.cumsum((0, *axis_chunks[:-1]), dtype=np.int64).tolist()) for axis_chunks in arr.chunks
    )
    assigned_blocks = []
    for block_indices, point_block, label_block in zip(np.ndindex(grid_shape), point_blocks, label_blocks, strict=True):
        block_start = tuple(
            starts[block_index] for starts, block_index in zip(starts_by_dimension, block_indices, strict=True)
        )
        assigned_blocks.append(
            dask.delayed(_lookup_points_in_label_block)(
                point_block,
                label_block,
                coordinate_keys=coordinate_key_by_dimension,
                translations=translation_by_dimension,
                block_start=block_start,
                requested_columns=returned_columns,
                instance_key=instance_key,
                label_dtype=arr.dtype,
            )
        )

    return dd.from_delayed(assigned_blocks, meta=result_meta)


def _normalize_pixel_translation(value: float, *, axis: str) -> int:
    """Normalize a numerically integral labels translation to pixel units."""
    if not np.isfinite(value):
        raise ValueError(f"Labels translation along {axis!r} must be finite, found {value!r}.")
    nearest = int(np.rint(value))
    if not np.isclose(value, nearest, rtol=0, atol=1e-6):
        raise ValueError(f"Labels translation along {axis!r} must be pixel-aligned, found {value!r}.")
    return nearest


def _classify_points_by_label_block(
    partition: pd.DataFrame,
    *,
    coordinate_keys: tuple[str, ...],
    boundaries: tuple[tuple[int, ...], ...],
    translations: tuple[int, ...],
    grid_shape: tuple[int, ...],
    block_id_key: str,
) -> pd.DataFrame:
    """Round and classify one points partition into row-major labels blocks."""
    result = partition.copy()
    local_coordinates = []
    inside = np.ones(len(result), dtype=bool)
    for key, axis_boundaries, translation in zip(coordinate_keys, boundaries, translations, strict=True):
        result[key] = result[key].round().astype(np.int64)
        local = result[key].to_numpy(dtype=np.int64, copy=False) - translation
        local_coordinates.append(local)
        inside &= (local >= axis_boundaries[0]) & (local < axis_boundaries[-1])

    result = result.loc[inside].copy()
    if result.empty:
        result[block_id_key] = pd.Series(index=result.index, dtype=np.int64)
        return result

    block_indices = tuple(
        np.searchsorted(axis_boundaries[1:], local[inside], side="right")
        for local, axis_boundaries in zip(local_coordinates, boundaries, strict=True)
    )
    result[block_id_key] = np.ravel_multi_index(block_indices, grid_shape).astype(np.int64, copy=False)
    return result


def _lookup_points_in_label_block(
    points: pd.DataFrame,
    labels: np.ndarray,
    *,
    coordinate_keys: tuple[str, ...],
    translations: tuple[int, ...],
    block_start: tuple[int, ...],
    requested_columns: list[str],
    instance_key: str,
    label_dtype: np.dtype,
) -> pd.DataFrame:
    """Assign one routed points partition from its corresponding labels block.

    The preceding block-classification and shuffle stages guarantee that every
    row in ``points`` lies within the spatial extent of ``labels``. Point
    coordinates are still expressed in the selected coordinate system, whereas
    ``labels`` is the NumPy array for one chunk of the untranslated labels
    raster. For each spatial axis, the coordinate inside that chunk is

    ::

        chunk_coordinate = point_coordinate - translation - block_start

    The resulting coordinate arrays are used together for one vectorized,
    pointwise lookup. For example, coordinates ``y=[1, 2]`` and ``x=[3, 4]``
    retrieve ``labels[1, 3]`` and ``labels[2, 4]``; they do not select their
    Cartesian product. The retrieved raster value is written to
    ``instance_key`` and rows assigned to background label zero are removed.

    An empty points block returns an empty dataframe with the same schema and a
    label-ID column using ``label_dtype``. The returned dataframe contains only
    ``requested_columns`` and receives a fresh, partition-local range index;
    neither its input index nor row ordering is part of the contract.

    Parameters
    ----------
    points
        In-memory points partition routed to this labels block. Its coordinates
        are rounded integer positions in the selected coordinate system.
    labels
        In-memory two- or three-dimensional array containing the corresponding
        labels chunk.
    coordinate_keys
        Coordinate columns in labels-array axis order: ``(y, x)`` for 2D or
        ``(z, y, x)`` for 3D.
    translations
        Integer origin of the complete labels raster in the selected coordinate
        system, ordered like ``coordinate_keys``. The z translation is zero.
    block_start
        Origin of this chunk within the untranslated labels raster, ordered like
        ``coordinate_keys``.
    requested_columns
        Ordered columns retained in the returned dataframe, including
        ``instance_key``.
    instance_key
        Name of the column receiving the overlapping label ID.
    label_dtype
        Labels dtype used to construct the label-ID column for an empty block.

    Returns
    -------
    pandas.DataFrame
        Points assigned to nonzero labels, projected to ``requested_columns``.
    """
    result = points.copy()
    if result.empty:
        result[instance_key] = pd.Series(index=result.index, dtype=label_dtype)
    else:
        local_coordinates = tuple(
            result[key].to_numpy(dtype=np.int64, copy=False) - translation - start
            for key, translation, start in zip(coordinate_keys, translations, block_start, strict=True)
        )
        result[instance_key] = labels[local_coordinates]
        result = result.loc[result[instance_key] != 0]

    return result[requested_columns].reset_index(drop=True)
