from __future__ import annotations

import re
import uuid
from collections import Counter, namedtuple
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
from harpy.table._table import add_table
from harpy.table._utils import _sanity_check_append_region
from harpy.utils._aggregate import RasterAggregator
from harpy.utils._keys import _CELL_INDEX, _FEATURE_MATRICES_KEY, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from harpy.utils._transformations import _identity_check_transformations_points
from harpy.utils.utils import _make_list

_FEATURE_CLASS_AGGREGATION_KEY = "feature_class_aggregation"
_AUXILIARY_FEATURE_MATRIX_KEY = "auxiliary_feature_counts"
_FEATURE_MATRIX_SCHEMA_VERSION = 1
_CONTROL_FRACTION_COLUMN = "control_fraction"
_AGGREGATE_SOURCE_KIND = "harpy_aggregate_points"
_DEPRECATED_ATTRIBUTES_WARNED: set[str] = set()


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
    class is treated as a control class. Output count-column names, control
    classes, and control denominators are derived from the panel so they
    cannot disagree with its metadata. The contract contains no spatial
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
        if not self.control_classes:
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
    def control_classes(self) -> tuple[str, ...]:
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
        return tuple(feature for feature_class in self.control_classes for feature in features_by_class[feature_class])

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
        for feature_class in self.control_classes:
            stop = start + len(features_by_class[feature_class])
            result.append((feature_class, slice(start, stop)))
            start = stop
        return tuple(result)

    @property
    def control_class_denominators(self) -> tuple[tuple[str, int], ...]:
        """Return the panel-defined feature count for each non-expression class.

        Each denominator is the number of features assigned to that class in
        the authoritative panel, including panel features for which the
        selected points elements contain zero transcript detections. These
        values are recorded with the aggregation metadata so downstream QC can
        normalize per-class point counts without inferring panel size from the
        observed data.
        """
        features_by_class = self.panel.features_by_class
        return tuple((feature_class, len(features_by_class[feature_class])) for feature_class in self.control_classes)


@dataclass(frozen=True)
class _PairReductions:
    """Materialized reductions for one paired points and labels aggregation.

    Attributes
    ----------
    pair
        Names and coordinate system identifying the paired labels and points
        elements.
    centers
        Segmentation-mask centers of mass, indexed by instance ID.
    coordinate_columns
        Coordinate column names derived from ``centers`` in matrix order.
    feature_counts
        Series mapping each observed ``(instance, feature)`` pair to the
        number of assigned points carrying that feature.
    """

    pair: _AggregationPair
    centers: pd.DataFrame
    feature_counts: pd.Series

    def __post_init__(self) -> None:
        coordinate_columns = self.coordinate_columns
        if coordinate_columns not in {("x", "y"), ("x", "y", "z")}:
            raise ValueError(
                "Pair-reduction coordinates must use columns ('x', 'y') or ('x', 'y', 'z'), "
                f"found {coordinate_columns!r}."
            )
        if not self.centers.index.is_unique:
            raise ValueError("Pair-reduction center instance IDs must be unique.")

        feature_instances = self._validate_feature_counts()
        coordinate_instances = self.centers.index
        if (
            len(coordinate_instances) != len(feature_instances)
            or not coordinate_instances.isin(feature_instances).all()
            or not feature_instances.isin(coordinate_instances).all()
        ):
            raise ValueError("Pair-reduction centers and feature counts must use the same instances.")

    @property
    def coordinate_columns(self) -> tuple[str, ...]:
        """Return the coordinate columns in their stored matrix order."""
        return tuple(self.centers.columns)

    @property
    def instance_ids(self) -> np.ndarray:
        """Return instance IDs in the pair's established output-row order."""
        return self.centers.index.to_numpy()

    def _validate_feature_counts(self) -> pd.Index:
        """Validate feature counts and return their referenced instance IDs."""
        counts = self.feature_counts
        if not isinstance(counts, pd.Series):
            raise ValueError("Pair-reduction feature_counts must be a pandas Series.")
        if counts.dtype != np.dtype(np.uint32):
            raise ValueError(f"Pair-reduction feature_counts must use uint32 values, found {counts.dtype}.")
        if not isinstance(counts.index, pd.MultiIndex) or counts.index.nlevels != 2:
            raise ValueError("Pair-reduction feature_counts must use a two-level MultiIndex.")
        if counts.index.names[0] != self.centers.index.name:
            raise ValueError(
                f"Pair-reduction feature_counts instance level {counts.index.names[0]!r} must match "
                f"center index {self.centers.index.name!r}."
            )
        level = counts.index.levels[0]
        codes = counts.index.codes[0]
        used_codes = np.flatnonzero(np.bincount(codes, minlength=len(level)))
        return level.take(used_codes)


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
    cell_index_name: str = _CELL_INDEX,
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

    When ``expression_class`` is ``None``, all observed values from
    ``feature_key`` are retained in the expression matrix. When an
    expression class is selected, aggregation resolves the points element's
    feature panel through::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
            -> sdata.attrs["harpy"]["feature_panels"][feature_panel]

    The referenced panel supplies ``feature_key``, ``feature_class_key``,
    ``classes``, and ``features_by_class``. Features in ``expression_class``
    define ``adata.X``. Every remaining panel feature is retained in the
    independent sparse matrix ``adata.obsm["auxiliary_feature_counts"]``, whose
    ordered columns are recorded under
    ``adata.uns["feature_matrices"]["auxiliary_feature_counts"]``. Instances
    receiving only non-expression points remain in the table with an all-zero
    expression row.

    Every panel class is also summarized as ``n_<class>_points`` in
    ``adata.obs``, together with ``control_fraction``. These summaries are
    derived from the persisted expression and auxiliary matrices. Control
    denominators are the lengths of the panel's non-expression
    ``features_by_class`` lists and are recorded in
    ``adata.uns["feature_class_aggregation"]`` for later QC; no per-feature
    rates are persisted in ``adata.obs``. Class-aware aggregation fails when
    the panel metadata is unavailable, malformed, or incompatible across
    selected points elements.

    Coordinates in ``adata.obsm[spatial_key]`` are geometric centers of mass
    calculated from the paired labels rasters for the retained instance IDs.
    This definition is used in both ordinary and class-aware modes and does not
    depend on point positions or feature classes.

    Table construction does not revalidate the payload it has just assembled.
    Call :func:`harpy.tb.validate_table` explicitly to check a table against its
    persisted SpatialData annotation, feature-matrix records, source references,
    and authoritative feature-panel metadata.

    Parameters
    ----------
    sdata
        The SpatialData object.
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
        ``feature_key``.
    expression_class
        Feature class retained in ``adata.X``. If ``None``, feature-panel
        metadata is not consulted and ordinary aggregation retains all observed
        features.
    instance_key
        Column in ``adata.obs`` holding instance identifiers.
    region_key
        Categorical column in ``adata.obs`` holding labels-element names.
    spatial_key
        Key in ``adata.obsm`` holding segmentation-label centers of mass in the
        selected coordinate systems.
    cell_index_name
        Name of the resulting observation index.
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
    pairs = _normalize_aggregation_pairs(
        sdata,
        labels_name=labels_name,
        points_name=points_name,
        to_coordinate_system=to_coordinate_system,
        feature_key=feature_key,
    )
    if output_table_name in sdata.tables and not overwrite:
        raise ValueError(
            f"Table element {output_table_name!r} already exists in 'sdata.tables'. Set 'overwrite=True' to replace it."
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
        )

    reductions = tuple(
        _reduce_aggregation_pair(
            sdata,
            pair=pair,
            feature_key=feature_key,
            cell_index_name=cell_index_name,
            chunks=chunks,
            contract=contract,
        )
        for pair in pairs
    )
    adata = _assemble_aggregation_table(
        reductions,
        feature_key=feature_key,
        cell_index_name=cell_index_name,
        region_key=region_key,
        instance_key=instance_key,
        spatial_key=spatial_key,
        contract=contract,
    )

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=[pair.labels_name for pair in pairs],
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )
    return sdata


def __getattr__(name: str) -> object:
    if name == "allocate":
        if name not in _DEPRECATED_ATTRIBUTES_WARNED:
            _DEPRECATED_ATTRIBUTES_WARNED.add(name)
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
    if not isinstance(feature_key, str) or not feature_key:
        raise ValueError(f"Parameter 'feature_key' must be a non-empty string, found {feature_key!r}.")

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


def _resolve_feature_class_contract(
    sdata: SpatialData,
    *,
    pairs: tuple[_AggregationPair, ...],
    feature_key: str,
    expression_class: str,
    region_key: str,
    instance_key: str,
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
    collisions = sorted(generated & {region_key, instance_key, _CONTROL_FRACTION_COLUMN})
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


def _reduce_aggregation_pair(
    sdata: SpatialData,
    *,
    pair: _AggregationPair,
    feature_key: str,
    cell_index_name: str,
    chunks: str | tuple[int, ...] | int | None,
    contract: _FeatureClassAggregationContract | None,
) -> _PairReductions:
    """Assign one points element once and compute its compact pair-level reductions."""
    points = sdata.points[pair.points_name]
    coordinate_columns = ("x", "y", "z") if "z" in points.columns else ("x", "y")
    class_key = None if contract is None else contract.panel.feature_class_key
    normalized_points = points
    if contract is not None:
        categorical_dtype = pd.CategoricalDtype(categories=contract.panel.classes)
        # Keep the SpatialData-owned points object unchanged because Dask
        # ``assign`` does not preserve its transformation metadata. The
        # normalized branch is used only for panel-content validation.
        normalized_points = points.assign(**{class_key: points[class_key].astype(categorical_dtype)})

    # One row per point overlapping a non-background label. Each row retains
    # the requested feature columns and gains the overlapping label ID in
    # ``cell_index_name``; for example: (feature="EPCAM", label_id=42).
    # Coordinates are dropped because table coordinates are calculated from
    # the labels raster after the assigned-instance universe is known.
    assigned_points = _assign_points_to_labels(
        se=_get_spatial_element(sdata, element_name=pair.labels_name),
        ddf=points,
        value_key=feature_key if class_key is None else [feature_key, class_key],
        drop_coordinates=True,
        to_coordinate_system=pair.coordinate_system,
        chunks=chunks,
        cell_index_name=cell_index_name,
    )
    # Convert categorical features to strings before grouping. Otherwise, the
    # categorical groupby may emit the Cartesian product of instances and
    # declared feature categories, including zero-count combinations.
    points_for_feature_counts = assigned_points.assign(**{feature_key: assigned_points[feature_key].astype("str")})
    # Number of assigned points for every (label ID, feature) pair; these
    # counts become the expression matrix when the output table is assembled,
    # for example: (label_id=42, feature="EPCAM") -> 7 points.
    feature_counts = points_for_feature_counts.groupby([cell_index_name, feature_key]).size()
    feature_counts = feature_counts.map_partitions(lambda value: value.astype(np.uint32))

    if contract is None:
        errors = None
    else:
        errors = normalized_points[[feature_key, class_key]].map_partitions(
            _feature_panel_partition_errors,
            feature_key=feature_key,
            feature_class_key=class_key,
            class_by_feature=contract.panel.class_by_feature,
            meta=pd.Series(name="error", dtype="object"),
        )

    if errors is None:
        computed_feature_counts = feature_counts.compute()
    else:
        computed_feature_counts, computed_errors = dask.compute(feature_counts, errors)
        if not computed_errors.empty:
            raise ValueError(
                f"Points element {pair.points_name!r} disagrees with its feature panel: {computed_errors.iloc[0]}"
            )

    instance_ids = np.sort(_used_count_level(computed_feature_counts, key=cell_index_name).to_numpy())
    centers = _label_centers(
        sdata,
        pair=pair,
        instance_ids=instance_ids,
        coordinate_columns=coordinate_columns,
        cell_index_name=cell_index_name,
    )

    return _PairReductions(
        pair=pair,
        centers=centers,
        feature_counts=computed_feature_counts,
    )


def _label_centers(
    sdata: SpatialData,
    *,
    pair: _AggregationPair,
    instance_ids: np.ndarray,
    coordinate_columns: tuple[str, ...],
    cell_index_name: str,
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
    cell_index_name
        Name used for the returned instance-ID index.

    Returns
    -------
    pandas.DataFrame
        One finite center-of-mass coordinate row per requested instance ID.
    """
    if instance_ids.size == 0:
        return pd.DataFrame(
            index=pd.Index(instance_ids, name=cell_index_name),
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
        instance_key=cell_index_name,
        run_on_gpu=False,
    )
    centers = aggregator.center_of_mass(index=instance_ids)
    centers = centers.rename(columns={0: "z", 1: "y", 2: "x"}).set_index(cell_index_name)
    if not centers.index.is_unique:
        raise ValueError(f"Labels element {pair.labels_name!r} produced duplicate center-of-mass instance IDs.")
    centers = centers.reindex(instance_ids)
    if centers[["z", "y", "x"]].isna().any(axis=None):
        raise ValueError(f"Labels centers of mass are missing for assigned instances in {pair.labels_name!r}.")

    translation_x, translation_y = _get_translation(labels, to_coordinate_system=pair.coordinate_system)
    centers["x"] += translation_x
    centers["y"] += translation_y
    centers.index.name = cell_index_name
    return centers.loc[:, list(coordinate_columns)]


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


def _count_index_level(counts: pd.Series, *, key: str) -> tuple[pd.Index, np.ndarray]:
    """Return one count-index level and its compact code per observed pair."""
    index = counts.index
    if not isinstance(index, pd.MultiIndex):
        raise RuntimeError("Reduced point counts must use a MultiIndex.")
    try:
        level_number = index.names.index(key)
    except ValueError as exc:
        raise RuntimeError(f"Reduced point counts are missing index level {key!r}.") from exc
    codes = index.codes[level_number]
    if (codes < 0).any():
        raise RuntimeError(f"Reduced point counts contain missing values in index level {key!r}.")
    return index.levels[level_number], codes


def _used_count_level(counts: pd.Series, *, key: str) -> pd.Index:
    """Return unique level values referenced by at least one observed count pair."""
    level, codes = _count_index_level(counts, key=key)
    used_codes = np.flatnonzero(np.bincount(codes, minlength=len(level)))
    return level.take(used_codes)


def _assemble_aggregation_table(
    reductions: tuple[_PairReductions, ...],
    *,
    feature_key: str,
    cell_index_name: str,
    region_key: str,
    instance_key: str,
    spatial_key: str,
    contract: _FeatureClassAggregationContract | None,
) -> AnnData:
    """Align pair reductions to their output axes and construct one AnnData table."""
    if contract is None:
        observed_features: set[str] = set()
        for result in reductions:
            # Ordinary mode has no panel-defined feature axis. Collect only
            # features referenced by each count MultiIndex; _counts_to_sparse
            # later maps every pair's internal level codes onto the resulting
            # shared sorted axis, which also becomes adata.var_names.
            observed_features.update(_used_count_level(result.feature_counts, key=feature_key))
        expression_feature_axis = tuple(sorted(observed_features))
        auxiliary_feature_axis: tuple[str, ...] = ()
    else:
        expression_feature_axis = contract.expression_feature_axis
        auxiliary_feature_axis = contract.auxiliary_feature_axis
    if not expression_feature_axis:
        raise ValueError("Aggregation produced no expression features.")

    coordinate_columns = reductions[0].coordinate_columns
    if any(result.coordinate_columns != coordinate_columns for result in reductions[1:]):
        raise ValueError("All aggregation pairs must use the same coordinate dimensions.")

    expression_matrices: list[sparse.csr_matrix] = []
    auxiliary_matrices: list[sparse.csr_matrix] = []
    obs_frames: list[pd.DataFrame] = []
    coordinate_blocks: list[np.ndarray] = []
    uuid_value = str(uuid.uuid4())[:8]

    for result in reductions:
        # Retain every instance receiving any assigned point. In class-aware
        # mode this includes control-only instances, whose expression row is
        # deliberately all zero.
        instance_ids = result.instance_ids
        expression_matrix = _counts_to_sparse(
            result.feature_counts,
            instance_ids=instance_ids,
            feature_axis=expression_feature_axis,
            cell_index_name=cell_index_name,
            feature_key=feature_key,
        )
        expression_matrices.append(expression_matrix)

        obs_index = [f"{instance}_{result.pair.labels_name}_{uuid_value}" for instance in instance_ids]
        obs = pd.DataFrame(index=pd.Index(obs_index, name=cell_index_name))
        obs[instance_key] = instance_ids.astype(int, copy=False)
        obs[region_key] = result.pair.labels_name
        if contract is not None:
            auxiliary_matrix = _counts_to_sparse(
                result.feature_counts,
                instance_ids=instance_ids,
                feature_axis=auxiliary_feature_axis,
                cell_index_name=cell_index_name,
                feature_key=feature_key,
            )
            auxiliary_matrices.append(auxiliary_matrix)

            class_counts = {
                contract.expression_class: _sparse_row_counts(expression_matrix),
                **{
                    feature_class: _sparse_row_counts(auxiliary_matrix[:, class_slice])
                    for feature_class, class_slice in contract.auxiliary_class_slices
                },
            }
            for feature_class, column_name in contract.count_columns:
                obs[column_name] = class_counts[feature_class]
            total_counts = sum(
                (class_counts[feature_class].astype(np.uint64) for feature_class in contract.panel.classes),
                start=np.zeros(len(instance_ids), dtype=np.uint64),
            )
            if np.any(total_counts == 0):
                raise RuntimeError("Retained aggregation rows must contain at least one assigned point.")
            control_counts = sum(
                (class_counts[feature_class].astype(np.uint64) for feature_class in contract.control_classes),
                start=np.zeros(len(instance_ids), dtype=np.uint64),
            )
            obs[_CONTROL_FRACTION_COLUMN] = control_counts / total_counts
        obs_frames.append(obs)

        centers = result.centers.reindex(instance_ids)
        if centers.isna().any(axis=None):
            raise ValueError(
                f"Label centers of mass are missing for retained instances in {result.pair.labels_name!r}."
            )
        coordinate_blocks.append(centers.loc[:, list(coordinate_columns)].to_numpy())

    X = sparse.vstack(expression_matrices, format="csr")
    obs = pd.concat(obs_frames, axis=0)
    labels_names = [result.pair.labels_name for result in reductions]
    obs[region_key] = pd.Categorical(obs[region_key], categories=labels_names)
    adata = AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(expression_feature_axis, name=feature_key)),
    )
    adata.obsm[spatial_key] = np.vstack(coordinate_blocks)

    if contract is not None:
        auxiliary_matrix = sparse.vstack(auxiliary_matrices, format="csr")
        adata.obsm[_AUXILIARY_FEATURE_MATRIX_KEY] = auxiliary_matrix
        adata.uns[_FEATURE_MATRICES_KEY] = {
            _AUXILIARY_FEATURE_MATRIX_KEY: {
                "schema_version": _FEATURE_MATRIX_SCHEMA_VERSION,
                "source_kind": _AGGREGATE_SOURCE_KIND,
                "feature_columns": list(auxiliary_feature_axis),
            }
        }
        adata.uns[_FEATURE_CLASS_AGGREGATION_KEY] = {
            "schema_version": 1,
            "source_kind": _AGGREGATE_SOURCE_KIND,
            "feature_key": contract.panel.feature_key,
            "feature_class_key": contract.panel.feature_class_key,
            "expression_class": contract.expression_class,
            "classes": list(contract.panel.classes),
            "control_class_denominators": dict(contract.control_class_denominators),
            "count_columns": dict(contract.count_columns),
            "control_fraction_column": _CONTROL_FRACTION_COLUMN,
            "auxiliary_feature_matrix_key": _AUXILIARY_FEATURE_MATRIX_KEY,
            "regions": {
                result.pair.labels_name: {
                    "points_element": result.pair.points_name,
                    "coordinate_system": result.pair.coordinate_system,
                }
                for result in reductions
            },
        }
    return adata


def _sparse_row_counts(matrix: sparse.spmatrix) -> np.ndarray:
    """Return checked dense ``uint32`` row sums without densifying ``matrix``.

    The sparse input remains sparse; only one sum per matrix row is
    materialized. SciPy accumulates ``uint32`` values as ``uint64``, allowing
    overflow to be checked before the result is converted back to the compact
    ``uint32`` count dtype.
    """
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(counts > np.iinfo(np.uint32).max):
        raise ValueError("Per-instance assigned-point counts exceed the uint32 output range.")
    return counts.astype(np.uint32, copy=False)


def _counts_to_sparse(
    counts: pd.Series,
    *,
    instance_ids: np.ndarray,
    feature_axis: tuple[str, ...],
    cell_index_name: str,
    feature_key: str,
) -> sparse.csr_matrix:
    """Align observed instance-feature counts to a CSR matrix.

    ``counts`` uses a two-level MultiIndex to represent the nonzero count for
    each observed ``(instance, feature)`` pair. This function maps the unique
    values in those levels to the requested output axes and then expands only
    their compact integer codes. A pair is omitted when either its instance is
    absent from ``instance_ids`` or its feature is absent from
    ``feature_axis``. The caller can therefore construct expression and
    auxiliary matrices from the same complete count reduction.

    Parameters
    ----------
    counts
        Nonzero point counts indexed by instance ID and feature identifier.
    instance_ids
        Sorted instance IDs defining the rows of the output matrix.
    feature_axis
        Ordered feature identifiers defining the columns of the output
        matrix.
    cell_index_name
        Name of the instance-ID level in ``counts.index``.
    feature_key
        Name of the feature level in ``counts.index``.

    Returns
    -------
    scipy.sparse.csr_matrix
        A ``uint32`` matrix aligned to ``instance_ids`` and ``feature_axis``.

    Examples
    --------
    Consider these observed counts:

    ==========  ===========  =====
    Instance    Feature      Count
    ==========  ===========  =====
    42          EPCAM        7
    42          Negative1    2
    51          VIM          3
    99          Negative1    1
    ==========  ===========  =====

    The requested output axes are ``instance_ids = [42, 51]`` and
    ``feature_axis = ("EPCAM", "VIM")``. The MultiIndex represents the table
    with unique levels and one integer code per observed pair::

        instance_level       = [42, 51, 99]
        instance_level_codes = [ 0,  0,  1,  2]
        feature_level        = ["EPCAM", "Negative1", "VIM"]
        feature_level_codes  = [      0,           1,     2,           1]

    Mapping the unique levels to the requested axes produces ``-1`` for
    values that should not appear in the output::

        row_lookup    = [0, 1, -1]
        column_lookup = [0, -1, 1]

    Indexing those lookup tables with the pair codes gives::

        row_codes    = [0,  0, 1, -1]
        column_codes = [0, -1, 1, -1]
        selected     = [T,  F, T,  F]

    The retained coordinate triplets are therefore ``(row=0, column=0,
    value=7)`` and ``(row=1, column=1, value=3)``. They form the matrix::

                  EPCAM  VIM
        instance
        42            7    0
        51            0    3

    ``Negative1`` is excluded because it is not on ``feature_axis``;
    instance 99 is excluded because it is not in ``instance_ids``.
    """
    if counts.empty:
        return sparse.csr_matrix((0, len(feature_axis)), dtype=np.uint32)
    instance_level, instance_level_codes = _count_index_level(counts, key=cell_index_name)
    feature_level, feature_level_codes = _count_index_level(counts, key=feature_key)
    code_dtype = np.int32 if max(len(instance_ids), len(feature_axis)) <= np.iinfo(np.int32).max else np.int64
    # Map each unique level value once; indexing these small lookup tables with
    # the existing MultiIndex codes avoids expanding repeated strings or IDs.
    row_lookup = pd.Index(instance_ids).get_indexer(instance_level).astype(code_dtype, copy=False)
    column_lookup = pd.Index(feature_axis).get_indexer(feature_level).astype(code_dtype, copy=False)
    row_codes = row_lookup[instance_level_codes]
    column_codes = column_lookup[feature_level_codes]
    selected = (row_codes >= 0) & (column_codes >= 0)
    return sparse.coo_matrix(
        (counts.to_numpy(dtype=np.uint32)[selected], (row_codes[selected], column_codes[selected])),
        shape=(len(instance_ids), len(feature_axis)),
        dtype=np.uint32,
    ).tocsr()


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
    chunks: str | tuple[int, ...] | int | None = 10000,
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
    chunks: str | tuple[int, ...] | int | None = 10000,
    to_coordinate_system: str = "global",
    name_x: str = "x",
    name_y: str = "y",
    name_z: str = "z",
    cell_index_name: str = _CELL_INDEX,
) -> DaskDataFrame:
    """Assign each spatially overlapping point to the corresponding nonzero label ID."""
    assert np.issubdtype(se.data.dtype, np.integer), "Only integer arrays are supported."
    assert name_y in ddf and name_x in ddf, f"Dask Dataframe must contain '{name_y}' and '{name_x}' columns."
    Coords = namedtuple("Coords", ["x0", "y0"])
    coords = Coords(*_get_translation(se, to_coordinate_system=to_coordinate_system))
    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)

    requested_value_keys = [value_key] if isinstance(value_key, str) else list(value_key)
    missing_value_keys = [key for key in requested_value_keys if key not in ddf.columns]
    if missing_value_keys:
        raise ValueError(f"Dask DataFrame does not contain requested value columns: {missing_value_keys}.")
    coordinate_keys = [name_x, name_y, name_z] if name_z in ddf.columns else [name_x, name_y]
    value_keys = list(dict.fromkeys([*coordinate_keys, *requested_value_keys]))

    ddf = ddf[value_keys]

    arr = se.data

    if chunks is not None:
        arr = arr.rechunk(chunks)
    else:
        arr = arr.rechunk(arr.chunksize)

    if arr.ndim == 2:
        arr = arr[None, ...]

    ddf[name_x] = ddf[name_x].round().astype(int)
    ddf[name_y] = ddf[name_y].round().astype(int)
    if name_z in ddf.columns:
        ddf[name_z] = ddf[name_z].round().astype(int)

    delayed_chunks = arr.to_delayed().flatten()

    # chunk info needed for querying
    chunk_info = []
    _chunks = arr.chunks

    # Iterate over each chunk and compute its coordinates and size, needed for query
    for i in range(delayed_chunks.shape[0]):
        z, y, x = np.unravel_index(i, [len(_chunks[0]), len(_chunks[1]), len(_chunks[2])])
        size = (_chunks[0][z], _chunks[1][y], _chunks[2][x])
        start_coords = (sum(_chunks[0][:z]), sum(_chunks[1][:y]), sum(_chunks[2][:x]))
        chunk_info.append((start_coords, size))

    log.info("Calculating cell counts.")

    @dask.delayed
    def _process_partition(_chunk, _chunk_info, ddf_partition):
        ddf_partition = ddf_partition.copy()

        z_start, y_start, x_start = _chunk_info[0]

        if name_z in ddf_partition.columns:
            z_coords = ddf_partition[name_z].values.astype(int) - z_start
        else:
            z_coords = 0

        y_coords = ddf_partition[name_y].values.astype(int) - (int(coords.y0) + y_start)
        x_coords = ddf_partition[name_x].values.astype(int) - (int(coords.x0) + x_start)

        ddf_partition.loc[:, cell_index_name] = _chunk[
            z_coords,
            y_coords,
            x_coords,
        ]

        return ddf_partition

    # Create a list to store delayed operations
    delayed_objects = []

    for _chunk, _chunk_info in zip(delayed_chunks, chunk_info, strict=True):
        # Query the partition lazily without computing it
        z_start, y_start, x_start = _chunk_info[0]
        _chunk_shape = _chunk_info[1]

        y_query = f"{y_start + coords.y0} <= {name_y} < {y_start + coords.y0 + _chunk_shape[1]}"
        x_query = f"{x_start + coords.x0} <= {name_x} < {x_start + coords.x0 + _chunk_shape[2]}"
        query = f"{y_query} and {x_query}"

        if name_z in ddf.columns:
            z_query = f"{z_start} <= {name_z} < {z_start + _chunk_shape[0]}"
            query = f"{z_query} and {query}"

        ddf_partition = ddf.query(query)
        delayed_partition = _process_partition(_chunk, _chunk_info, ddf_partition)
        delayed_objects.append(delayed_partition)

    # Combine the delayed partitions into a single Dask DataFrame
    combined_partitions = dd.from_delayed(delayed_objects)

    # remove background
    combined_partitions = combined_partitions[combined_partitions[cell_index_name] != 0]

    if drop_coordinates:
        combined_partitions = combined_partitions[[*requested_value_keys, cell_index_name]]

    return combined_partitions
