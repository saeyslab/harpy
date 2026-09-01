from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation

from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _HARPY_METADATA_KEY,
    _METADATA_VERSION,
    _METADATA_VERSION_KEY,
    _POINTS_METADATA_KEY,
)
from harpy.table._allocation import (
    _FeatureClassAggregationContract,
    _FeaturePanelContract,
    _parse_feature_panel,
    _require_mapping,
    _require_nonempty_string,
    _validate_feature_class_dtype,
)
from harpy.table._metadata import (
    _AGGREGATE_POINTS_SOURCE_KIND,
    _AUXILIARY_FEATURE_MATRIX_KEY,
    _AUXILIARY_POINTS_FRACTION_COLUMN,
    _FEATURE_CLASS_AGGREGATION_KEY,
    _FEATURE_CLASS_AGGREGATION_SCHEMA_VERSION,
    _FEATURE_MATRIX_SCHEMA_VERSION,
)
from harpy.utils._keys import _FEATURE_MATRICES_KEY


def validate_table(sdata: SpatialData, table_name: str) -> None:
    """Validate a SpatialData table and recognized Harpy table metadata.

    The validator is read-only and independent from table construction. It
    checks a SpatialData table annotation when present, including the
    registered regions and unique ``(region, instance)`` observation keys, and
    validates every matrix registered in ``adata.uns["feature_matrices"]``
    against its corresponding ``adata.obsm`` value.

    For a class-aware table created by :func:`harpy.tb.aggregate_points`, it
    additionally follows each region's points-element reference through::

        adata.uns["feature_class_aggregation"]["regions"]
            -> sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
            -> sdata.attrs["harpy"]["feature_panels"][panel_name]

    The root feature panel is authoritative. Its feature keys, class definitions,
    and auxiliary class feature counts must agree with the table-local metadata.
    The current expression and auxiliary feature axes may be filtered or
    reordered, but every retained feature must still belong to the appropriate
    panel class. Metadata-declared summary columns must remain present in
    ``adata.obs``, but their values and dtypes are not validated because these
    derived summaries may be recalculated downstream. Matrix storage format and
    dtype are likewise not validation constraints because preprocessing may
    legitimately change either representation.

    This function does not recompute point-to-label assignment, scan points
    partitions, inspect label pixels, repair metadata, or write to ``sdata``.

    Parameters
    ----------
    sdata
        SpatialData object containing the table and any source elements to
        which its metadata refers. It may be backed or in memory.
    table_name
        Name of the table element to validate.

    Raises
    ------
    ValueError
        If the table is missing or its annotation, registered feature matrices,
        source references, feature-panel metadata, matrices, or summaries are
        inconsistent.

    Examples
    --------
    .. code-block:: python

        hp.tb.validate_table(sdata, "table")
    """
    if not isinstance(sdata, SpatialData):
        raise ValueError(f"Parameter 'sdata' must be a SpatialData object, found {type(sdata).__name__}.")
    if not isinstance(table_name, str) or not table_name:
        raise ValueError(f"Parameter 'table_name' must be a non-empty string, found {table_name!r}.")
    if table_name not in sdata.tables:
        raise ValueError(f"Table element {table_name!r} is not present in 'sdata.tables'.")

    adata = sdata.tables[table_name]
    annotation = _validate_table_annotation(sdata, adata, table_name=table_name)
    feature_matrices = _validate_feature_matrices(adata)

    aggregation_metadata = adata.uns.get(_FEATURE_CLASS_AGGREGATION_KEY)
    if aggregation_metadata is None:
        if _AUXILIARY_FEATURE_MATRIX_KEY in adata.obsm:
            raise ValueError(
                f"Table {table_name!r} contains {_AUXILIARY_FEATURE_MATRIX_KEY!r} without "
                f"{_FEATURE_CLASS_AGGREGATION_KEY!r} metadata."
            )
        return

    if annotation is None:
        raise ValueError("Feature-class aggregation metadata requires a SpatialData table annotation.")
    _, instance_key, regions = annotation

    _validate_feature_class_aggregation(
        sdata,
        adata,
        aggregation_metadata=_require_mapping(aggregation_metadata, path=_FEATURE_CLASS_AGGREGATION_KEY),
        feature_matrices=feature_matrices,
        instance_key=instance_key,
        annotated_regions=regions,
    )


def _validate_table_annotation(
    sdata: SpatialData,
    adata: AnnData,
    *,
    table_name: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Validate and normalize the SpatialData annotation of one table."""
    if TableModel.ATTRS_KEY not in adata.uns:
        return None
    annotation = _require_mapping(
        adata.uns.get(TableModel.ATTRS_KEY),
        path=f"tables.{table_name}.uns.{TableModel.ATTRS_KEY}",
    )
    region_key = _require_nonempty_string(
        annotation.get(TableModel.REGION_KEY_KEY),
        path=f"tables.{table_name}.uns.{TableModel.ATTRS_KEY}.{TableModel.REGION_KEY_KEY}",
    )
    instance_key = _require_nonempty_string(
        annotation.get(TableModel.INSTANCE_KEY),
        path=f"tables.{table_name}.uns.{TableModel.ATTRS_KEY}.{TableModel.INSTANCE_KEY}",
    )
    regions = _metadata_string_sequence(
        annotation.get(TableModel.REGION_KEY),
        path=f"tables.{table_name}.uns.{TableModel.ATTRS_KEY}.{TableModel.REGION_KEY}",
    )

    for column in (region_key, instance_key):
        if column not in adata.obs:
            raise ValueError(f"Table {table_name!r} annotation references missing observation column {column!r}.")
    if adata.obs[[region_key, instance_key]].isna().any(axis=None):
        raise ValueError(f"Table {table_name!r} contains missing region or instance identifiers.")
    if adata.obs.duplicated(subset=[region_key, instance_key]).any():
        raise ValueError(f"Table {table_name!r} contains duplicate (region, instance) observation keys.")

    observed_regions = set(adata.obs[region_key].astype("str").unique())
    undeclared = sorted(observed_regions - set(regions))
    if undeclared:
        raise ValueError(f"Table {table_name!r} contains regions absent from its annotation: {undeclared!r}.")
    missing_elements = sorted(region for region in regions if region not in sdata)
    if missing_elements:
        raise ValueError(f"Table {table_name!r} annotation references missing spatial elements: {missing_elements!r}.")
    return region_key, instance_key, regions


def _validate_feature_matrices(adata: AnnData) -> Mapping[str, object]:
    """Validate generic ``obsm`` feature-matrix registrations.

    Each entry in ``adata.uns["feature_matrices"]`` registers one matrix under
    the same key in ``adata.obsm``. For every registered matrix, this function
    checks that:

    - the registry key is a non-empty string and its record is a mapping;
    - the corresponding ``adata.obsm`` matrix exists;
    - the matrix is two-dimensional and has one row per observation;
    - ``feature_columns`` is a non-empty sequence of unique strings whose
      length equals the matrix's second dimension;
    - ``schema_version`` is the supported feature-matrix schema version; and
    - ``source_kind`` is a non-empty string.

    Matrix values, storage backend, sparse format and dtype are deliberately
    not checked because downstream preprocessing may legitimately change them
    without invalidating the registered feature axis. If no registry exists,
    an empty mapping is returned.

    The validated mapping is reused by the source-specific validation stage::

        adata.uns["feature_matrices"]
                    |
                    v
        _validate_feature_matrices()
          validates generic matrix contracts
                    |
                    v
        feature_matrices
                    |
                    v
        _validate_feature_class_aggregation()
          selects "auxiliary_feature_counts"
          and validates its aggregation contract

    Returns
    -------
    Mapping[str, object]
        The validated registry, for reuse by source-specific validators.
    """
    value = adata.uns.get(_FEATURE_MATRICES_KEY)
    if value is None:
        return {}
    records = _require_mapping(value, path=_FEATURE_MATRICES_KEY)
    for matrix_name, record_value in records.items():
        matrix_name = _require_nonempty_string(matrix_name, path=f"{_FEATURE_MATRICES_KEY} key")
        record = _require_mapping(record_value, path=f"{_FEATURE_MATRICES_KEY}.{matrix_name}")
        if matrix_name not in adata.obsm:
            raise ValueError(f"Feature-matrix metadata references missing adata.obsm[{matrix_name!r}].")
        matrix = adata.obsm[matrix_name]
        if len(matrix.shape) != 2 or matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"Registered feature matrix {matrix_name!r} must have shape (n_obs, n_features), found {matrix.shape}."
            )
        columns = _metadata_string_sequence(
            record.get("feature_columns"),
            path=f"{_FEATURE_MATRICES_KEY}.{matrix_name}.feature_columns",
        )
        if len(columns) != matrix.shape[1]:
            raise ValueError(
                f"Registered feature matrix {matrix_name!r} has {matrix.shape[1]} columns but its metadata "
                f"declares {len(columns)} feature columns."
            )
        version = record.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != _FEATURE_MATRIX_SCHEMA_VERSION:
            raise ValueError(
                f"Feature-matrix metadata {_FEATURE_MATRICES_KEY}.{matrix_name}.schema_version must equal "
                f"{_FEATURE_MATRIX_SCHEMA_VERSION}, found {version!r}."
            )
        _require_nonempty_string(
            record.get("source_kind"),
            path=f"{_FEATURE_MATRICES_KEY}.{matrix_name}.source_kind",
        )
    return records


def _validate_feature_class_aggregation(
    sdata: SpatialData,
    adata: AnnData,
    *,
    aggregation_metadata: Mapping[str, object],
    feature_matrices: Mapping[str, object],
    instance_key: str,
    annotated_regions: tuple[str, ...],
) -> None:
    """Validate one persisted class-aware point-aggregation contract.

    Validation follows the source references recorded by the aggregation::

        aggregation region
            -> labels element + points element + coordinate system
            -> points element's root feature panel
            -> table-local aggregation metadata and feature axes

    Specifically, this helper requires the aggregation regions to match the
    SpatialData table annotation, verifies that every referenced labels and
    points element exists in the recorded coordinate system, and resolves the
    authoritative feature panel for every region. All regions contributing to
    one table must resolve to equivalent panel contracts. The table-local
    feature keys, classes, expression class, auxiliary class feature counts,
    count-column
    names, and auxiliary-matrix registration must agree with that panel.

    The expression matrix may contain any unique, reordered subset of the
    panel's expression features. Likewise, the auxiliary matrix may contain a
    reordered subset of the panel's non-expression features. Instance IDs must
    be positive integers, and all metadata-declared summary columns must remain
    present in ``adata.obs``.

    This is a structural and metadata-consistency check. It does not rescan the
    source points against the panel, inspect label pixels, recompute
    point-to-label assignment, or compare the numerical matrix and summary
    values with a newly calculated aggregation.

    Parameters
    ----------
    sdata
        SpatialData object containing the table's referenced source elements
        and root Harpy feature-panel metadata.
    adata
        Class-aware aggregation table being validated.
    aggregation_metadata
        Parsed ``adata.uns["feature_class_aggregation"]`` record.
    feature_matrices
        Generic feature-matrix records already validated against ``adata.obsm``.
    instance_key
        Name of the validated instance-identifier column in ``adata.obs``.
    annotated_regions
        Regions declared by the already validated SpatialData table annotation.
    """
    regions = _require_mapping(
        aggregation_metadata.get("regions"),
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.regions",
    )
    if not regions:
        raise ValueError("Feature-class aggregation metadata must reference at least one region.")
    if set(regions) != set(annotated_regions):
        raise ValueError("Feature-class aggregation regions disagree with the SpatialData table annotation.")

    panels = tuple(
        _resolve_region_panel(
            sdata,
            labels_name=labels_name,
            record_value=record,
        )
        for labels_name, record in regions.items()
    )
    panel = panels[0]
    # One combined AnnData table has one expression axis, auxiliary axis, and
    # class schema. Region-specific panel names may differ, but their normalized
    # contracts must therefore be identical.
    if any(candidate != panel for candidate in panels[1:]):
        raise ValueError("Feature-class aggregation regions reference incompatible feature panels.")

    expression_class = _require_nonempty_string(
        aggregation_metadata.get("expression_class"),
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.expression_class",
    )
    contract = _FeatureClassAggregationContract(panel=panel, expression_class=expression_class)
    _validate_feature_class_metadata(aggregation_metadata, contract=contract)

    if adata.X is None:
        raise ValueError("Class-aware aggregation requires an expression matrix in adata.X.")
    if adata.var_names.name != panel.feature_key:
        raise ValueError(f"The expression feature index must be named {panel.feature_key!r}.")
    if not adata.var_names.is_unique:
        raise ValueError("The expression feature index must contain unique values.")
    unexpected_expression_features = sorted(set(adata.var_names) - set(contract.expression_feature_axis))
    if unexpected_expression_features:
        raise ValueError(
            "The expression matrix contains features outside the panel's expression class: "
            f"{unexpected_expression_features!r}."
        )

    auxiliary_key = aggregation_metadata.get("auxiliary_feature_matrix_key")
    if auxiliary_key != _AUXILIARY_FEATURE_MATRIX_KEY:
        raise ValueError(
            f"Feature-class aggregation metadata 'auxiliary_feature_matrix_key' must equal "
            f"{_AUXILIARY_FEATURE_MATRIX_KEY!r}, found {auxiliary_key!r}."
        )
    if auxiliary_key not in adata.obsm:
        raise ValueError(f"Class-aware aggregation is missing adata.obsm[{auxiliary_key!r}].")
    feature_matrix_record = _require_mapping(
        feature_matrices.get(auxiliary_key),
        path=f"{_FEATURE_MATRICES_KEY}.{auxiliary_key}",
    )
    expected_record = {
        "schema_version": _FEATURE_MATRIX_SCHEMA_VERSION,
        "source_kind": _AGGREGATE_POINTS_SOURCE_KIND,
    }
    if any(feature_matrix_record.get(key) != expected for key, expected in expected_record.items()):
        raise ValueError("Auxiliary feature-matrix metadata disagrees with the aggregation contract.")
    feature_columns = _metadata_string_sequence(
        feature_matrix_record.get("feature_columns"),
        path=f"{_FEATURE_MATRICES_KEY}.{auxiliary_key}.feature_columns",
    )
    unexpected_auxiliary_features = sorted(set(feature_columns) - set(contract.auxiliary_feature_axis))
    if unexpected_auxiliary_features:
        raise ValueError(
            "The auxiliary matrix contains features outside the panel's non-expression classes: "
            f"{unexpected_auxiliary_features!r}."
        )

    instances = adata.obs[instance_key]
    if not pd.api.types.is_integer_dtype(instances.dtype) or (instances <= 0).any():
        raise ValueError("Class-aware aggregation instance identifiers must be positive integers.")
    required_summary_columns = [column for _, column in contract.count_columns]
    required_summary_columns.append(_AUXILIARY_POINTS_FRACTION_COLUMN)
    missing_summary_columns = sorted(set(required_summary_columns) - set(adata.obs))
    if missing_summary_columns:
        raise ValueError(
            f"Feature-class aggregation metadata references missing observation columns: {missing_summary_columns!r}."
        )


def _resolve_region_panel(
    sdata: SpatialData,
    *,
    labels_name: object,
    record_value: object,
) -> _FeaturePanelContract:
    """Resolve one aggregation region to its root feature-panel contract."""
    labels_name = _require_nonempty_string(labels_name, path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.regions key")
    if labels_name not in sdata.labels:
        raise ValueError(f"Feature-class aggregation references missing labels element {labels_name!r}.")
    record = _require_mapping(
        record_value,
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.regions.{labels_name}",
    )
    points_name = _require_nonempty_string(
        record.get("points_element"),
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.regions.{labels_name}.points_element",
    )
    coordinate_system = _require_nonempty_string(
        record.get("coordinate_system"),
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.regions.{labels_name}.coordinate_system",
    )
    if points_name not in sdata.points:
        raise ValueError(f"Feature-class aggregation references missing points element {points_name!r}.")
    for element_name, element in ((labels_name, sdata.labels[labels_name]), (points_name, sdata.points[points_name])):
        if coordinate_system not in get_transformation(element, get_all=True):
            raise ValueError(
                f"Source element {element_name!r} is not registered in coordinate system {coordinate_system!r}."
            )

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
    points_record = _require_mapping(
        points_registry.get(points_name),
        path=f"{_HARPY_METADATA_KEY}.{_POINTS_METADATA_KEY}.{points_name}",
    )
    panel_name = _require_nonempty_string(
        points_record.get("feature_panel"),
        path=f"{_HARPY_METADATA_KEY}.{_POINTS_METADATA_KEY}.{points_name}.feature_panel",
    )
    panel_registry = _require_mapping(
        harpy_metadata.get(_FEATURE_PANELS_METADATA_KEY),
        path=f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}",
    )
    panel_record = _require_mapping(
        panel_registry.get(panel_name),
        path=f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}.{panel_name}",
    )
    panel = _parse_feature_panel(panel_record, panel_name=panel_name)
    points = sdata.points[points_name]
    for column in (panel.feature_key, panel.feature_class_key):
        if column not in points.columns:
            raise ValueError(f"Points element {points_name!r} is missing panel-declared column {column!r}.")
    _validate_feature_class_dtype(points, points_name=points_name, panel=panel)
    return panel


def _validate_feature_class_metadata(
    aggregation_metadata: Mapping[str, object],
    *,
    contract: _FeatureClassAggregationContract,
) -> None:
    """Compare table-local aggregation metadata with a root panel contract.

    Auxiliary class feature counts are not inferred from observed points. They
    are derived from the complete authoritative panel in root SpatialData
    metadata and compared with the values persisted in the AnnData table::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
                                      |
                                      v
        sdata.attrs["harpy"]["feature_panels"][panel_name]
                                      |
                                      v
                              features_by_class
                                      |
                                      v
               len(features_by_class[auxiliary_class])
                                      |
                                      v
        adata.uns["feature_class_aggregation"]
                 ["auxiliary_class_feature_counts"]

    Consequently, panel features with zero observed points still contribute to
    their auxiliary class's recorded feature count.
    """
    expected = {
        "schema_version": _FEATURE_CLASS_AGGREGATION_SCHEMA_VERSION,
        "source_kind": _AGGREGATE_POINTS_SOURCE_KIND,
        "feature_key": contract.panel.feature_key,
        "feature_class_key": contract.panel.feature_class_key,
        "expression_class": contract.expression_class,
        "auxiliary_points_fraction_column": _AUXILIARY_POINTS_FRACTION_COLUMN,
        "auxiliary_feature_matrix_key": _AUXILIARY_FEATURE_MATRIX_KEY,
    }
    for key, expected_value in expected.items():
        if aggregation_metadata.get(key) != expected_value:
            raise ValueError(
                f"Feature-class aggregation metadata {key!r} must equal {expected_value!r}, "
                f"found {aggregation_metadata.get(key)!r}."
            )
    classes = _metadata_string_sequence(
        aggregation_metadata.get("classes"),
        path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.classes",
    )
    if classes != contract.panel.classes:
        raise ValueError("Feature-class aggregation classes disagree with the authoritative feature panel.")
    auxiliary_class_feature_counts = dict(
        _require_mapping(
            aggregation_metadata.get("auxiliary_class_feature_counts"),
            path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.auxiliary_class_feature_counts",
        )
    )
    if auxiliary_class_feature_counts != dict(contract.auxiliary_class_feature_counts):
        raise ValueError(
            "Feature-class aggregation auxiliary class feature counts disagree with the authoritative feature panel."
        )
    count_columns = dict(
        _require_mapping(
            aggregation_metadata.get("count_columns"),
            path=f"{_FEATURE_CLASS_AGGREGATION_KEY}.count_columns",
        )
    )
    if count_columns != dict(contract.count_columns):
        raise ValueError("Feature-class aggregation count columns disagree with the authoritative feature panel.")


def _metadata_string_sequence(value: object, *, path: str) -> tuple[str, ...]:
    """Normalize a one-dimensional AnnData metadata sequence of unique strings."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"Harpy table metadata {path} must be one-dimensional.")
        value = value.tolist()
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"Harpy table metadata {path} must be a non-empty sequence of strings.")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"Harpy table metadata {path} must contain unique non-empty strings.")
    return result
