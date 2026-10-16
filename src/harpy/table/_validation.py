"""Validate recognized Harpy table contracts in their SpatialData context.

Table-validation responsibilities:

- ``harpy.table.io._write_validation`` checks AnnData structure and ordered
  identities for complete-table and component writes, without requiring the
  surrounding SpatialData object.
- ``validate_table()`` and its helpers check SpatialData annotation, recognized
  Harpy metadata and references to other elements, including canonical centers.

Annotation and observation-identity checks currently overlap between these
paths. Keep their common invariants consistent and reuse shared checks rather
than adding independent validation blocks to each writer. Add Harpy-specific
metadata checks to the existing table-validation helpers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._feature_panels import _require_mapping, _require_nonempty_string
from harpy.table._metadata import _FEATURE_MATRIX_SCHEMA_VERSION
from harpy.table.canonical_centers import (
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    validate_canonical_payload,
)
from harpy.utils._keys import _FEATURE_MATRICES_KEY


def validate_table(
    sdata: SpatialData,
    table_name: str,
    *,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
) -> None:
    """Validate a SpatialData table and recognized Harpy table metadata.

    The validator is read-only and independent from table construction. It
    checks what holds for every table:

    - SpatialData's table model;
    - the annotation, when present: the region and instance columns exist
      without missing values, ``(region, instance)`` pairs are unique, and the
      declared regions equal the observed ones and exist in ``sdata``;
    - every matrix registered in ``adata.uns[feature_matrices_key]``: it
      exists in ``adata.obsm`` under the same key with one row per
      observation, and its ``feature_columns`` name each of its columns once;
      and
    - the canonical centers, when present.

    A class-aware table from :func:`harpy.tb.aggregate_points` is validated as
    any other table: its non-expression class matrices are registered feature matrices. It is
    not validated against its feature panel, which is authoritative when the
    table is written, so the table may be subset or edited afterwards.
    Matrix values, storage format and dtype are not validation constraints
    because preprocessing may legitimately change them.

    When the canonical-center matrix or its metadata record is present, both
    must be present. Their fixed axes, dtype, shape, region coverage, instance
    digests and labels ``scale0`` source signatures are checked without
    recomputing centers or reading labels pixels.

    This function does not recompute point-to-label assignment, scan points
    partitions, inspect label pixels, repair metadata, or write to ``sdata``.

    Parameters
    ----------
    sdata
        SpatialData object containing the table and any source elements to
        which its metadata refers. It may be backed or in memory.
    table_name
        Name of the table element to validate.
    feature_matrices_key
        Key in ``adata.uns`` of the feature-matrix registry to validate. Each
        record ``adata.uns[feature_matrices_key][name]`` describes the matrix
        ``adata.obsm[name]``; the shared name is their only link. The default,
        ``"feature_matrices"``, is where :func:`harpy.tb.aggregate_points` and
        :func:`harpy.tb.add_feature_matrix` write their records. Only this
        registry is checked: ``adata.obsm`` matrices without a record, and
        registries under other keys, are not validated.

    Raises
    ------
    TypeError
        If the table or its instance identifiers violate SpatialData's model
        type requirements.
    ValueError
        If the table is missing or its annotation, registered feature matrices
        or canonical centers are inconsistent.

    Examples
    --------
    .. code-block:: python

        hp.tb.validate_table(sdata, "table")

        # A registry written by add_feature_matrix under a custom key.
        hp.tb.validate_table(sdata, "table", feature_matrices_key="custom_feature_matrices")
    """
    annotation = _validate_table_without_canonical(sdata, table_name, feature_matrices_key=feature_matrices_key)
    adata = sdata.tables[table_name]
    has_canonical_component = CANONICAL_OBSM_KEY in adata.obsm or (
        isinstance(adata.uns.get(SPATIAL_COORDINATES_KEY), Mapping)
        and CANONICAL_OBSM_KEY in adata.uns[SPATIAL_COORDINATES_KEY]
    )
    if annotation is None:
        if has_canonical_component:
            raise ValueError("Canonical centers require a SpatialData table annotation.")
    else:
        region_key, instance_key, regions = annotation
        validate_canonical_payload(
            sdata,
            adata,
            table_name=table_name,
            region_key=region_key,
            instance_key=instance_key,
            regions=regions,
        )


def _validate_table_without_canonical(
    sdata: SpatialData,
    table_name: str,
    *,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Validate the SpatialData table model, annotation and registered feature matrices.

    SpatialData's model is checked before the Harpy-specific contracts. This
    deliberately partial validator excludes ``obsm["spatial_canonical"]`` and
    its matching metadata record. Successful return does not establish complete
    table validity: canonical components must be validated separately.
    ``feature_matrices_key`` selects the one feature-matrix registry that is
    checked, as in :func:`validate_table`.

    Previously validated contracts need not be checked again when the data and
    metadata they cover remain unchanged.

    Returns
    -------
    tuple[str, str, tuple[str, ...]] | None
        The normalized ``(region_key, instance_key, regions)`` annotation, or
        ``None`` when the table is not a SpatialData regions table.
    """
    if not isinstance(sdata, SpatialData):
        raise ValueError(f"Parameter 'sdata' must be a SpatialData object, found {type(sdata).__name__}.")
    if not isinstance(table_name, str) or not table_name:
        raise ValueError(f"Parameter 'table_name' must be a non-empty string, found {table_name!r}.")
    if not isinstance(feature_matrices_key, str) or not feature_matrices_key:
        raise ValueError(
            f"Parameter 'feature_matrices_key' must be a non-empty string, found {feature_matrices_key!r}."
        )
    if table_name not in sdata.tables:
        raise ValueError(f"Table element {table_name!r} is not present in 'sdata.tables'.")

    adata = sdata.tables[table_name]
    TableModel.validate(adata)
    annotation = _validate_table_annotation(sdata, adata, table_name=table_name)
    _validate_feature_matrices(adata, feature_matrices_key=feature_matrices_key)
    return annotation


def _validate_table_annotation(
    sdata: SpatialData,
    adata: AnnData,
    *,
    table_name: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Validate the annotation, requiring declared and observed region sets to match."""
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
    raw_regions = annotation.get(TableModel.REGION_KEY)
    regions = _metadata_string_sequence(
        [raw_regions] if isinstance(raw_regions, str) else raw_regions,
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
    unobserved = sorted(set(regions) - observed_regions)
    if unobserved:
        raise ValueError(f"Table {table_name!r} declares regions without observations: {unobserved!r}.")
    missing_elements = sorted(region for region in regions if region not in sdata)
    if missing_elements:
        raise ValueError(f"Table {table_name!r} annotation references missing spatial elements: {missing_elements!r}.")
    return region_key, instance_key, regions


def _validate_feature_matrices(adata: AnnData, *, feature_matrices_key: str) -> Mapping[str, object]:
    """Validate generic ``obsm`` feature-matrix registrations.

    Each entry in ``adata.uns[feature_matrices_key]`` registers one matrix under
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
    without invalidating the registered feature axis. Nor is the meaning of the
    columns: the non-expression class matrices of :func:`harpy.tb.aggregate_points` are
    checked like any other registered matrix, not against a feature panel. If
    no registry exists, an empty mapping is returned.

    Returns
    -------
    Mapping[str, object]
        The validated registry.
    """
    value = adata.uns.get(feature_matrices_key)
    if value is None:
        return {}
    records = _require_mapping(value, path=feature_matrices_key)
    for matrix_name, record_value in records.items():
        matrix_name = _require_nonempty_string(matrix_name, path=f"{feature_matrices_key} key")
        record = _require_mapping(record_value, path=f"{feature_matrices_key}.{matrix_name}")
        if matrix_name not in adata.obsm:
            raise ValueError(f"Feature-matrix metadata references missing adata.obsm[{matrix_name!r}].")
        matrix = adata.obsm[matrix_name]
        if len(matrix.shape) != 2 or matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"Registered feature matrix {matrix_name!r} must have shape (n_obs, n_features), found {matrix.shape}."
            )
        columns = _metadata_string_sequence(
            record.get("feature_columns"),
            path=f"{feature_matrices_key}.{matrix_name}.feature_columns",
        )
        if len(columns) != matrix.shape[1]:
            raise ValueError(
                f"Registered feature matrix {matrix_name!r} has {matrix.shape[1]} columns but its metadata "
                f"declares {len(columns)} feature columns."
            )
        version = record.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != _FEATURE_MATRIX_SCHEMA_VERSION:
            raise ValueError(
                f"Feature-matrix metadata {feature_matrices_key}.{matrix_name}.schema_version must equal "
                f"{_FEATURE_MATRIX_SCHEMA_VERSION}, found {version!r}."
            )
        _require_nonempty_string(
            record.get("source_kind"),
            path=f"{feature_matrices_key}.{matrix_name}.source_kind",
        )
    return records


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
