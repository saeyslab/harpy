# Canonical-center interoperability

This page documents the stable integration contract used to exchange
label-instance centers between Harpy and downstream packages such as
napari-harpy. It is intended for integration authors. Most Harpy users should
obtain canonical centers through `harpy.tb.aggregate_points`, or attach them to
an existing regions table with `harpy.tb.add_canonical_centers`, and do not need
to call the symbols below directly.

Canonical centers are stored as two coordinated AnnData components:

```text
adata.obsm["spatial_canonical"]
    dense float64 coordinates in fixed (z, y, x) order

adata.uns["spatial_coordinates"]["spatial_canonical"]
    schema, source, calculation and table-row coverage metadata
```

The public `harpy.table.canonical_centers` package is the supported import
boundary. Integration code should not import its private `_models`, `_schema`
or `_calculation` modules directly.

## Calculation and inspection

These operations inspect, calculate, read and validate canonical payloads. They
operate on an explicit table-to-label binding and do not provide the ordinary
high-level Harpy workflow.

```{eval-rst}

.. currentmodule:: harpy.table.canonical_centers

.. autosummary::
    :toctree: generated

    inspect_canonical_cache
    calculate_canonical_centers
    read_canonical_centers_from_cache
    validate_canonical_payload
```

## Storage contract

The constants and helpers below define the versioned representation persisted
in AnnData. They are intended for packages that create, update or validate the
same canonical payload.

```{eval-rst}

.. currentmodule:: harpy.table.canonical_centers

.. autosummary::
    :toctree: generated

    CANONICAL_OBSM_KEY
    SPATIAL_COORDINATES_KEY
    CANONICAL_AXES
    CANONICAL_SCHEMA_VERSION
    CANONICAL_ALGORITHM_VERSION
    build_canonical_source_signature
    build_canonical_region_binding
    build_canonical_cache_update_payload
    build_canonical_metadata
    canonical_metadata_to_storage
    parse_canonical_metadata
    build_instance_set_digest
```

## Typed integration models

These immutable models carry validated source identity, table-row binding,
cache state and calculation results across the package boundary.

```{eval-rst}

.. currentmodule:: harpy.table.canonical_centers

.. autosummary::
    :toctree: generated

    CanonicalSourceSignature
    CanonicalRegionBinding
    CanonicalRegionMetadata
    CanonicalMetadata
    CanonicalCacheState
    CanonicalMismatchCode
    CanonicalCacheMismatch
    CanonicalCacheReport
    CanonicalCacheUpdateAction
    CanonicalCacheUpdatePayload
    CanonicalCacheUpdateResult
    CanonicalCentersResult
```
