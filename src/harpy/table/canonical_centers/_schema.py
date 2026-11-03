from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from scipy.sparse import issparse
from spatialdata.models import get_table_keys

from harpy.image import get_dataarray
from harpy.table.canonical_centers._models import (
    CanonicalCacheMismatch,
    CanonicalCacheReport,
    CanonicalCacheUpdatePayload,
    CanonicalMetadata,
    CanonicalMismatchCode,
    CanonicalRegionBinding,
    CanonicalRegionMetadata,
    CanonicalSourceSignature,
)
from harpy.utils._names import _names

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

CANONICAL_OBSM_KEY = "spatial_canonical"
SPATIAL_COORDINATES_KEY = "spatial_coordinates"
CANONICAL_SCHEMA_VERSION = 2
CANONICAL_ALGORITHM_VERSION = 1
CANONICAL_AXES = ("z", "y", "x")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "axes",
    "regions",
}
_REGION_KEYS = {
    "source_element_type",
    "source_scale",
    "coordinate_frame",
    "algorithm_version",
    "source",
}
_SUPPORTED_SOURCE_DIMS = {("y", "x"), ("z", "y", "x")}


class _UnsupportedSchemaError(ValueError):
    pass


class _TopLevelContractError(ValueError):
    pass


class _RegionMetadataError(ValueError):
    pass


def build_canonical_source_signature(sdata: SpatialData, labels_name: str) -> CanonicalSourceSignature:
    """Build the structural ``scale0`` signature without reading labels pixels."""
    if not isinstance(labels_name, str) or not labels_name:
        raise ValueError("Labels name must be a non-empty string.")
    if labels_name not in sdata.labels:
        raise ValueError(f"Labels element `{labels_name}` is not available in the selected SpatialData object.")

    scale0 = get_dataarray(sdata, labels_name, scale="scale0")
    dims = tuple(str(dim) for dim in scale0.dims)
    shape = tuple(int(size) for size in scale0.shape)
    try:
        dtype = np.dtype(scale0.dtype)
    except TypeError as exc:
        raise ValueError(f"Labels element `{labels_name}` exposes an unsupported dtype.") from exc
    if dtype.kind not in "iu":
        raise ValueError(f"Labels element `{labels_name}` must use an integer dtype.")
    if dims not in _SUPPORTED_SOURCE_DIMS:
        raise ValueError("Canonical centers require source dims (`y`, `x`) or (`z`, `y`, `x`).")

    return CanonicalSourceSignature(
        labels_name=labels_name,
        source_scale="scale0",
        dims=dims,
        shape=shape,
        dtype=dtype.name,
    )


def build_canonical_region_binding(
    table: AnnData,
    *,
    table_name: str,
    labels_name: str,
    region_key: str,
    instance_key: str,
    regions: Sequence[str] | None = None,
) -> CanonicalRegionBinding:
    """Validate and capture one labels region's current table-row binding."""
    if regions is not None and labels_name not in regions:
        raise ValueError(f"Table `{table_name}` does not declare labels region `{labels_name}`.")
    for key in (region_key, instance_key):
        if key not in table.obs.columns:
            raise ValueError(f"Table `{table_name}` is missing required obs column `{key}`.")

    mask = np.asarray(table.obs[region_key] == labels_name, dtype=bool)
    row_positions = np.flatnonzero(mask).astype(np.intp, copy=False)
    if len(row_positions) == 0:
        raise ValueError(f"Table `{table_name}` contains no rows for labels region `{labels_name}`.")

    return CanonicalRegionBinding(
        table_name=table_name,
        labels_name=labels_name,
        region_key=region_key,
        instance_key=instance_key,
        row_positions=row_positions,
        instance_ids=table.obs.iloc[row_positions][instance_key].to_numpy(),
    )


def build_canonical_metadata(
    *,
    regions: Mapping[str, CanonicalRegionMetadata],
    schema_version: int = CANONICAL_SCHEMA_VERSION,
) -> CanonicalMetadata:
    """Build typed canonical metadata for storage."""
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Only canonical schema version {CANONICAL_SCHEMA_VERSION} can be built.")
    for region_metadata in regions.values():
        _validate_source(region_metadata.source_signature)
    return CanonicalMetadata(schema_version=schema_version, regions=regions)


def canonical_metadata_to_storage(metadata: CanonicalMetadata) -> dict[str, object]:
    """Serialize typed metadata using AnnData-Zarr-compatible values."""
    if metadata.schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical schema version {metadata.schema_version}.")
    regions: dict[str, object] = {}
    for region, region_metadata in metadata.regions.items():
        source = region_metadata.source_signature
        regions[region] = {
            "source_element_type": "labels",
            "source_scale": "scale0",
            "coordinate_frame": {
                "type": "element_intrinsic",
                "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
            },
            "algorithm_version": region_metadata.algorithm_version,
            "source": {
                "dims": list(source.dims),
                "shape": list(source.shape),
                "dtype": source.dtype,
            },
        }

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "axes": list(CANONICAL_AXES),
        "regions": regions,
    }


def parse_canonical_metadata(value: object) -> CanonicalMetadata:
    """Strictly parse the canonical storage representation."""
    mapping = _require_mapping(value, "canonical metadata")
    if "schema_version" not in mapping:
        raise _TopLevelContractError("canonical metadata is missing `schema_version`.")
    schema_version = _require_integer(mapping["schema_version"], "schema_version")
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise _UnsupportedSchemaError(
            f"Unsupported canonical schema version {schema_version}; Harpy reads version "
            f"{CANONICAL_SCHEMA_VERSION}. Rebuild the canonical centers with "
            "hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)."
        )
    _require_exact_keys(mapping, _TOP_LEVEL_KEYS, "canonical metadata", _TopLevelContractError)

    _require_string_sequence(mapping["axes"], CANONICAL_AXES, "axes", _TopLevelContractError)
    raw_regions = _require_mapping(mapping["regions"], "regions", _RegionMetadataError)

    regions: dict[str, CanonicalRegionMetadata] = {}
    for raw_region, raw_entry in raw_regions.items():
        if not isinstance(raw_region, str) or not raw_region:
            raise _RegionMetadataError("Region names must be non-empty strings.")
        regions[raw_region] = _parse_region_metadata(raw_region, raw_entry)
    return CanonicalMetadata(schema_version=schema_version, regions=regions)


def inspect_canonical_cache(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str,
) -> CanonicalCacheReport:
    """Inspect one labels region's canonical payload without mutating it."""
    if table_name not in sdata.tables:
        raise ValueError(f"Table `{table_name}` is not available in the selected SpatialData object.")
    table = sdata.tables[table_name]
    regions, region_key, instance_key = get_table_keys(table)
    regions = _names(regions)
    source_signature = build_canonical_source_signature(sdata, labels_name)
    binding = build_canonical_region_binding(
        table,
        table_name=table_name,
        labels_name=labels_name,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )

    matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    metadata_exists = isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry
    if not matrix_exists and not metadata_exists:
        return _report(None, source_signature, binding)
    if matrix_exists and not metadata_exists:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.MATRIX_WITHOUT_METADATA),
        )
    if metadata_exists and not matrix_exists:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.METADATA_WITHOUT_MATRIX),
        )

    matrix = _validate_canonical_matrix(table.obsm[CANONICAL_OBSM_KEY], table.n_obs)
    if isinstance(matrix, str):
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.MATRIX_INVALID, matrix),
        )
    try:
        stored_metadata = parse_canonical_metadata(registry[CANONICAL_OBSM_KEY])  # type: ignore[index]
    except _UnsupportedSchemaError as exc:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.SCHEMA_VERSION_UNSUPPORTED, str(exc)),
        )
    except _TopLevelContractError as exc:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH, str(exc)),
        )
    except _RegionMetadataError as exc:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.REGION_METADATA_INVALID, str(exc)),
        )
    except (TypeError, ValueError, KeyError) as exc:
        return _report(
            None,
            source_signature,
            binding,
            _all_regions_mismatch(CanonicalMismatchCode.METADATA_INVALID, str(exc)),
        )

    if stored_metadata.region_key != region_key or stored_metadata.instance_key != instance_key:
        return _report(
            stored_metadata,
            source_signature,
            binding,
            _all_regions_mismatch(
                CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH,
                "Canonical linkage keys do not match the current SpatialData table metadata.",
            ),
        )

    region_metadata = stored_metadata.regions.get(labels_name)
    if region_metadata is None:
        return _report(
            stored_metadata,
            source_signature,
            binding,
            _region_mismatch(CanonicalMismatchCode.REGION_NOT_REGISTERED, labels_name),
        )

    mismatches: list[CanonicalCacheMismatch] = []
    if region_metadata.source_signature != source_signature:
        mismatches.append(_region_mismatch(CanonicalMismatchCode.SOURCE_SIGNATURE_MISMATCH, labels_name))
    if region_metadata.n_obs != binding.n_obs or region_metadata.instance_set_digest != binding.instance_set_digest:
        mismatches.append(_region_mismatch(CanonicalMismatchCode.TABLE_SIGNATURE_MISMATCH, labels_name))
    if region_metadata.algorithm_version != CANONICAL_ALGORITHM_VERSION:
        mismatches.append(_region_mismatch(CanonicalMismatchCode.ALGORITHM_VERSION_MISMATCH, labels_name))
    region_centers = matrix[binding.row_positions]
    if not np.isfinite(region_centers).all() or (
        source_signature.dims == ("y", "x") and np.any(region_centers[:, 0] != 0.0)
    ):
        mismatches.append(_region_mismatch(CanonicalMismatchCode.REGION_COORDINATES_INVALID, labels_name))
    return _report(stored_metadata, source_signature, binding, *mismatches)


def build_canonical_cache_update_payload(
    *,
    binding: CanonicalRegionBinding,
    centers: object,
    source_signature: CanonicalSourceSignature,
) -> CanonicalCacheUpdatePayload:
    """Validate calculated centers and capture an immutable update payload."""
    return CanonicalCacheUpdatePayload(
        binding=binding,
        centers=centers,
        source_signature=source_signature,
    )


def validate_canonical_payload(sdata: SpatialData, *, table_name: str) -> CanonicalMetadata | None:
    """Validate the canonical centers of a table, when present.

    The matrix and its metadata record must be present together, the record
    with an entry for every region the table declares. Validation is
    structural: it compares the record with the labels elements' metadata and
    reads no labels pixels. Returns the record's entries for the declared
    regions, or ``None`` when the table has no canonical centers.

    No check ties a region to its rows, so subsetting, reordering or
    concatenating valid tables keeps them valid; aligning ``obs`` and ``obsm``
    is AnnData's responsibility. The table's annotation is assumed valid:
    :func:`harpy.tb.validate_table` checks it.
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table element {table_name!r} is not present in 'sdata.tables'.")
    return _validate_canonical_payload(sdata, sdata.tables[table_name])


def _validate_canonical_payload(sdata: SpatialData, table: AnnData) -> CanonicalMetadata | None:
    """Validate the canonical components of a table that need not be in ``sdata``, such as a staged one."""
    metadata = _validate_canonical_without_sources(table)
    if metadata is None:
        return None
    for labels_name, stored in metadata.regions.items():
        if stored.source_signature != build_canonical_source_signature(sdata, labels_name):
            raise ValueError(f"Canonical source signature for labels element `{labels_name}` is stale.")
    return metadata


def _validate_canonical_without_sources(table: AnnData) -> CanonicalMetadata | None:
    """Validate the canonical components without ``sdata``: every check but the labels source signatures.

    Return the record's entries of the declared regions.
    """
    matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    metadata_exists = isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry
    if not matrix_exists and not metadata_exists:
        return None
    if matrix_exists != metadata_exists:
        raise ValueError("Canonical centers require both the matrix and its spatial-coordinate metadata.")
    regions, region_key, _ = get_table_keys(table)
    regions = _names(regions)

    matrix = _validate_canonical_matrix(table.obsm[CANONICAL_OBSM_KEY], table.n_obs)
    if isinstance(matrix, str):
        raise ValueError(matrix)
    assert isinstance(registry, Mapping)
    metadata = parse_canonical_metadata(registry[CANONICAL_OBSM_KEY])
    missing = sorted(set(regions) - set(metadata.regions))
    if missing:
        raise ValueError(f"Canonical metadata has no entry for declared regions {missing!r}.")
    metadata = CanonicalMetadata(
        schema_version=metadata.schema_version,
        regions={labels_name: metadata.regions[labels_name] for labels_name in regions},
    )

    for labels_name in regions:
        stored = metadata.regions[labels_name]
        if stored.algorithm_version != CANONICAL_ALGORITHM_VERSION:
            raise ValueError(f"Canonical algorithm version for labels element `{labels_name}` is unsupported.")
        rows = np.asarray(table.obs[region_key] == labels_name, dtype=bool)
        if stored.source_signature.dims == ("y", "x") and np.any(matrix[rows, 0] != 0.0):
            raise ValueError(f"Canonical centers for 2D labels element `{labels_name}` must use z=0.")
    return metadata


def _parse_region_metadata(region: str, value: object) -> CanonicalRegionMetadata:
    entry = _require_mapping(value, f"region `{region}`", _RegionMetadataError)
    _require_exact_keys(entry, _REGION_KEYS, f"region `{region}`", _RegionMetadataError)
    _require_equal(entry["source_element_type"], "labels", "source_element_type", _RegionMetadataError)
    _require_equal(entry["source_scale"], "scale0", "source_scale", _RegionMetadataError)

    coordinate_frame = _require_mapping(entry["coordinate_frame"], "coordinate_frame", _RegionMetadataError)
    _require_exact_keys(
        coordinate_frame, {"type", "pixel_coordinate_convention"}, "coordinate_frame", _RegionMetadataError
    )
    _require_equal(coordinate_frame["type"], "element_intrinsic", "coordinate_frame.type", _RegionMetadataError)
    _require_equal(
        coordinate_frame["pixel_coordinate_convention"],
        "integer_indices_are_pixel_centers",
        "coordinate_frame.pixel_coordinate_convention",
        _RegionMetadataError,
    )
    algorithm_version = _require_integer(
        entry["algorithm_version"], "algorithm_version", _RegionMetadataError, positive=True
    )

    source = _require_mapping(entry["source"], "source", _RegionMetadataError)
    _require_exact_keys(source, {"dims", "shape", "dtype"}, "source", _RegionMetadataError)
    dims_values = _require_sequence(source["dims"], "source.dims", _RegionMetadataError)
    if not all(isinstance(dim, str) for dim in dims_values) or tuple(dims_values) not in _SUPPORTED_SOURCE_DIMS:
        raise _RegionMetadataError("source.dims must equal ['y', 'x'] or ['z', 'y', 'x'].")
    dims = tuple(dims_values)
    shape_values = _require_sequence(source["shape"], "source.shape", _RegionMetadataError)
    if len(shape_values) != len(dims):
        raise _RegionMetadataError("source.shape length must match source.dims.")
    shape = tuple(
        _require_integer(size, f"source.shape[{index}]", _RegionMetadataError, positive=True)
        for index, size in enumerate(shape_values)
    )
    dtype_value = _require_nonempty_string(source["dtype"], "source.dtype", _RegionMetadataError)
    try:
        source_dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise _RegionMetadataError("source.dtype must be a valid NumPy dtype.") from exc
    if source_dtype.kind not in "iu" or dtype_value != source_dtype.name:
        raise _RegionMetadataError("source.dtype must be a normalized integer NumPy dtype name.")

    try:
        source_signature = CanonicalSourceSignature(
            labels_name=region,
            source_scale="scale0",
            dims=dims,
            shape=shape,
            dtype=source_dtype.name,
        )
        _validate_source(source_signature)
        return CanonicalRegionMetadata(source_signature=source_signature, algorithm_version=algorithm_version)
    except (TypeError, ValueError) as exc:
        raise _RegionMetadataError(str(exc)) from exc


def _validate_canonical_matrix(value: object, n_obs: int) -> np.ndarray | str:
    if issparse(value):
        return "Canonical matrix must be dense."
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError):
        return "Canonical matrix must be a NumPy-compatible dense array."
    if matrix.shape != (n_obs, 3):
        return f"Canonical matrix must have shape ({n_obs}, 3) in z, y, x order."
    if matrix.dtype != np.dtype(np.float64):
        return "Canonical matrix dtype must be float64."
    if not np.isfinite(matrix).all():
        return "Canonical matrix must contain only finite values."
    return matrix


def _validate_source(source: CanonicalSourceSignature) -> None:
    if source.dims not in _SUPPORTED_SOURCE_DIMS:
        raise ValueError("Canonical centers require 2D or 3D source dimensions.")
    try:
        dtype = np.dtype(source.dtype)
    except TypeError as exc:
        raise ValueError("Canonical source dtype must be a valid NumPy dtype.") from exc
    if dtype.kind not in "iu" or source.dtype != dtype.name:
        raise ValueError("Canonical source dtype must be a normalized integer NumPy dtype name.")


def _report(
    stored_metadata: CanonicalMetadata | None,
    source_signature: CanonicalSourceSignature,
    binding: CanonicalRegionBinding,
    *mismatches: CanonicalCacheMismatch,
) -> CanonicalCacheReport:
    return CanonicalCacheReport(
        stored_metadata=stored_metadata,
        source_signature=source_signature,
        binding=binding,
        mismatches=tuple(mismatches),
    )


def _all_regions_mismatch(code: CanonicalMismatchCode, detail: str | None = None) -> CanonicalCacheMismatch:
    return CanonicalCacheMismatch(code=code, detail=_bounded_detail(detail))


def _region_mismatch(code: CanonicalMismatchCode, region: str, detail: str | None = None) -> CanonicalCacheMismatch:
    return CanonicalCacheMismatch(code=code, region=region, detail=_bounded_detail(detail))


def _bounded_detail(detail: str | None) -> str | None:
    return None if detail is None else detail[:240]


def _require_mapping(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type(f"{field} must be a mapping.")
    return value


def _require_sequence(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> list[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise error_type(f"{field} must be a one-dimensional sequence.")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise error_type(f"{field} must be a sequence.")


def _require_string_sequence(
    value: object,
    expected: tuple[str, ...],
    field: str,
    error_type: type[ValueError] = _TopLevelContractError,
) -> tuple[str, ...]:
    values = _require_sequence(value, field, error_type)
    if not all(isinstance(item, str) for item in values) or tuple(values) != expected:
        raise error_type(f"{field} must equal {list(expected)!r}.")
    return expected


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    field: str,
    error_type: type[ValueError],
) -> None:
    if set(mapping) != expected:
        _raise_key_error(mapping, expected, expected, field, error_type)


def _raise_key_error(
    mapping: Mapping[str, object],
    required: set[str],
    allowed: set[str],
    field: str,
    error_type: type[ValueError],
) -> NoReturn:
    missing = sorted(required - set(mapping))
    extra = sorted(set(mapping) - allowed)
    parts: list[str] = []
    if missing:
        parts.append(f"missing {missing!r}")
    if extra:
        parts.append(f"unsupported {extra!r}")
    raise error_type(f"{field} fields are invalid ({'; '.join(parts)}).")


def _require_equal(
    value: object,
    expected: object,
    field: str,
    error_type: type[ValueError] = _TopLevelContractError,
) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if value != expected:
        raise error_type(f"{field} must equal {expected!r}.")


def _require_integer(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
    *,
    positive: bool = False,
) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise error_type(f"{field} must be an integer.")
    integer = int(value)
    if positive and integer <= 0:
        raise error_type(f"{field} must be a positive integer.")
    return integer


def _require_nonempty_string(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str) or not value:
        raise error_type(f"{field} must be a non-empty string.")
    return value
