from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from scipy.sparse import issparse
from spatialdata.models import TableModel

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

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

CANONICAL_OBSM_KEY = "spatial_canonical"
SPATIAL_COORDINATES_KEY = "spatial_coordinates"
CANONICAL_SCHEMA_VERSION = 1
CANONICAL_ALGORITHM_VERSION = 1
CANONICAL_AXES = ("z", "y", "x")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "obsm_key",
    "axes",
    "dtype",
    "region_key",
    "instance_key",
    "regions",
}
_REGION_KEYS = {
    "source_element",
    "source_element_type",
    "source_scale",
    "coordinate_frame",
    "calculation",
    "coverage",
    "source",
    "generated_by",
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
        raise ValueError("Canonical metadata schema version 1 requires source dims (`y`, `x`) or (`z`, `y`, `x`).")

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
    region_key: str,
    instance_key: str,
    regions: Mapping[str, CanonicalRegionMetadata],
    schema_version: int = CANONICAL_SCHEMA_VERSION,
) -> CanonicalMetadata:
    """Build typed canonical metadata for storage."""
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Only canonical schema version {CANONICAL_SCHEMA_VERSION} can be built.")
    for region_metadata in regions.values():
        _validate_schema_v1_source(region_metadata.source_signature)
    return CanonicalMetadata(
        schema_version=schema_version,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )


def canonical_metadata_to_storage(metadata: CanonicalMetadata) -> dict[str, object]:
    """Serialize typed metadata using AnnData-Zarr-compatible values."""
    if metadata.schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical schema version {metadata.schema_version}.")
    regions: dict[str, object] = {}
    for region, region_metadata in metadata.regions.items():
        source = region_metadata.source_signature
        entry: dict[str, object] = {
            "source_element": region,
            "source_element_type": "labels",
            "source_scale": "scale0",
            "coordinate_frame": {
                "type": "element_intrinsic",
                "element": region,
                "axes": list(CANONICAL_AXES),
            },
            "calculation": {
                "method": "center_of_mass",
                "weighting": "uniform_label_pixels",
                "background_value": 0,
                "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
                "implementation": "harpy.utils.RasterAggregator.center_of_mass",
                "algorithm_version": region_metadata.algorithm_version,
            },
            "coverage": {
                "scope": "all_rows_for_region",
                "n_obs": region_metadata.n_obs,
                "instance_set_digest": region_metadata.instance_set_digest,
            },
            "source": {
                "element_path": f"labels/{region}",
                "dims": list(source.dims),
                "shape": list(source.shape),
                "dtype": source.dtype,
            },
        }
        generated_by = _serialize_generated_by(region_metadata)
        if generated_by is not None:
            entry["generated_by"] = generated_by
        regions[region] = entry

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "obsm_key": CANONICAL_OBSM_KEY,
        "axes": list(CANONICAL_AXES),
        "dtype": "float64",
        "region_key": metadata.region_key,
        "instance_key": metadata.instance_key,
        "regions": regions,
    }


def parse_canonical_metadata(value: object) -> CanonicalMetadata:
    """Strictly parse the canonical schema-v1 storage representation."""
    mapping = _require_mapping(value, "canonical metadata")
    if "schema_version" not in mapping:
        raise _TopLevelContractError("canonical metadata is missing `schema_version`.")
    schema_version = _require_integer(mapping["schema_version"], "schema_version")
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise _UnsupportedSchemaError(f"Unsupported canonical schema version {schema_version}.")
    _require_exact_keys(mapping, _TOP_LEVEL_KEYS, "canonical metadata", _TopLevelContractError)

    _require_equal(mapping["obsm_key"], CANONICAL_OBSM_KEY, "obsm_key")
    _require_string_sequence(mapping["axes"], CANONICAL_AXES, "axes", _TopLevelContractError)
    _require_equal(mapping["dtype"], "float64", "dtype")
    region_key = _require_nonempty_string(mapping["region_key"], "region_key")
    instance_key = _require_nonempty_string(mapping["instance_key"], "instance_key")
    raw_regions = _require_mapping(mapping["regions"], "regions", _RegionMetadataError)

    regions: dict[str, CanonicalRegionMetadata] = {}
    for raw_region, raw_entry in raw_regions.items():
        if not isinstance(raw_region, str) or not raw_region:
            raise _RegionMetadataError("Region names must be non-empty strings.")
        regions[raw_region] = _parse_region_metadata(raw_region, raw_entry)
    return CanonicalMetadata(
        schema_version=schema_version,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )


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
    region_key, instance_key, regions = _table_linkage(table, table_name=table_name)
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


def validate_canonical_payload(
    sdata: SpatialData,
    table: AnnData,
    *,
    table_name: str,
    region_key: str,
    instance_key: str,
    regions: Sequence[str],
) -> CanonicalMetadata | None:
    """Validate a complete canonical matrix and metadata registry when present.

    Validation is structural and reads only the small ``(n_obs, 3)`` canonical
    matrix plus labels metadata. It does not recompute centers or inspect labels
    pixels. A table containing neither canonical component is accepted and
    returns ``None``; matrix/metadata asymmetry is rejected.
    """
    matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    metadata_exists = isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry
    if not matrix_exists and not metadata_exists:
        return None
    if matrix_exists != metadata_exists:
        raise ValueError("Canonical centers require both the matrix and its spatial-coordinate metadata.")

    matrix = _validate_canonical_matrix(table.obsm[CANONICAL_OBSM_KEY], table.n_obs)
    if isinstance(matrix, str):
        raise ValueError(matrix)
    assert isinstance(registry, Mapping)
    metadata = parse_canonical_metadata(registry[CANONICAL_OBSM_KEY])
    if metadata.region_key != region_key or metadata.instance_key != instance_key:
        raise ValueError("Canonical linkage keys disagree with the SpatialData table annotation.")

    expected_regions = tuple(regions)
    if not expected_regions or len(set(expected_regions)) != len(expected_regions):
        raise ValueError("Canonical validation requires unique declared table regions.")
    if set(metadata.regions) != set(expected_regions):
        raise ValueError("Canonical metadata regions disagree with the SpatialData table annotation.")

    covered = np.zeros(table.n_obs, dtype=bool)
    for labels_name in expected_regions:
        source_signature = build_canonical_source_signature(sdata, labels_name)
        binding = build_canonical_region_binding(
            table,
            table_name=table_name,
            labels_name=labels_name,
            region_key=region_key,
            instance_key=instance_key,
            regions=expected_regions,
        )
        stored = metadata.regions[labels_name]
        if stored.source_signature != source_signature:
            raise ValueError(f"Canonical source signature for labels element `{labels_name}` is stale.")
        if stored.n_obs != binding.n_obs or stored.instance_set_digest != binding.instance_set_digest:
            raise ValueError(f"Canonical table binding for labels element `{labels_name}` is stale.")
        if stored.algorithm_version != CANONICAL_ALGORITHM_VERSION:
            raise ValueError(f"Canonical algorithm version for labels element `{labels_name}` is unsupported.")
        if covered[binding.row_positions].any():
            raise ValueError("Canonical region bindings overlap in table-row space.")
        covered[binding.row_positions] = True
        region_centers = matrix[binding.row_positions]
        if not np.isfinite(region_centers).all():
            raise ValueError(f"Canonical centers for labels element `{labels_name}` must be finite.")
        if source_signature.dims == ("y", "x") and np.any(region_centers[:, 0] != 0.0):
            raise ValueError(f"Canonical centers for 2D labels element `{labels_name}` must use z=0.")
    if not covered.all():
        raise ValueError("Canonical metadata does not cover every table row.")
    return metadata


def _table_linkage(table: AnnData, *, table_name: str) -> tuple[str, str, tuple[str, ...]]:
    attrs = _require_mapping(table.uns.get(TableModel.ATTRS_KEY), "SpatialData table annotation")
    region_key = _require_nonempty_string(attrs.get(TableModel.REGION_KEY_KEY), "region_key")
    instance_key = _require_nonempty_string(attrs.get(TableModel.INSTANCE_KEY), "instance_key")
    raw_regions = attrs.get(TableModel.REGION_KEY)
    values = [raw_regions] if isinstance(raw_regions, str) else _require_sequence(raw_regions, "regions")
    regions = tuple(_require_nonempty_string(region, "region") for region in values)
    if not regions or len(set(regions)) != len(regions):
        raise ValueError(f"Table `{table_name}` must declare unique labels regions.")
    return region_key, instance_key, regions


def _parse_region_metadata(region: str, value: object) -> CanonicalRegionMetadata:
    entry = _require_mapping(value, f"region `{region}`", _RegionMetadataError)
    required_keys = _REGION_KEYS - {"generated_by"}
    actual_keys = set(entry)
    if not required_keys.issubset(actual_keys) or not actual_keys.issubset(_REGION_KEYS):
        _raise_key_error(entry, required_keys, _REGION_KEYS, f"region `{region}`", _RegionMetadataError)
    _require_equal(entry["source_element"], region, "source_element", _RegionMetadataError)
    _require_equal(entry["source_element_type"], "labels", "source_element_type", _RegionMetadataError)
    _require_equal(entry["source_scale"], "scale0", "source_scale", _RegionMetadataError)

    coordinate_frame = _require_mapping(entry["coordinate_frame"], "coordinate_frame", _RegionMetadataError)
    _require_exact_keys(coordinate_frame, {"type", "element", "axes"}, "coordinate_frame", _RegionMetadataError)
    _require_equal(coordinate_frame["type"], "element_intrinsic", "coordinate_frame.type", _RegionMetadataError)
    _require_equal(coordinate_frame["element"], region, "coordinate_frame.element", _RegionMetadataError)
    _require_string_sequence(coordinate_frame["axes"], CANONICAL_AXES, "coordinate_frame.axes", _RegionMetadataError)

    calculation = _require_mapping(entry["calculation"], "calculation", _RegionMetadataError)
    _require_exact_keys(
        calculation,
        {
            "method",
            "weighting",
            "background_value",
            "pixel_coordinate_convention",
            "implementation",
            "algorithm_version",
        },
        "calculation",
        _RegionMetadataError,
    )
    for key, expected in (
        ("method", "center_of_mass"),
        ("weighting", "uniform_label_pixels"),
        ("background_value", 0),
        ("pixel_coordinate_convention", "integer_indices_are_pixel_centers"),
        ("implementation", "harpy.utils.RasterAggregator.center_of_mass"),
    ):
        _require_equal(calculation[key], expected, f"calculation.{key}", _RegionMetadataError)
    algorithm_version = _require_integer(
        calculation["algorithm_version"], "calculation.algorithm_version", _RegionMetadataError, positive=True
    )

    coverage = _require_mapping(entry["coverage"], "coverage", _RegionMetadataError)
    _require_exact_keys(coverage, {"scope", "n_obs", "instance_set_digest"}, "coverage", _RegionMetadataError)
    _require_equal(coverage["scope"], "all_rows_for_region", "coverage.scope", _RegionMetadataError)
    n_obs = _require_integer(coverage["n_obs"], "coverage.n_obs", _RegionMetadataError, positive=True)
    digest = _require_nonempty_string(
        coverage["instance_set_digest"], "coverage.instance_set_digest", _RegionMetadataError
    )

    source = _require_mapping(entry["source"], "source", _RegionMetadataError)
    _require_exact_keys(source, {"element_path", "dims", "shape", "dtype"}, "source", _RegionMetadataError)
    _require_equal(source["element_path"], f"labels/{region}", "source.element_path", _RegionMetadataError)
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

    generated_by_package: str | None = None
    generated_by_version: str | None = None
    generated_at: str | None = None
    if "generated_by" in entry:
        generated_by = _require_mapping(entry["generated_by"], "generated_by", _RegionMetadataError)
        if not set(generated_by).issubset({"package", "version", "generated_at"}):
            raise _RegionMetadataError("generated_by contains unsupported fields.")
        if "package" in generated_by:
            generated_by_package = _require_nonempty_string(
                generated_by["package"], "generated_by.package", _RegionMetadataError
            )
        if "version" in generated_by:
            generated_by_version = _require_nonempty_string(
                generated_by["version"], "generated_by.version", _RegionMetadataError
            )
        if "generated_at" in generated_by:
            generated_at = _require_nonempty_string(
                generated_by["generated_at"], "generated_by.generated_at", _RegionMetadataError
            )

    try:
        source_signature = CanonicalSourceSignature(
            labels_name=region,
            source_scale="scale0",
            dims=dims,
            shape=shape,
            dtype=source_dtype.name,
        )
        _validate_schema_v1_source(source_signature)
        return CanonicalRegionMetadata(
            source_signature=source_signature,
            n_obs=n_obs,
            instance_set_digest=digest,
            algorithm_version=algorithm_version,
            generated_by_package=generated_by_package,
            generated_by_version=generated_by_version,
            generated_at=generated_at,
        )
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
    if np.isinf(matrix).any():
        return "Canonical matrix must not contain infinite values."
    return matrix


def _validate_schema_v1_source(source: CanonicalSourceSignature) -> None:
    if source.dims not in _SUPPORTED_SOURCE_DIMS:
        raise ValueError("Canonical metadata schema version 1 requires 2D or 3D source dimensions.")
    try:
        dtype = np.dtype(source.dtype)
    except TypeError as exc:
        raise ValueError("Canonical source dtype must be a valid NumPy dtype.") from exc
    if dtype.kind not in "iu" or source.dtype != dtype.name:
        raise ValueError("Canonical source dtype must be a normalized integer NumPy dtype name.")


def _serialize_generated_by(metadata: CanonicalRegionMetadata) -> dict[str, str] | None:
    generated_by: dict[str, str] = {}
    if metadata.generated_by_package is not None:
        generated_by["package"] = metadata.generated_by_package
    if metadata.generated_by_version is not None:
        generated_by["version"] = metadata.generated_by_version
    if metadata.generated_at is not None:
        generated_by["generated_at"] = metadata.generated_at
    return generated_by or None


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
