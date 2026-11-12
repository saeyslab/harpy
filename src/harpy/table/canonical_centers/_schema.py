from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from spatialdata.models import get_table_keys

from harpy.image import get_dataarray
from harpy.table.canonical_centers._models import (
    CanonicalMetadata,
    CanonicalRegionMetadata,
    _CanonicalRegionBinding,
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


def _build_region_metadata(sdata: SpatialData, labels_name: str) -> CanonicalRegionMetadata:
    """Build the canonical-center metadata of a labels element from its ``scale0`` array; reads no pixels."""
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

    return CanonicalRegionMetadata(
        dims=dims,
        shape=shape,
        dtype=dtype.name,
        algorithm_version=CANONICAL_ALGORITHM_VERSION,
    )


def _build_canonical_region_binding(table: AnnData, *, labels_name: str) -> _CanonicalRegionBinding:
    """Capture the rows and instance IDs of one labels region the table declares.

    The region and instance keys and the declared regions come from the table's
    SpatialData annotation.
    """
    regions, region_key, instance_key = get_table_keys(table)
    if labels_name not in _names(regions):
        raise ValueError(f"Table does not declare labels region `{labels_name}`.")

    mask = np.asarray(table.obs[region_key] == labels_name, dtype=bool)
    row_positions = np.flatnonzero(mask).astype(np.intp, copy=False)
    if len(row_positions) == 0:
        raise ValueError(f"Table contains no rows for labels region `{labels_name}`.")

    return _CanonicalRegionBinding(
        labels_name=labels_name,
        row_positions=row_positions,
        instance_ids=table.obs.iloc[row_positions][instance_key].to_numpy(),
    )


def _build_canonical_metadata(
    *,
    regions: Mapping[str, CanonicalRegionMetadata],
    schema_version: int = CANONICAL_SCHEMA_VERSION,
) -> CanonicalMetadata:
    """Build typed canonical metadata for storage."""
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Only canonical schema version {CANONICAL_SCHEMA_VERSION} can be built.")
    return CanonicalMetadata(schema_version=schema_version, regions=regions)


def _canonical_metadata_to_storage(metadata: CanonicalMetadata) -> dict[str, object]:
    """Serialize typed metadata using AnnData-Zarr-compatible values."""
    if metadata.schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical schema version {metadata.schema_version}.")
    regions: dict[str, object] = {}
    for region, entry in metadata.regions.items():
        regions[region] = {
            "source_element_type": "labels",
            "source_scale": "scale0",
            "coordinate_frame": {
                "type": "element_intrinsic",
                "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
            },
            "algorithm_version": entry.algorithm_version,
            "source": {
                "dims": list(entry.dims),
                "shape": list(entry.shape),
                "dtype": entry.dtype,
            },
        }

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "axes": list(CANONICAL_AXES),
        "regions": regions,
    }


def _parse_canonical_metadata(value: object) -> CanonicalMetadata:
    """Strictly parse the canonical storage representation."""
    mapping = _require_mapping(value, "canonical metadata")
    if "schema_version" not in mapping:
        raise ValueError("canonical metadata is missing `schema_version`.")
    schema_version = _require_integer(mapping["schema_version"], "schema_version")
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported canonical schema version {schema_version}; Harpy reads version "
            f"{CANONICAL_SCHEMA_VERSION}. Rebuild the canonical centers with "
            "hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)."
        )
    _require_exact_keys(mapping, _TOP_LEVEL_KEYS, "canonical metadata")
    _require_string_sequence(mapping["axes"], CANONICAL_AXES, "axes")
    raw_regions = _require_mapping(mapping["regions"], "regions")
    regions = {region: _parse_region_metadata(region, entry) for region, entry in raw_regions.items()}
    return CanonicalMetadata(schema_version=schema_version, regions=regions)


def validate_canonical_centers(sdata: SpatialData, *, table_name: str) -> CanonicalMetadata | None:
    """Validate the canonical centers of a table, when present.

    The matrix and its metadata record must be present together, the record
    with an entry for every region the table declares. Validation is
    structural: it compares the stored ``dims``, ``shape`` and ``dtype`` of each
    region with its labels element at ``scale0``, and reads no labels pixels.
    A labels element replaced by one of the same shape and dtype is not
    detected. Returns the canonical-center metadata of the
    regions the table declares, or ``None`` when the table has no canonical
    centers.

    No check ties a region to its rows, so subsetting, reordering or
    concatenating valid tables keeps them valid; aligning ``obs`` and ``obsm``
    is AnnData's responsibility. The table's annotation is assumed valid:
    :func:`harpy.tb.validate_table` checks it.
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table element {table_name!r} is not present in 'sdata.tables'.")
    return _validate_canonical_centers(sdata, sdata.tables[table_name])


def _validate_canonical_centers(sdata: SpatialData, table: AnnData) -> CanonicalMetadata | None:
    """Validate the canonical centers of ``table`` against the labels elements in ``sdata``.

    As :func:`validate_canonical_centers`, but takes the table itself, so a table
    can be validated before it is added to ``sdata``.
    """
    metadata = _validate_canonical_centers_without_sources(table)
    if metadata is None:
        return None
    for labels_name, stored in metadata.regions.items():
        # Algorithm versions already match (checked above), so a difference is in the source.
        if stored != _build_region_metadata(sdata, labels_name):
            raise ValueError(
                f"Canonical centers of labels element `{labels_name}` are stale: the labels element no longer "
                "matches the one they were calculated from. Rebuild them with "
                "hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)."
            )
    return metadata


def _validate_canonical_centers_without_sources(table: AnnData) -> CanonicalMetadata | None:
    """Validate the canonical centers of ``table`` without ``sdata``: every check except the comparison with the labels elements.

    Return the canonical-center metadata of the regions the table declares.
    """
    matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    metadata_exists = isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry
    if not matrix_exists and not metadata_exists:
        return None
    if matrix_exists != metadata_exists:
        matrix = f"obsm[{CANONICAL_OBSM_KEY!r}]"
        metadata = f"uns[{SPATIAL_COORDINATES_KEY!r}][{CANONICAL_OBSM_KEY!r}]"
        present, absent = (matrix, metadata) if matrix_exists else (metadata, matrix)
        raise ValueError(
            f"The table has {present} but not {absent}; canonical centers need both. Rebuild them with "
            f"hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True), or delete {present}."
        )
    regions, region_key, _ = get_table_keys(table)
    regions = _names(regions)

    matrix = _validate_canonical_matrix(table.obsm[CANONICAL_OBSM_KEY], table.n_obs)
    if isinstance(matrix, str):
        raise ValueError(matrix)
    assert isinstance(registry, Mapping)
    metadata = _parse_canonical_metadata(registry[CANONICAL_OBSM_KEY])
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
        if stored.dims == ("y", "x") and np.any(matrix[rows, 0] != 0.0):
            raise ValueError(f"Canonical centers for 2D labels element `{labels_name}` must use z=0.")
    return metadata


def _parse_region_metadata(region: str, value: object) -> CanonicalRegionMetadata:
    # Checks the structure and constants and converts stored values; CanonicalRegionMetadata checks the values.
    entry = _require_mapping(value, f"region `{region}`")
    _require_exact_keys(entry, _REGION_KEYS, f"region `{region}`")
    _require_equal(entry["source_element_type"], "labels", "source_element_type")
    _require_equal(entry["source_scale"], "scale0", "source_scale")

    coordinate_frame = _require_mapping(entry["coordinate_frame"], "coordinate_frame")
    _require_exact_keys(coordinate_frame, {"type", "pixel_coordinate_convention"}, "coordinate_frame")
    _require_equal(coordinate_frame["type"], "element_intrinsic", "coordinate_frame.type")
    _require_equal(
        coordinate_frame["pixel_coordinate_convention"],
        "integer_indices_are_pixel_centers",
        "coordinate_frame.pixel_coordinate_convention",
    )

    source = _require_mapping(entry["source"], "source")
    _require_exact_keys(source, {"dims", "shape", "dtype"}, "source")
    return CanonicalRegionMetadata(
        dims=tuple(_require_sequence(source["dims"], "source.dims")),
        shape=tuple(
            _require_integer(size, f"source.shape[{index}]")
            for index, size in enumerate(_require_sequence(source["shape"], "source.shape"))
        ),
        dtype=_require_nonempty_string(source["dtype"], "source.dtype"),
        algorithm_version=_require_integer(entry["algorithm_version"], "algorithm_version"),
    )


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


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _require_sequence(value: object, field: str) -> list[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{field} must be a one-dimensional sequence.")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise ValueError(f"{field} must be a sequence.")


def _require_string_sequence(value: object, expected: tuple[str, ...], field: str) -> tuple[str, ...]:
    values = _require_sequence(value, field)
    if not all(isinstance(item, str) for item in values) or tuple(values) != expected:
        raise ValueError(f"{field} must equal {list(expected)!r}.")
    return expected


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(mapping) == expected:
        return
    parts: list[str] = []
    if missing := sorted(expected - set(mapping)):
        parts.append(f"missing {missing!r}")
    if extra := sorted(set(mapping) - expected):
        parts.append(f"unsupported {extra!r}")
    raise ValueError(f"{field} fields are invalid ({'; '.join(parts)}).")


def _require_equal(value: object, expected: object, field: str) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if value != expected:
        raise ValueError(f"{field} must equal {expected!r}.")


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer.")
    return int(value)


def _require_nonempty_string(value: object, field: str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value
