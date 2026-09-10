from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type SpatialDimension = Literal["z", "y", "x"]

_DIGEST_DOMAIN = b"napari-harpy/spatial-canonical/instance-set"
_DIGEST_ENCODING_VERSION = 1


class CanonicalCacheState(StrEnum):
    """State of a canonical-coordinate cache."""

    ABSENT = "absent"
    PARTIAL = "partial"
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"


class CanonicalCacheUpdateAction(StrEnum):
    """Mutation performed while applying a canonical-cache update."""

    CREATE = "create"
    EXTEND = "extend"
    REFRESH = "refresh"
    REBUILD = "rebuild"


class CanonicalMismatchCode(StrEnum):
    """Stable, behaviorally meaningful cache mismatch categories."""

    MATRIX_WITHOUT_METADATA = "matrix_without_metadata"
    METADATA_WITHOUT_MATRIX = "metadata_without_matrix"
    MATRIX_INVALID = "matrix_invalid"
    METADATA_INVALID = "metadata_invalid"
    SCHEMA_VERSION_UNSUPPORTED = "schema_version_unsupported"
    TOP_LEVEL_CONTRACT_MISMATCH = "top_level_contract_mismatch"
    REGION_NOT_REGISTERED = "region_not_registered"
    REGION_METADATA_INVALID = "region_metadata_invalid"
    SOURCE_SIGNATURE_MISMATCH = "source_signature_mismatch"
    TABLE_SIGNATURE_MISMATCH = "table_signature_mismatch"
    ALGORITHM_VERSION_MISMATCH = "algorithm_version_mismatch"
    REGION_COORDINATES_INVALID = "region_coordinates_invalid"


_ALL_REGIONS_MISMATCH_CODES = frozenset(
    {
        CanonicalMismatchCode.MATRIX_WITHOUT_METADATA,
        CanonicalMismatchCode.METADATA_WITHOUT_MATRIX,
        CanonicalMismatchCode.MATRIX_INVALID,
        CanonicalMismatchCode.METADATA_INVALID,
        CanonicalMismatchCode.SCHEMA_VERSION_UNSUPPORTED,
        CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH,
        CanonicalMismatchCode.REGION_METADATA_INVALID,
    }
)


@dataclass(frozen=True)
class CanonicalSourceSignature:
    """Structural identity of the labels scale used for calculation."""

    labels_name: str
    source_scale: Literal["scale0"]
    dims: tuple[SpatialDimension, ...]
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.labels_name:
            raise ValueError("Source labels name must not be empty.")
        if self.source_scale != "scale0":
            raise ValueError("Canonical coordinates must use labels source scale `scale0`.")
        if not self.dims:
            raise ValueError("Source dims must not be empty.")
        if len(self.dims) != len(self.shape):
            raise ValueError("Source dims and shape must have equal lengths.")
        if len(set(self.dims)) != len(self.dims):
            raise ValueError("Source dims must be unique.")
        if any(dim not in ("z", "y", "x") for dim in self.dims):
            raise ValueError("Source dims must contain only `z`, `y`, and `x`.")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("Source shape must contain positive integers.")
        if not self.dtype:
            raise ValueError("Source dtype must not be empty.")

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return len(self.dims)


@dataclass(frozen=True)
class CanonicalRegionBinding:
    """Current table rows and normalized instance IDs for one labels region."""

    table_name: str
    labels_name: str
    region_key: str
    instance_key: str
    row_positions: NDArray[np.intp] = field(repr=False, compare=False)
    instance_ids: NDArray[np.integer] = field(repr=False, compare=False)
    instance_set_digest: str = field(init=False)

    @property
    def n_obs(self) -> int:
        """Return the number of table rows bound to this region."""
        return len(self.instance_ids)

    def __post_init__(self) -> None:
        if not self.table_name or not self.labels_name or not self.region_key or not self.instance_key:
            raise ValueError("Canonical binding names and linkage keys must not be empty.")
        row_positions = _readonly_array(self.row_positions, dtype=np.intp)
        instance_ids = _readonly_integer_ids(self.instance_ids)
        if row_positions.ndim != 1 or instance_ids.ndim != 1:
            raise ValueError("Canonical region binding arrays must be one-dimensional.")
        if len(row_positions) != len(instance_ids):
            raise ValueError("Canonical region binding arrays must have equal lengths.")
        if np.any(row_positions < 0):
            raise ValueError("Canonical row positions must be non-negative.")
        if len(np.unique(row_positions)) != len(row_positions):
            raise ValueError("Canonical row positions must be unique.")
        digest = _build_instance_set_digest(self.labels_name, instance_ids)
        object.__setattr__(self, "row_positions", row_positions)
        object.__setattr__(self, "instance_ids", instance_ids)
        object.__setattr__(self, "instance_set_digest", digest)


@dataclass(frozen=True)
class CanonicalRegionMetadata:
    """Persisted calculation identity for one labels region."""

    source_signature: CanonicalSourceSignature
    n_obs: int
    instance_set_digest: str
    algorithm_version: int
    generated_by_package: str | None = None
    generated_by_version: str | None = None
    generated_at: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_obs, bool) or not isinstance(self.n_obs, int) or self.n_obs <= 0:
            raise ValueError("Canonical region metadata n_obs must be a positive integer.")
        _validate_digest(self.instance_set_digest)
        if (
            isinstance(self.algorithm_version, bool)
            or not isinstance(self.algorithm_version, int)
            or self.algorithm_version <= 0
        ):
            raise ValueError("Canonical algorithm version must be a positive integer.")
        for name, value in (
            ("generated_by_package", self.generated_by_package),
            ("generated_by_version", self.generated_by_version),
            ("generated_at", self.generated_at),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string when provided.")


@dataclass(frozen=True)
class CanonicalMetadata:
    """Typed schema-v1 registry for canonical coordinates."""

    schema_version: int
    region_key: str
    instance_key: str
    regions: Mapping[str, CanonicalRegionMetadata]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("Canonical schema version must be an integer.")
        if not self.region_key or not self.instance_key:
            raise ValueError("Canonical linkage keys must not be empty.")
        copied: dict[str, CanonicalRegionMetadata] = {}
        for region, metadata in self.regions.items():
            if not isinstance(region, str) or not region:
                raise ValueError("Canonical region names must be non-empty strings.")
            if not isinstance(metadata, CanonicalRegionMetadata):
                raise TypeError("Canonical regions must contain CanonicalRegionMetadata values.")
            if metadata.source_signature.labels_name != region:
                raise ValueError("Canonical region keys must match their source labels names.")
            copied[region] = metadata
        object.__setattr__(self, "regions", MappingProxyType(copied))


@dataclass(frozen=True)
class CanonicalCacheMismatch:
    """One deterministic canonical-cache mismatch."""

    code: CanonicalMismatchCode
    region: str | None = None
    detail: str | None = None

    @property
    def scope(self) -> Literal["all_regions", "region"]:
        """Return whether this mismatch invalidates all regions or one region."""
        return "all_regions" if self.code in _ALL_REGIONS_MISMATCH_CODES else "region"

    def __post_init__(self) -> None:
        if self.scope == "all_regions" and self.region is not None:
            raise ValueError("All-regions canonical mismatches must not name a region.")
        if self.scope == "region" and not self.region:
            raise ValueError("Region-local canonical mismatches must name a region.")
        if self.detail is not None:
            if not isinstance(self.detail, str) or not self.detail:
                raise ValueError("Canonical mismatch detail must be a non-empty string when provided.")
            if len(self.detail) > 240:
                raise ValueError("Canonical mismatch detail must not exceed 240 characters.")


@dataclass(frozen=True)
class CanonicalCacheReport:
    """Non-mutating cache inspection result for one selected labels region."""

    stored_metadata: CanonicalMetadata | None
    source_signature: CanonicalSourceSignature
    binding: CanonicalRegionBinding
    mismatches: tuple[CanonicalCacheMismatch, ...] = ()

    @property
    def state(self) -> CanonicalCacheState:
        """Derive the cache state from metadata presence and mismatch reasons."""
        if any(mismatch.scope == "all_regions" for mismatch in self.mismatches):
            return CanonicalCacheState.INVALID
        if any(mismatch.code is CanonicalMismatchCode.REGION_NOT_REGISTERED for mismatch in self.mismatches):
            return CanonicalCacheState.PARTIAL
        if self.mismatches:
            return CanonicalCacheState.STALE
        if self.stored_metadata is None:
            return CanonicalCacheState.ABSENT
        return CanonicalCacheState.VALID

    @property
    def table_name(self) -> str:
        """Return the table name from the selected-region binding."""
        return self.binding.table_name

    @property
    def labels_name(self) -> str:
        """Return the labels name from the selected-region binding."""
        return self.binding.labels_name


@dataclass(frozen=True)
class CanonicalCacheUpdatePayload:
    """Calculated centers and identities prepared for a canonical payload."""

    binding: CanonicalRegionBinding
    centers: NDArray[np.float64] = field(repr=False, compare=False)
    source_signature: CanonicalSourceSignature

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CanonicalRegionBinding):
            raise TypeError("Canonical payload binding must be a CanonicalRegionBinding.")
        object.__setattr__(
            self,
            "centers",
            _normalize_canonical_centers(
                self.centers,
                self.binding,
                self.source_signature,
                owner="Canonical cache-update payload",
            ),
        )

    @property
    def table_name(self) -> str:
        """Return the table name from the calculation-time binding."""
        return self.binding.table_name

    @property
    def labels_name(self) -> str:
        """Return the labels name from the calculation-time binding."""
        return self.binding.labels_name


@dataclass(frozen=True)
class CanonicalCacheUpdateResult:
    """Summary of a successfully applied canonical-cache update."""

    action: CanonicalCacheUpdateAction
    mismatches: tuple[CanonicalCacheMismatch, ...]


@dataclass(frozen=True)
class CanonicalCentersResult:
    """Selected-region canonical centers ready for immediate use."""

    source_signature: CanonicalSourceSignature
    binding: CanonicalRegionBinding
    centers: NDArray[np.float64] = field(repr=False, compare=False)
    cache_update: CanonicalCacheUpdateResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CanonicalRegionBinding):
            raise TypeError("Canonical centers result binding must be a CanonicalRegionBinding.")
        centers = _normalize_canonical_centers(
            self.centers,
            self.binding,
            self.source_signature,
            owner="Canonical centers result",
        )
        if self.cache_update is not None and not isinstance(self.cache_update, CanonicalCacheUpdateResult):
            raise TypeError("Canonical centers cache update must be a CanonicalCacheUpdateResult or None.")
        object.__setattr__(self, "centers", centers)

    @property
    def table_name(self) -> str:
        """Return the table name from the selected-region binding."""
        return self.binding.table_name

    @property
    def labels_name(self) -> str:
        """Return the selected labels region name."""
        return self.binding.labels_name

    @property
    def n_obs(self) -> int:
        """Return the number of selected-region centers."""
        return self.binding.n_obs

    @property
    def reused(self) -> bool:
        """Return whether an existing valid cache was reused."""
        return self.cache_update is None


def build_instance_set_digest(labels_name: str, instance_ids: Sequence[int] | np.ndarray) -> str:
    """Return the versioned, order-independent digest for one instance set."""
    if not isinstance(labels_name, str) or not labels_name:
        raise ValueError("Labels name must be a non-empty string.")
    return _build_instance_set_digest(labels_name, _readonly_integer_ids(instance_ids))


def _readonly_array(value: object, *, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _normalize_canonical_centers(
    value: object,
    binding: CanonicalRegionBinding,
    source_signature: CanonicalSourceSignature,
    *,
    owner: str,
) -> NDArray[np.float64]:
    if not isinstance(source_signature, CanonicalSourceSignature):
        raise TypeError(f"{owner} source signature must be a CanonicalSourceSignature.")
    if source_signature.labels_name != binding.labels_name:
        raise ValueError(f"{owner} source signature labels name does not match.")
    try:
        centers = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} centers must be a dense numeric array.") from exc
    if centers.shape != (binding.n_obs, 3):
        raise ValueError(f"{owner} centers must have shape (n_instances, 3) in z, y, x order.")
    if centers.dtype.kind not in "fiu":
        raise ValueError(f"{owner} centers must have a numeric dtype.")
    normalized = _readonly_array(centers, dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{owner} centers must contain only finite values.")
    if source_signature.dims == ("y", "x") and np.any(normalized[:, 0] != 0.0):
        raise ValueError(f"{owner} centers for a 2D labels source must use z=0.")
    return normalized


def _readonly_integer_ids(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("Instance IDs must be one-dimensional.")
    if array.dtype.kind not in "iu" or array.dtype.itemsize > np.dtype(np.uint64).itemsize:
        raise TypeError("Instance IDs must use an integer NumPy dtype of at most 64 bits.")
    if np.any(array == 0) or (array.dtype.kind == "i" and np.any(array < 0)):
        raise ValueError("Instance IDs must be positive integers.")
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


def _build_instance_set_digest(labels_name: str, instance_ids: np.ndarray) -> str:
    if len(instance_ids) == 0:
        raise ValueError("Instance IDs must not be empty.")
    canonical_ids = np.sort(instance_ids).astype(">u8", copy=False)
    if len(canonical_ids) > 1 and np.any(canonical_ids[1:] == canonical_ids[:-1]):
        raise ValueError("Instance IDs must be unique within a labels region.")
    hasher = hashlib.sha256()
    _update_length_delimited(hasher, _DIGEST_DOMAIN)
    hasher.update(_DIGEST_ENCODING_VERSION.to_bytes(2, byteorder="big", signed=False))
    _update_length_delimited(hasher, labels_name.encode("utf-8"))
    hasher.update(_encode_u64(len(canonical_ids)))
    hasher.update(memoryview(canonical_ids).cast("B"))
    return f"sha256:{hasher.hexdigest()}"


def _encode_u64(value: int) -> bytes:
    return value.to_bytes(8, byteorder="big", signed=False)


def _update_length_delimited(hasher: Any, value: bytes) -> None:
    hasher.update(_encode_u64(len(value)))
    hasher.update(value)


def _validate_digest(value: str) -> None:
    prefix = "sha256:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("Instance-set digest must use the `sha256:` prefix.")
    hexadecimal = value[len(prefix) :]
    if len(hexadecimal) != 64 or any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ValueError("Instance-set digest must contain a lowercase SHA-256 hexadecimal digest.")
