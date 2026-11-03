"""Canonical label-instance centers and their persisted table contract."""

from harpy.table.canonical_centers._models import (
    CanonicalMetadata,
    CanonicalRegionMetadata,
)
from harpy.table.canonical_centers._schema import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_AXES,
    CANONICAL_OBSM_KEY,
    CANONICAL_SCHEMA_VERSION,
    SPATIAL_COORDINATES_KEY,
    validate_canonical_payload,
)

__all__ = [
    "CANONICAL_ALGORITHM_VERSION",
    "CANONICAL_AXES",
    "CANONICAL_OBSM_KEY",
    "CANONICAL_SCHEMA_VERSION",
    "SPATIAL_COORDINATES_KEY",
    "CanonicalMetadata",
    "CanonicalRegionMetadata",
    "validate_canonical_payload",
]
