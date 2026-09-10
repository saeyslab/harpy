from __future__ import annotations

from typing import Any

from spatialdata import SpatialData

_HARPY_METADATA_KEY = "harpy"
_METADATA_VERSION_KEY = "metadata_version"
_METADATA_VERSION = 1
_PROVENANCE_METADATA_KEY = "provenance"
_IMAGES_METADATA_KEY = "images"
_LABELS_METADATA_KEY = "labels"
_POINTS_METADATA_KEY = "points"
_FEATURE_PANELS_METADATA_KEY = "feature_panels"
_METADATA_REGISTRIES = (
    _IMAGES_METADATA_KEY,
    _LABELS_METADATA_KEY,
    _POINTS_METADATA_KEY,
    _FEATURE_PANELS_METADATA_KEY,
)


def _validate_metadata_destination(sdata: SpatialData, *registries: str) -> None:
    """Validate Harpy root metadata and the requested element registries."""
    unknown = sorted(set(registries) - set(_METADATA_REGISTRIES))
    if unknown:
        raise ValueError(f"Unknown Harpy metadata registries: {unknown}.")

    harpy_metadata = sdata.attrs.get(_HARPY_METADATA_KEY)
    if harpy_metadata is None:
        return
    if not isinstance(harpy_metadata, dict):
        raise ValueError("SpatialData attribute 'harpy' must be a mapping.")

    version = harpy_metadata.get(_METADATA_VERSION_KEY)
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION
    ):
        raise ValueError(
            f"SpatialData attribute 'harpy.{_METADATA_VERSION_KEY}' must equal {_METADATA_VERSION}, found {version!r}."
        )
    for registry in registries:
        value = harpy_metadata.get(registry)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"SpatialData attribute 'harpy.{registry}' must be a mapping.")


def _metadata_registry(attrs: dict[str, Any], registry: str) -> dict[str, Any]:
    """Return a mutable, versioned Harpy element-metadata registry."""
    if registry not in _METADATA_REGISTRIES:
        raise ValueError(f"Unknown Harpy metadata registry {registry!r}.")
    harpy_metadata = _harpy_metadata(attrs)
    value = harpy_metadata.setdefault(registry, {})
    if not isinstance(value, dict):
        raise ValueError(f"SpatialData attribute 'harpy.{registry}' must be a mapping.")
    return value


def _harpy_metadata(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable versioned Harpy root metadata mapping."""
    harpy_metadata = attrs.setdefault(_HARPY_METADATA_KEY, {})
    if not isinstance(harpy_metadata, dict):
        raise ValueError("SpatialData attribute 'harpy' must be a mapping.")
    version = harpy_metadata.setdefault(_METADATA_VERSION_KEY, _METADATA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION:
        raise ValueError(
            f"SpatialData attribute 'harpy.{_METADATA_VERSION_KEY}' must equal {_METADATA_VERSION}, found {version!r}."
        )
    return harpy_metadata
