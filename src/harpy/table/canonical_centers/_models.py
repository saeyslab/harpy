from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type SpatialDimension = Literal["z", "y", "x"]

# A tuple, not a set: membership compares with ``==``, so unhashable dims from a malformed record raise ValueError.
_SUPPORTED_SOURCE_DIMS = (("y", "x"), ("z", "y", "x"))


@dataclass(frozen=True)
class _CanonicalRegionBinding:
    """Current table rows and normalized instance IDs for one labels region."""

    labels_name: str
    row_positions: NDArray[np.intp] = field(repr=False, compare=False)
    instance_ids: NDArray[np.integer] = field(repr=False, compare=False)

    @property
    def n_obs(self) -> int:
        """Return the number of table rows bound to this region."""
        return len(self.instance_ids)

    def __post_init__(self) -> None:
        if not self.labels_name:
            raise ValueError("Canonical binding labels name must not be empty.")
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
        if len(instance_ids) == 0:
            raise ValueError("Instance IDs must not be empty.")
        if len(np.unique(instance_ids)) != len(instance_ids):
            raise ValueError("Instance IDs must be unique within a labels region.")
        object.__setattr__(self, "row_positions", row_positions)
        object.__setattr__(self, "instance_ids", instance_ids)


@dataclass(frozen=True)
class CanonicalRegionMetadata:
    """Persisted calculation identity for one labels region: its ``scale0`` source and the algorithm version."""

    dims: tuple[SpatialDimension, ...]
    shape: tuple[int, ...]
    dtype: str
    algorithm_version: int

    def __post_init__(self) -> None:
        if self.dims not in _SUPPORTED_SOURCE_DIMS:
            raise ValueError("Canonical centers require source dims (`y`, `x`) or (`z`, `y`, `x`).")
        if len(self.shape) != len(self.dims):
            raise ValueError("Source dims and shape must have equal lengths.")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("Source shape must contain positive integers.")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError("Source dtype must be a valid NumPy dtype.") from exc
        if dtype.kind not in "iu" or self.dtype != dtype.name:
            raise ValueError("Source dtype must be a normalized integer NumPy dtype name.")
        if (
            isinstance(self.algorithm_version, bool)
            or not isinstance(self.algorithm_version, int)
            or self.algorithm_version <= 0
        ):
            raise ValueError("Canonical algorithm version must be a positive integer.")


@dataclass(frozen=True)
class CanonicalMetadata:
    """Typed registry for canonical coordinates."""

    schema_version: int
    regions: Mapping[str, CanonicalRegionMetadata]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("Canonical schema version must be an integer.")
        copied: dict[str, CanonicalRegionMetadata] = {}
        for region, metadata in self.regions.items():
            if not isinstance(region, str) or not region:
                raise ValueError("Canonical region names must be non-empty strings.")
            if not isinstance(metadata, CanonicalRegionMetadata):
                raise TypeError("Canonical regions must contain CanonicalRegionMetadata values.")
            copied[region] = metadata
        object.__setattr__(self, "regions", MappingProxyType(copied))


def _readonly_array(value: object, *, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


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
