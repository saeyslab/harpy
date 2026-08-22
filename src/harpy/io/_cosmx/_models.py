from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MORPHOLOGY_PRODUCT = "morphology"
_INSTANCE_LABELS_PRODUCT = "instance_labels"
_COMPARTMENT_LABELS_PRODUCT = "compartment_labels"
_TRANSCRIPTS_PRODUCT = "transcripts"
_PRODUCTS = (
    _MORPHOLOGY_PRODUCT,
    _INSTANCE_LABELS_PRODUCT,
    _COMPARTMENT_LABELS_PRODUCT,
    _TRANSCRIPTS_PRODUCT,
)


@dataclass(frozen=True)
class _CosmxChannel:
    channel_id: str
    name: str

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("CosMx channel ID must not be empty.")
        if not self.name:
            raise ValueError("CosMx channel name must not be empty.")


@dataclass(frozen=True)
class _CosmxRunMetadata:
    declared_fov_count: int | None
    channels: tuple[_CosmxChannel, ...]
    pixel_size_um: float
    tile_shape: tuple[int, int]
    morphology_dtype: str
    instance_labels_dtype: str | None
    compartment_labels_dtype: str | None

    def __post_init__(self) -> None:
        if self.declared_fov_count is not None and self.declared_fov_count < 1:
            raise ValueError(f"Declared CosMx FOV count must be positive, found {self.declared_fov_count}.")
        if not self.channels:
            raise ValueError("CosMx run metadata must contain at least one channel.")
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError(f"CosMx channel IDs must be unique, found {channel_ids}.")
        if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError(f"CosMx pixel size must be finite and positive, found {self.pixel_size_um}.")
        if len(self.tile_shape) != 2 or any(size <= 0 for size in self.tile_shape):
            raise ValueError(f"CosMx tile shape must contain two positive dimensions, found {self.tile_shape}.")
        _validate_dtype(self.morphology_dtype, name="morphology")
        for name, dtype in (
            ("instance labels", self.instance_labels_dtype),
            ("compartment labels", self.compartment_labels_dtype),
        ):
            if dtype is not None and _validate_dtype(dtype, name=name).kind != "u":
                raise ValueError(f"CosMx {name} dtype must be unsigned integer, found {dtype}.")


@dataclass(frozen=True)
class _CosmxFovFiles:
    fov: int
    morphology: Path | None = None
    instance_labels: Path | None = None
    compartment_labels: Path | None = None
    transcripts: Path | None = None

    def __post_init__(self) -> None:
        _validate_fov(self.fov)


@dataclass(frozen=True)
class _CosmxFovPosition:
    fov: int
    x_px: int
    y_px: int
    x_mm: float
    y_mm: float

    def __post_init__(self) -> None:
        _validate_fov(self.fov)
        _validate_stage_position(self.x_mm, self.y_mm, fov=self.fov)


@dataclass(frozen=True)
class _CosmxManifest:
    root: Path
    fovs: tuple[_CosmxFovFiles, ...]
    positions: tuple[_CosmxFovPosition, ...]
    run: _CosmxRunMetadata
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        fov_ids = self.fov_ids
        if not fov_ids:
            raise ValueError("CosMx manifest must contain at least one FOV.")
        _validate_sorted_unique_fovs(fov_ids, name="manifest FOVs")
        position_fovs = tuple(position.fov for position in self.positions)
        _validate_sorted_unique_fovs(position_fovs, name="position FOVs")
        unknown_position_fovs = set(position_fovs) - set(fov_ids)
        if unknown_position_fovs:
            raise ValueError(f"CosMx positions reference unknown FOVs {sorted(unknown_position_fovs)}.")
        if self.run.declared_fov_count is not None:
            expected_fovs = tuple(range(1, self.run.declared_fov_count + 1))
            if fov_ids != expected_fovs:
                raise ValueError(
                    f"CosMx manifest FOVs must match the declared range {expected_fovs}, found {fov_ids}."
                )

    @property
    def fov_ids(self) -> tuple[int, ...]:
        return tuple(item.fov for item in self.fovs)

    @property
    def fovs_by_id(self) -> dict[int, _CosmxFovFiles]:
        return {item.fov: item for item in self.fovs}

    @property
    def positions_by_fov(self) -> dict[int, _CosmxFovPosition]:
        return {item.fov: item for item in self.positions}

    def available_fovs(self, product: str) -> tuple[int, ...]:
        if product not in _PRODUCTS:
            raise ValueError(f"Unknown CosMx product {product!r}. Expected one of {_PRODUCTS}.")
        return tuple(item.fov for item in self.fovs if getattr(item, product) is not None)


@dataclass(frozen=True)
class _CosmxMosaicGeometry:
    mosaic: int
    fovs: tuple[int, ...]
    origin_x_px: int
    origin_y_px: int
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.mosaic < 1:
            raise ValueError(f"CosMx mosaic number must be positive, found {self.mosaic}.")
        if not self.fovs:
            raise ValueError("CosMx mosaic geometry must contain at least one FOV.")
        _validate_sorted_unique_fovs(self.fovs, name=f"mosaic {self.mosaic} FOVs")
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape):
            raise ValueError(f"CosMx mosaic shape must contain two positive dimensions, found {self.shape}.")


@dataclass(frozen=True)
class _CosmxMosaicSizeEstimate:
    mosaic: int
    image_nbytes: int
    instance_labels_nbytes: int
    compartment_labels_nbytes: int

    def __post_init__(self) -> None:
        if self.mosaic < 1:
            raise ValueError(f"CosMx mosaic number must be positive, found {self.mosaic}.")
        estimates = (self.image_nbytes, self.instance_labels_nbytes, self.compartment_labels_nbytes)
        if any(estimate < 0 for estimate in estimates):
            raise ValueError(f"CosMx mosaic byte estimates must be nonnegative, found {estimates}.")


@dataclass(frozen=True)
class _CosmxPreview:
    manifest: _CosmxManifest
    included_fovs: tuple[int, ...]
    excluded_fovs: tuple[int, ...]
    unpositioned_fovs: tuple[int, ...]
    mosaics: tuple[_CosmxMosaicGeometry, ...]
    estimates: tuple[_CosmxMosaicSizeEstimate, ...]
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_sorted_unique_fovs(self.included_fovs, name="included FOVs")
        _validate_sorted_unique_fovs(self.excluded_fovs, name="excluded FOVs")
        _validate_sorted_unique_fovs(self.unpositioned_fovs, name="unpositioned FOVs")

        manifest_fovs = set(self.manifest.fov_ids)
        included = set(self.included_fovs)
        excluded = set(self.excluded_fovs)
        if included & excluded or included | excluded != manifest_fovs:
            raise ValueError("CosMx included and excluded FOVs must form a disjoint partition of the manifest.")

        positioned = set(self.manifest.positions_by_fov)
        expected_unpositioned = manifest_fovs - positioned
        if set(self.unpositioned_fovs) != expected_unpositioned:
            raise ValueError(
                f"CosMx unpositioned FOVs must be {sorted(expected_unpositioned)}, found {self.unpositioned_fovs}."
            )
        if not included <= positioned:
            raise ValueError(f"CosMx included FOVs must have positions, found unpositioned {sorted(included - positioned)}.")

        mosaic_ids = tuple(mosaic.mosaic for mosaic in self.mosaics)
        if mosaic_ids != tuple(range(1, len(self.mosaics) + 1)):
            raise ValueError(f"CosMx mosaic numbers must be consecutive from 1, found {mosaic_ids}.")
        mosaic_fovs = tuple(fov for mosaic in self.mosaics for fov in mosaic.fovs)
        if len(set(mosaic_fovs)) != len(mosaic_fovs) or set(mosaic_fovs) != included:
            raise ValueError("Every included CosMx FOV must belong to exactly one mosaic.")

        estimate_ids = tuple(estimate.mosaic for estimate in self.estimates)
        if estimate_ids != mosaic_ids:
            raise ValueError(f"CosMx size-estimate mosaic IDs {estimate_ids} do not match geometries {mosaic_ids}.")

    @property
    def estimated_image_nbytes(self) -> int:
        return sum(estimate.image_nbytes for estimate in self.estimates)

    @property
    def estimated_instance_labels_nbytes(self) -> int:
        return sum(estimate.instance_labels_nbytes for estimate in self.estimates)

    @property
    def estimated_compartment_labels_nbytes(self) -> int:
        return sum(estimate.compartment_labels_nbytes for estimate in self.estimates)


@dataclass(frozen=True)
class _MorphologyPosition:
    fov: int
    x_mm: float
    y_mm: float

    def __post_init__(self) -> None:
        _validate_fov(self.fov)
        _validate_stage_position(self.x_mm, self.y_mm, fov=self.fov)


def _validate_dtype(dtype: str, *, name: str) -> np.dtype:
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid CosMx {name} dtype {dtype!r}.") from error


def _validate_fov(fov: int) -> None:
    if fov < 1:
        raise ValueError(f"CosMx FOV number must be positive, found {fov}.")


def _validate_sorted_unique_fovs(fovs: tuple[int, ...], *, name: str) -> None:
    if tuple(sorted(fovs)) != fovs or len(set(fovs)) != len(fovs):
        raise ValueError(f"CosMx {name} must be sorted and unique, found {fovs}.")
    for fov in fovs:
        _validate_fov(fov)


def _validate_stage_position(x_mm: float, y_mm: float, *, fov: int) -> None:
    if not math.isfinite(x_mm) or not math.isfinite(y_mm):
        raise ValueError(f"CosMx stage position for FOV {fov} must be finite, found {(x_mm, y_mm)}.")
