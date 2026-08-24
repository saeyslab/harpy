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
_INSTANCE_ID_DTYPE = np.dtype(np.uint32)


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
                raise ValueError(f"CosMx manifest FOVs must match the declared range {expected_fovs}, found {fov_ids}.")

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
    """Geometry of one derived spatial group of nearby CosMx FOVs.

    The FOVs form one connected mosaic group under the configured adjacency
    rule. The origin and shape define their bounding pixel canvas; the group is
    not an authoritative vendor ROI or biological compartment.
    """

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
class _CosmxPreview:
    manifest: _CosmxManifest
    included_fovs: tuple[int, ...]
    excluded_fovs: tuple[int, ...]
    unpositioned_fovs: tuple[int, ...]
    mosaics: tuple[_CosmxMosaicGeometry, ...]
    diagnostics: tuple[str, ...]
    adjacency_tolerance_px: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.adjacency_tolerance_px, int)
            or isinstance(self.adjacency_tolerance_px, bool)
            or self.adjacency_tolerance_px < 0
        ):
            raise ValueError(
                "CosMx adjacency tolerance must be a nonnegative integer, "
                f"found {self.adjacency_tolerance_px!r}."
            )
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
            raise ValueError(
                f"CosMx included FOVs must have positions, found unpositioned {sorted(included - positioned)}."
            )

        mosaic_ids = tuple(mosaic.mosaic for mosaic in self.mosaics)
        if mosaic_ids != tuple(range(1, len(self.mosaics) + 1)):
            raise ValueError(f"CosMx mosaic numbers must be consecutive from 1, found {mosaic_ids}.")
        mosaic_fovs = tuple(fov for mosaic in self.mosaics for fov in mosaic.fovs)
        if len(set(mosaic_fovs)) != len(mosaic_fovs) or set(mosaic_fovs) != included:
            raise ValueError("Every included CosMx FOV must belong to exactly one mosaic.")
        _validate_preview_mosaics(self.manifest, self.mosaics)

    @property
    def estimated_image_nbytes(self) -> int:
        """Estimated bytes for dense mosaics containing every image channel."""
        return (
            self._mosaic_pixel_count
            * len(self.manifest.run.channels)
            * np.dtype(self.manifest.run.morphology_dtype).itemsize
        )

    @property
    def estimated_instance_labels_nbytes(self) -> int:
        """Estimated bytes for dense ``uint32`` instance-label mosaics."""
        return self._mosaic_pixel_count * _INSTANCE_ID_DTYPE.itemsize

    @property
    def estimated_compartment_labels_nbytes(self) -> int:
        """Estimated bytes for dense compartment-label mosaics."""
        if not self.mosaics:
            return 0
        dtype = self.manifest.run.compartment_labels_dtype
        assert dtype is not None
        return self._mosaic_pixel_count * np.dtype(dtype).itemsize

    @property
    def _mosaic_pixel_count(self) -> int:
        return sum(mosaic.shape[0] * mosaic.shape[1] for mosaic in self.mosaics)


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


def _validate_preview_mosaics(
    manifest: _CosmxManifest,
    mosaics: tuple[_CosmxMosaicGeometry, ...],
) -> None:
    """Validate mosaic sources and geometry against the manifest.

    FOV rectangles may share an edge or meet at a corner: either case has zero
    intersection area and does not create conflicting raster pixels. FOVs may
    also be separated by uncovered gaps within a mosaic. An intersection with
    positive extent along both x and y is rejected because the reader has no
    image-blending or label-conflict policy.

    Edge contact is allowed because its overlap width is zero::

        ┌────────┐┌────────┐
        │ FOV 1  ││ FOV 2  │
        └────────┘└────────┘
                 ↑ overlap width = 0

    Positive-area overlap is rejected::

        ┌────────┐
        │ FOV 1  │
        │    ┌───┼────┐
        └────┼───┘    │
             │ FOV 2  │
             └────────┘
    """
    positions = manifest.positions_by_fov
    fovs_by_id = manifest.fovs_by_id
    tile_height, tile_width = manifest.run.tile_shape

    if mosaics:
        instance_labels_dtype = manifest.run.instance_labels_dtype
        compartment_labels_dtype = manifest.run.compartment_labels_dtype
        if instance_labels_dtype is None or compartment_labels_dtype is None:
            raise ValueError("CosMx preview mosaics require instance- and compartment-label dtypes.")
        # Reserve one complete source-dtype ID range per FOV:
        # global_id = (fov - 1) * number_of_source_dtype_values + local_id.
        # Validate against the full manifest so IDs remain stable across subsets.
        _validate_instance_id_encoding(instance_labels_dtype, max(manifest.fov_ids))

    for mosaic in mosaics:
        for product in _PRODUCTS:
            missing_sources = [fov for fov in mosaic.fovs if getattr(fovs_by_id[fov], product) is None]
            if missing_sources:
                raise ValueError(f"CosMx mosaic {mosaic.mosaic} FOVs have no {product} sources: {missing_sources}.")

        expected_origin_x = min(positions[fov].x_px for fov in mosaic.fovs)
        expected_origin_y = min(positions[fov].y_px for fov in mosaic.fovs)
        expected_max_x = max(positions[fov].x_px + tile_width for fov in mosaic.fovs)
        expected_max_y = max(positions[fov].y_px + tile_height for fov in mosaic.fovs)
        expected_origin = (expected_origin_x, expected_origin_y)
        expected_shape = (expected_max_y - expected_origin_y, expected_max_x - expected_origin_x)
        if (mosaic.origin_x_px, mosaic.origin_y_px) != expected_origin:
            raise ValueError(
                f"CosMx mosaic {mosaic.mosaic} origin must be {expected_origin}, found "
                f"{(mosaic.origin_x_px, mosaic.origin_y_px)}."
            )
        if mosaic.shape != expected_shape:
            raise ValueError(f"CosMx mosaic {mosaic.mosaic} shape must be {expected_shape}, found {mosaic.shape}.")

        for index, left in enumerate(mosaic.fovs):
            left_position = positions[left]
            for right in mosaic.fovs[index + 1 :]:
                right_position = positions[right]
                overlap_x0 = max(left_position.x_px, right_position.x_px)
                overlap_x1 = min(left_position.x_px + tile_width, right_position.x_px + tile_width)
                overlap_y0 = max(left_position.y_px, right_position.y_px)
                overlap_y1 = min(left_position.y_px + tile_height, right_position.y_px + tile_height)
                # A zero extent means allowed edge or corner contact. Only an
                # intersection that is positive along both axes covers pixels.
                if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
                    raise ValueError(
                        f"CosMx mosaic {mosaic.mosaic} has positive-area overlap between FOVs {left} and {right}: "
                        f"x=[{overlap_x0}, {overlap_x1}), y=[{overlap_y0}, {overlap_y1})."
                    )


def _validate_instance_id_encoding(source_dtype: str, max_fov: int) -> None:
    """Validate that FOV-local instance IDs can be encoded safely as ``uint32``.

    A later ingestion step will map each nonzero local instance ID using::

        global_id = (fov - 1) * number_of_source_dtype_values + local_id

    Here, ``number_of_source_dtype_values`` is the size of the complete value
    range representable by the unsigned source dtype (for example, 65,536 for
    ``uint16``). This reserves a non-overlapping ID range for every FOV while
    keeping zero as background.

    This function checks the encoding from dtype metadata and the maximum FOV
    number only. It neither reads label pixels nor performs the remapping.

    Parameters
    ----------
    source_dtype
        Dtype of the FOV-local instance-label rasters. It must be an unsigned
        integer dtype.
    max_fov
        Largest FOV number that the encoding must support. It must be positive
        and should cover the complete manifest rather than a selected subset.

    Raises
    ------
    ValueError
        If the source dtype is not unsigned, ``max_fov`` is not positive, or
        the largest possible encoded ID does not fit in ``uint32``.
    """
    source = np.dtype(source_dtype)
    base = _instance_id_base(source)
    if max_fov < 1:
        raise ValueError(f"Maximum CosMx FOV number must be positive, found {max_fov}.")
    max_global_id = (max_fov - 1) * base + (base - 1)
    if max_global_id > np.iinfo(_INSTANCE_ID_DTYPE).max:
        raise ValueError(
            f"CosMx instance-ID encoding requires maximum ID {max_global_id}, which does not fit in uint32."
        )


def _instance_id_base(source_dtype: str | np.dtype) -> int:
    """Return the complete value-range size reserved for each FOV's IDs."""
    source = np.dtype(source_dtype)
    if source.kind != "u":
        raise ValueError(f"CosMx instance labels must use an unsigned integer dtype, found {source.name}.")
    return 1 << (source.itemsize * 8)


def _validate_stage_position(x_mm: float, y_mm: float, *, fov: int) -> None:
    if not math.isfinite(x_mm) or not math.isfinite(y_mm):
        raise ValueError(f"CosMx stage position for FOV {fov} must be finite, found {(x_mm, y_mm)}.")
