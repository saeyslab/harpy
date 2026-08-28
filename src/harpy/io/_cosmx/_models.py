from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
_MOSAIC_MODES = ("spatial_groups", "single")
_MosaicMode = Literal["spatial_groups", "single"]
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CosmxSample:
    """Configuration for one independently named decoded CosMx run.

    ``coordinate_system`` is a base name. The reader prefixes it with the
    sample identifier supplied to :func:`harpy.io.cosmx` and appends the mosaic
    number. General FOV and channel sequences are copied to tuples so this
    frozen configuration never retains caller-owned mutable lists.

    Parameters
    ----------
    path
        Decoded CosMx run directory, or a parent containing exactly one
        decoded run.
    fovs
        Optional requested FOV numbers. An ID absent from the run manifest
        raises an error. A known requested FOV that lacks a position or a
        source for any enabled modality is excluded. Ingestion proceeds when
        at least one requested FOV remains usable and raises when none remain.
    channels
        Optional morphology channel IDs or unambiguous biological names.
        Selection preserves acquisition order and has no effect when morphology
        ingestion is disabled.
    mosaic_mode
        ``"spatial_groups"`` derives separate adjacency-based mosaics;
        ``"single"`` places every included FOV in one bounding canvas.
    adjacency_tolerance_px
        Maximum FOV gap bridged in spatial-group mode. ``None`` selects the
        reader default. The value is normalized to ``None`` in single-mosaic
        mode because adjacency grouping is not performed.
    coordinate_system
        Base name for this sample's independent pixel and micron coordinate
        systems. It must match ``^[A-Za-z][A-Za-z0-9_]*$``.
    flip_x, flip_y
        Local-axis orientation applied consistently to the sample's rasters
        and transcript coordinates.
    """

    path: str | Path
    fovs: Sequence[int] | None = None
    channels: Sequence[str] | None = None
    mosaic_mode: _MosaicMode = "spatial_groups"
    adjacency_tolerance_px: int | None = None
    coordinate_system: str = "global"
    flip_x: bool = True
    flip_y: bool = False

    def __post_init__(self) -> None:
        if self.fovs is not None:
            object.__setattr__(self, "fovs", tuple(self.fovs))
        if self.channels is not None:
            channels = (self.channels,) if isinstance(self.channels, str) else tuple(self.channels)
            object.__setattr__(self, "channels", channels)
        if self.mosaic_mode not in _MOSAIC_MODES:
            raise ValueError(f"Unknown CosMx mosaic mode {self.mosaic_mode!r}; expected one of {_MOSAIC_MODES}.")
        if self.mosaic_mode == "single":
            object.__setattr__(self, "adjacency_tolerance_px", None)
        elif (
            self.adjacency_tolerance_px is not None
            and (
                not isinstance(self.adjacency_tolerance_px, int)
                or isinstance(self.adjacency_tolerance_px, bool)
                or self.adjacency_tolerance_px < 0
            )
        ):
            raise ValueError(
                "CosMx adjacency tolerance must be a nonnegative integer or None, "
                f"found {self.adjacency_tolerance_px!r}."
            )
        _validate_identifier(self.coordinate_system, name="coordinate-system base name")


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
    acquisition_timestamp: str | None
    channels: tuple[_CosmxChannel, ...]
    pixel_size_um: float
    tile_shape: tuple[int, int]
    morphology_dtype: str
    instance_labels_dtype: str | None
    compartment_labels_dtype: str | None

    def __post_init__(self) -> None:
        if self.declared_fov_count is not None and self.declared_fov_count < 1:
            raise ValueError(f"Declared CosMx FOV count must be positive, found {self.declared_fov_count}.")
        if self.acquisition_timestamp is not None and (
            not isinstance(self.acquisition_timestamp, str)
            or not self.acquisition_timestamp
            or self.acquisition_timestamp != self.acquisition_timestamp.strip()
        ):
            raise ValueError(
                "CosMx acquisition timestamp must be a non-empty trimmed string or None, "
                f"found {self.acquisition_timestamp!r}."
            )
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
class _CosmxFeatureClass:
    """One authoritative feature class and its sorted panel targets.

    ``targets`` contains the CosMx plex display names assigned to this class.
    These are biological gene targets for an endogenous class, but they are
    negative-target or system-control identifiers for the corresponding control
    classes. Consequently, not every target is a gene.
    """

    name: str
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError(f"CosMx feature-class name must be a non-empty trimmed string, found {self.name!r}.")
        if not self.targets:
            raise ValueError(f"CosMx feature class {self.name!r} must contain at least one target.")
        if tuple(sorted(self.targets)) != self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError(f"CosMx targets for feature class {self.name!r} must be sorted and unique.")
        invalid = [target for target in self.targets if not target or target != target.strip()]
        if invalid:
            raise ValueError(
                f"CosMx targets for feature class {self.name!r} must be non-empty trimmed strings, found {invalid}."
            )


@dataclass(frozen=True)
class _CosmxFeaturePanel:
    """Authoritative relation between transcript features and assay classes.

    A feature is the named target stored in the transcript points element, such
    as a gene or control target. The panel records every target defined by the
    assay, including targets with zero detected transcripts, and assigns each
    one to exactly one feature class. It is target-level metadata rather than a
    description of individual physical probes; fields such as ``ProbeID`` are
    intentionally not represented.

    The relation is parsed once from the run-level plex and shared by all
    transcript mosaic elements from the run.

    For example, a CosMx feature panel is represented conceptually as::

        {
            "feature_column": "gene",
            "class_column": "code_class",
            "categories": ["Endogenous", "Negative", "SystemControl"],
            "targets_by_class": {
                "Endogenous": ["ACTB", "GAPDH", ...],
                "Negative": ["Negative1", "Negative2", ...],
                "SystemControl": ["SystemControl1", "SystemControl2", ...],
            },
        }
    """

    feature_column: str
    class_column: str
    classes: tuple[_CosmxFeatureClass, ...]

    def __post_init__(self) -> None:
        for field_name, value in (("feature column", self.feature_column), ("class column", self.class_column)):
            if not value or value != value.strip():
                raise ValueError(f"CosMx panel {field_name} must be a non-empty trimmed string, found {value!r}.")
        if not self.classes:
            raise ValueError("CosMx feature panel must contain at least one feature class.")
        categories = self.categories
        if tuple(sorted(categories)) != categories or len(set(categories)) != len(categories):
            raise ValueError(f"CosMx feature classes must be sorted and unique, found {categories}.")
        all_targets = tuple(target for feature_class in self.classes for target in feature_class.targets)
        if len(set(all_targets)) != len(all_targets):
            raise ValueError("Each CosMx panel target must belong to exactly one feature class.")

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(feature_class.name for feature_class in self.classes)

    @property
    def target_classes(self) -> dict[str, str]:
        return {target: feature_class.name for feature_class in self.classes for target in feature_class.targets}

    @property
    def targets_by_class(self) -> dict[str, tuple[str, ...]]:
        return {feature_class.name: feature_class.targets for feature_class in self.classes}


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
    feature_panel: _CosmxFeaturePanel | None = None

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
    products: tuple[str, ...] = _PRODUCTS
    mosaic_mode: _MosaicMode = "spatial_groups"
    adjacency_tolerance_px: int | None = 0

    def __post_init__(self) -> None:
        if not self.products or len(set(self.products)) != len(self.products):
            raise ValueError(f"CosMx preview products must be non-empty and unique, found {self.products}.")
        unknown_products = set(self.products) - set(_PRODUCTS)
        if unknown_products:
            raise ValueError(f"Unknown CosMx preview products {sorted(unknown_products)}; expected a subset of {_PRODUCTS}.")
        if self.mosaic_mode not in _MOSAIC_MODES:
            raise ValueError(f"Unknown CosMx mosaic mode {self.mosaic_mode!r}; expected one of {_MOSAIC_MODES}.")
        if self.mosaic_mode == "spatial_groups" and (
            not isinstance(self.adjacency_tolerance_px, int)
            or isinstance(self.adjacency_tolerance_px, bool)
            or self.adjacency_tolerance_px < 0
        ):
            raise ValueError(
                f"CosMx adjacency tolerance must be a nonnegative integer, found {self.adjacency_tolerance_px!r}."
            )
        if self.mosaic_mode == "single" and self.adjacency_tolerance_px is not None:
            raise ValueError("CosMx single-mosaic mode does not use an adjacency tolerance.")
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
        _validate_preview_mosaics(self.manifest, self.mosaics, products=self.products)

    @property
    def estimated_image_nbytes(self) -> int:
        """Estimated bytes for dense mosaics containing every image channel."""
        if _MORPHOLOGY_PRODUCT not in self.products:
            return 0
        return (
            self._mosaic_pixel_count
            * len(self.manifest.run.channels)
            * np.dtype(self.manifest.run.morphology_dtype).itemsize
        )

    @property
    def estimated_instance_labels_nbytes(self) -> int:
        """Estimated bytes for dense ``uint32`` instance-label mosaics."""
        if _INSTANCE_LABELS_PRODUCT not in self.products:
            return 0
        return self._mosaic_pixel_count * _INSTANCE_ID_DTYPE.itemsize

    @property
    def estimated_compartment_labels_nbytes(self) -> int:
        """Estimated bytes for dense compartment-label mosaics."""
        if not self.mosaics or _COMPARTMENT_LABELS_PRODUCT not in self.products:
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
    *,
    products: tuple[str, ...] = _PRODUCTS,
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

    if mosaics and _INSTANCE_LABELS_PRODUCT in products:
        instance_labels_dtype = manifest.run.instance_labels_dtype
        if instance_labels_dtype is None:
            raise ValueError("CosMx preview mosaics require an instance-label dtype when instance labels are enabled.")
        # Reserve one complete source-dtype ID range per FOV:
        # global_id = (fov - 1) * number_of_source_dtype_values + local_id.
        # Validate against the full manifest so IDs remain stable across subsets.
        _validate_instance_id_encoding(instance_labels_dtype, max(manifest.fov_ids))
    if mosaics and _COMPARTMENT_LABELS_PRODUCT in products and manifest.run.compartment_labels_dtype is None:
        raise ValueError(
            "CosMx preview mosaics require a compartment-label dtype when compartment labels are enabled."
        )

    for mosaic in mosaics:
        for product in products:
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


def _validate_identifier(value: object, *, name: str) -> None:
    """Require an exact identifier accepted by Harpy and SpatialData tooling."""
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"CosMx {name} must match {_IDENTIFIER_RE.pattern!r}, found {value!r}.")
