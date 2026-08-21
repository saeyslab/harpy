from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PRODUCTS = ("morphology", "cell_labels", "compartment_labels", "transcripts")


@dataclass(frozen=True)
class _CosmxChannel:
    channel_id: str
    name: str


@dataclass(frozen=True)
class _CosmxRunMetadata:
    declared_fov_count: int | None
    channels: tuple[_CosmxChannel, ...]
    pixel_size_um: float
    tile_shape: tuple[int, int]
    morphology_dtype: str
    cell_labels_dtype: str | None
    compartment_labels_dtype: str | None


@dataclass(frozen=True)
class _CosmxFovFiles:
    fov: int
    morphology: Path | None = None
    cell_labels: Path | None = None
    compartment_labels: Path | None = None
    transcripts: Path | None = None


@dataclass(frozen=True)
class _CosmxFovPosition:
    fov: int
    x_px: int
    y_px: int
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class _CosmxManifest:
    root: Path
    fovs: tuple[_CosmxFovFiles, ...]
    positions: tuple[_CosmxFovPosition, ...]
    run: _CosmxRunMetadata
    diagnostics: tuple[str, ...]

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
class _CosmxComponentPreview:
    component: int
    fovs: tuple[int, ...]
    origin_x_px: int
    origin_y_px: int
    shape: tuple[int, int]
    image_nbytes: int
    cell_labels_nbytes: int
    compartment_labels_nbytes: int


@dataclass(frozen=True)
class _CosmxPreview:
    manifest: _CosmxManifest
    included_fovs: tuple[int, ...]
    excluded_fovs: tuple[int, ...]
    unpositioned_fovs: tuple[int, ...]
    components: tuple[_CosmxComponentPreview, ...]
    diagnostics: tuple[str, ...]

    @property
    def estimated_image_nbytes(self) -> int:
        return sum(component.image_nbytes for component in self.components)

    @property
    def estimated_cell_labels_nbytes(self) -> int:
        return sum(component.cell_labels_nbytes for component in self.components)

    @property
    def estimated_compartment_labels_nbytes(self) -> int:
        return sum(component.compartment_labels_nbytes for component in self.components)


@dataclass(frozen=True)
class _MorphologyPosition:
    fov: int
    x_mm: float
    y_mm: float
