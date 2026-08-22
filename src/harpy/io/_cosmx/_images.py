from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
from dask import delayed
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Identity, Scale

from harpy.image._image import add_image
from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview

_DEFAULT_CHUNKS = (1, 1024, 1024)
_ORIENTATION = "identity"


@dataclass(frozen=True)
class _CosmxSelectedChannel:
    plane: int
    channel_id: str
    name: str
    output_coordinate: str


@dataclass(frozen=True)
class _CosmxMosaicCell:
    shape: tuple[int, int]
    fov: int | None = None
    source_y: slice | None = None
    source_x: slice | None = None


def _add_morphology_images(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    channels: Sequence[str] | None = None,
    output_image_name: str = "morphology_image",
    coordinate_system: str = "global",
    chunks: tuple[int, int, int] = _DEFAULT_CHUNKS,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Add one lazy, backed morphology image per CosMx mosaic group.

    The function consumes the FOV selection and geometries from ``preview``;
    it does not repeat discovery or spatial grouping. Each selected TIFF page
    becomes one delayed source task. Disjoint tile slices and lazy zero-filled
    gap cells are concatenated into a virtual mosaic and materialized only when
    Harpy writes the image element to the already-backed ``SpatialData`` store.

    Parameters
    ----------
    sdata
        Backed SpatialData object receiving the morphology images.
    preview
        Validated CosMx preview defining included FOVs and mosaic geometries.
    channels
        Optional acquisition channel IDs or unambiguous biological names.
        Selection preserves acquisition order. By default all channels are
        retained.
    output_image_name
        Base image-element name. Mosaic number ``n`` is appended as
        ``<base>_mosaic_n``.
    coordinate_system
        Base coordinate-system name. Each mosaic receives independent
        ``<base>_n`` pixel and ``<base>_n_micron`` physical systems.
    chunks
        Final ``(c, y, x)`` Dask/Zarr chunks. The channel chunk must be one.
    scale_factors
        Optional relative scale factors used to construct a lazy multiscale
        DataTree before the image element is written.
    overwrite
        Whether existing planned image elements may be replaced.

    Returns
    -------
    SpatialData
        The input object with one backed morphology image per mosaic.

    Raises
    ------
    ValueError
        If the destination is unbacked, selection or chunking is invalid,
        placements overlap or exceed a canvas, or output elements collide.
    """
    if not sdata.is_backed():
        raise ValueError("CosMx morphology ingestion requires a backed SpatialData object.")
    if not preview.mosaics:
        raise ValueError("CosMx morphology ingestion requires at least one selected mosaic.")
    if not output_image_name:
        raise ValueError("CosMx output image base name must not be empty.")
    if not coordinate_system:
        raise ValueError("CosMx coordinate-system base name must not be empty.")

    _validate_chunks(chunks)
    selected_channels = _select_channels(preview, channels)
    _validate_morphology_provenance_destination(sdata)
    placements = {
        mosaic.mosaic: _validate_mosaic_placements(preview, mosaic) for mosaic in preview.mosaics
    }
    element_names = tuple(_image_element_name(output_image_name, mosaic.mosaic) for mosaic in preview.mosaics)
    existing = sorted(set(element_names) & set(sdata.images))
    if existing and not overwrite:
        raise ValueError(f"CosMx morphology image elements already exist: {existing}.")

    written = []
    for mosaic, element_name in zip(preview.mosaics, element_names, strict=True):
        array = _morphology_mosaic(
            preview,
            mosaic,
            placements=placements[mosaic.mosaic],
            channels=selected_channels,
            chunks=chunks,
        )
        pixel_coordinate_system = _pixel_coordinate_system(coordinate_system, mosaic.mosaic)
        micron_coordinate_system = f"{pixel_coordinate_system}_micron"
        sdata = add_image(
            sdata,
            arr=array,
            output_image_name=element_name,
            dims=("c", "y", "x"),
            chunks=chunks,
            transformations={
                pixel_coordinate_system: Identity(),
                micron_coordinate_system: Scale(
                    [preview.manifest.run.pixel_size_um, preview.manifest.run.pixel_size_um],
                    axes=("x", "y"),
                ),
            },
            scale_factors=scale_factors,
            c_coords=[channel.output_coordinate for channel in selected_channels],
            overwrite=overwrite,
        )
        written.append((mosaic, element_name, pixel_coordinate_system, micron_coordinate_system))

    _record_morphology_provenance(
        sdata,
        preview=preview,
        channels=selected_channels,
        chunks=chunks,
        scale_factors=scale_factors,
        written=written,
    )
    return sdata


def _select_channels(
    preview: _CosmxPreview,
    channels: Sequence[str] | None,
) -> tuple[_CosmxSelectedChannel, ...]:
    run_channels = preview.manifest.run.channels
    channel_ids = {channel.channel_id: index for index, channel in enumerate(run_channels)}
    names_to_indices: dict[str, list[int]] = {}
    for index, channel in enumerate(run_channels):
        names_to_indices.setdefault(channel.name, []).append(index)

    if channels is None:
        selected_indices = set(range(len(run_channels)))
    else:
        selectors = (channels,) if isinstance(channels, str) else tuple(channels)
        if not selectors:
            raise ValueError("CosMx morphology channel selection must not be empty.")
        selected_indices = set()
        for selector in selectors:
            if not isinstance(selector, str) or not selector:
                raise ValueError(f"CosMx morphology channel selectors must be nonempty strings, found {selector!r}.")
            if selector in channel_ids:
                index = channel_ids[selector]
            elif selector in names_to_indices:
                candidates = names_to_indices[selector]
                if len(candidates) != 1:
                    candidate_ids = [run_channels[index].channel_id for index in candidates]
                    raise ValueError(
                        f"CosMx morphology channel name {selector!r} is ambiguous across channel IDs {candidate_ids}."
                    )
                index = candidates[0]
            else:
                raise ValueError(
                    f"Unknown CosMx morphology channel {selector!r}. Expected an acquisition ID from "
                    f"{list(channel_ids)} or an unambiguous name from {list(names_to_indices)}."
                )
            if index in selected_indices:
                raise ValueError(f"CosMx morphology channel {selector!r} selects a channel more than once.")
            selected_indices.add(index)

    name_counts = Counter(channel.name for channel in run_channels)
    return tuple(
        _CosmxSelectedChannel(
            plane=index,
            channel_id=channel.channel_id,
            name=channel.name,
            output_coordinate=(
                channel.name if name_counts[channel.name] == 1 else f"{channel.name} [{channel.channel_id}]"
            ),
        )
        for index, channel in enumerate(run_channels)
        if index in selected_indices
    )


def _validate_chunks(chunks: tuple[int, int, int]) -> None:
    if not isinstance(chunks, tuple) or len(chunks) != 3 or any(
        not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1 for chunk in chunks
    ):
        raise ValueError(f"CosMx morphology chunks must be three positive integers, found {chunks!r}.")
    if chunks[0] != 1:
        raise ValueError(f"CosMx morphology channel chunk must be 1, found {chunks[0]}.")


def _validate_mosaic_placements(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
) -> dict[int, tuple[int, int]]:
    """Return mosaic-local tile offsets after validating bounds and overlaps."""
    positions = preview.manifest.positions_by_fov
    tile_height, tile_width = preview.manifest.run.tile_shape
    canvas_height, canvas_width = mosaic.shape
    placements: dict[int, tuple[int, int]] = {}

    for fov in mosaic.fovs:
        position = positions.get(fov)
        if position is None:
            raise ValueError(f"CosMx mosaic {mosaic.mosaic} FOV {fov} has no morphology position.")
        files = preview.manifest.fovs_by_id[fov]
        if files.morphology is None:
            raise ValueError(f"CosMx mosaic {mosaic.mosaic} FOV {fov} has no morphology TIFF.")
        y = position.y_px - mosaic.origin_y_px
        x = position.x_px - mosaic.origin_x_px
        if y < 0 or x < 0 or y + tile_height > canvas_height or x + tile_width > canvas_width:
            raise ValueError(
                f"CosMx FOV {fov} placement {(y, x)} with tile shape {(tile_height, tile_width)} falls outside "
                f"mosaic {mosaic.mosaic} canvas {mosaic.shape}."
            )
        placements[fov] = (y, x)

    fovs = tuple(placements)
    for index, left in enumerate(fovs):
        left_y, left_x = placements[left]
        for right in fovs[index + 1 :]:
            right_y, right_x = placements[right]
            overlap_x0 = max(left_x, right_x)
            overlap_x1 = min(left_x + tile_width, right_x + tile_width)
            overlap_y0 = max(left_y, right_y)
            overlap_y1 = min(left_y + tile_height, right_y + tile_height)
            if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
                raise ValueError(
                    f"CosMx mosaic {mosaic.mosaic} has positive-area overlap between FOVs {left} and {right}: "
                    f"x=[{overlap_x0}, {overlap_x1}), y=[{overlap_y0}, {overlap_y1})."
                )
    return placements


def _morphology_mosaic(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
    *,
    placements: dict[int, tuple[int, int]],
    channels: tuple[_CosmxSelectedChannel, ...],
    chunks: tuple[int, int, int],
) -> da.Array:
    """Construct a lazy ``(c, y, x)`` morphology mosaic without reading pixels."""
    cell_grid = _mosaic_cell_grid(
        mosaic,
        placements=placements,
        tile_shape=preview.manifest.run.tile_shape,
    )
    dtype = np.dtype(preview.manifest.run.morphology_dtype)
    fovs_by_id = preview.manifest.fovs_by_id
    channel_arrays = []
    for channel in channels:
        planes = {
            fov: _lazy_morphology_plane(
                fovs_by_id[fov].morphology,
                plane=channel.plane,
                expected_shape=preview.manifest.run.tile_shape,
                expected_dtype=dtype,
            )
            for fov in mosaic.fovs
        }
        channel_arrays.append(_assemble_channel(cell_grid, planes=planes, dtype=dtype))

    result = da.stack(channel_arrays, axis=0).rechunk(chunks)
    expected_shape = (len(channels), *mosaic.shape)
    if result.shape != expected_shape:
        raise RuntimeError(f"Constructed CosMx morphology mosaic has shape {result.shape}; expected {expected_shape}.")
    return result


def _mosaic_cell_grid(
    mosaic: _CosmxMosaicGeometry,
    *,
    placements: dict[int, tuple[int, int]],
    tile_shape: tuple[int, int],
) -> tuple[tuple[_CosmxMosaicCell, ...], ...]:
    """Partition a mosaic into disjoint covered tile slices and uncovered gaps."""
    tile_height, tile_width = tile_shape
    y_boundaries = sorted(
        {0, mosaic.shape[0]}
        | {y for y, _ in placements.values()}
        | {y + tile_height for y, _ in placements.values()}
    )
    x_boundaries = sorted(
        {0, mosaic.shape[1]}
        | {x for _, x in placements.values()}
        | {x + tile_width for _, x in placements.values()}
    )

    rows = []
    for y0, y1 in zip(y_boundaries[:-1], y_boundaries[1:], strict=True):
        row = []
        for x0, x1 in zip(x_boundaries[:-1], x_boundaries[1:], strict=True):
            owners = [
                fov
                for fov, (tile_y, tile_x) in placements.items()
                if tile_y <= y0 and y1 <= tile_y + tile_height and tile_x <= x0 and x1 <= tile_x + tile_width
            ]
            if len(owners) > 1:
                raise RuntimeError(f"CosMx mosaic cell {(y0, y1, x0, x1)} has multiple FOV owners {owners}.")
            if not owners:
                row.append(_CosmxMosaicCell(shape=(y1 - y0, x1 - x0)))
                continue
            fov = owners[0]
            tile_y, tile_x = placements[fov]
            row.append(
                _CosmxMosaicCell(
                    shape=(y1 - y0, x1 - x0),
                    fov=fov,
                    source_y=slice(y0 - tile_y, y1 - tile_y),
                    source_x=slice(x0 - tile_x, x1 - tile_x),
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def _assemble_channel(
    cell_grid: tuple[tuple[_CosmxMosaicCell, ...], ...],
    *,
    planes: dict[int, da.Array],
    dtype: np.dtype,
) -> da.Array:
    rows = []
    for row in cell_grid:
        cells = []
        for cell in row:
            if cell.fov is None:
                cells.append(da.zeros(cell.shape, chunks=cell.shape, dtype=dtype))
            else:
                assert cell.source_y is not None
                assert cell.source_x is not None
                cells.append(planes[cell.fov][cell.source_y, cell.source_x])
        rows.append(cells[0] if len(cells) == 1 else da.concatenate(cells, axis=1))
    return rows[0] if len(rows) == 1 else da.concatenate(rows, axis=0)


def _lazy_morphology_plane(
    path: Path | None,
    *,
    plane: int,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype,
) -> da.Array:
    if path is None:
        raise ValueError("Cannot construct a lazy CosMx morphology plane without a TIFF path.")
    value = delayed(_read_morphology_plane, pure=True)(path, plane, expected_shape, expected_dtype.name)
    return da.from_delayed(value, shape=expected_shape, dtype=expected_dtype)


def _read_morphology_plane(
    path: Path,
    plane: int,
    expected_shape: tuple[int, int],
    expected_dtype: str,
) -> np.ndarray:
    """Read and revalidate one morphology TIFF page in its stored row order."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        if len(shape) != 3 or shape[1:] != expected_shape:
            raise ValueError(
                f"CosMx morphology TIFF {path} changed shape after discovery: found {shape}, expected "
                f"(channels, {expected_shape[0]}, {expected_shape[1]})."
            )
        if dtype != np.dtype(expected_dtype):
            raise ValueError(
                f"CosMx morphology TIFF {path} changed dtype after discovery: found {dtype.name}, "
                f"expected {np.dtype(expected_dtype).name}."
            )
        if plane < 0 or plane >= shape[0]:
            raise ValueError(f"CosMx morphology TIFF {path} has no plane {plane}; found {shape[0]} planes.")
        result = np.asarray(series.pages[plane].asarray())

    if result.shape != expected_shape or result.dtype != dtype:
        raise ValueError(
            f"CosMx morphology plane {plane} in {path} has {(result.shape, result.dtype.name)}; expected "
            f"{(expected_shape, dtype.name)}."
        )
    return result


def _record_morphology_provenance(
    sdata: SpatialData,
    *,
    preview: _CosmxPreview,
    channels: tuple[_CosmxSelectedChannel, ...],
    chunks: tuple[int, int, int],
    scale_factors: ScaleFactors_t | None,
    written: list[tuple[_CosmxMosaicGeometry, str, str, str]],
) -> None:
    attrs = deepcopy(sdata.attrs)
    cosmx = attrs.setdefault("cosmx", {})
    if not isinstance(cosmx, dict):
        raise ValueError("SpatialData attribute 'cosmx' must be a mapping.")
    morphology_images = cosmx.setdefault("morphology_images", {})
    if not isinstance(morphology_images, dict):
        raise ValueError("SpatialData attribute 'cosmx.morphology_images' must be a mapping.")

    channel_records = [
        {
            "channel_id": channel.channel_id,
            "name": channel.name,
            "source_plane": channel.plane,
            "output_coordinate": channel.output_coordinate,
        }
        for channel in channels
    ]
    serialized_scale_factors = (
        None
        if scale_factors is None
        else [dict(factor) if isinstance(factor, dict) else int(factor) for factor in scale_factors]
    )
    for mosaic, element_name, pixel_coordinate_system, micron_coordinate_system in written:
        morphology_images[element_name] = {
            "mosaic": mosaic.mosaic,
            "fovs": list(mosaic.fovs),
            "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
            "shape_yx": list(mosaic.shape),
            "orientation": _ORIENTATION,
            "source_dtype": preview.manifest.run.morphology_dtype,
            "pixel_size_um": preview.manifest.run.pixel_size_um,
            "chunks_cyx": list(chunks),
            "scale_factors": serialized_scale_factors,
            "pixel_coordinate_system": pixel_coordinate_system,
            "micron_coordinate_system": micron_coordinate_system,
            "channels": deepcopy(channel_records),
        }

    sdata.attrs = attrs
    sdata.write_attrs()


def _validate_morphology_provenance_destination(sdata: SpatialData) -> None:
    """Validate existing CosMx provenance mappings before any image write."""
    cosmx = sdata.attrs.get("cosmx")
    if cosmx is not None and not isinstance(cosmx, dict):
        raise ValueError("SpatialData attribute 'cosmx' must be a mapping.")
    morphology_images = None if cosmx is None else cosmx.get("morphology_images")
    if morphology_images is not None and not isinstance(morphology_images, dict):
        raise ValueError("SpatialData attribute 'cosmx.morphology_images' must be a mapping.")


def _image_element_name(base: str, mosaic: int) -> str:
    return f"{base}_mosaic_{mosaic}"


def _pixel_coordinate_system(base: str, mosaic: int) -> str:
    return f"{base}_{mosaic}"
