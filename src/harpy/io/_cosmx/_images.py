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
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Identity, Scale

from harpy._metadata import _IMAGES_METADATA_KEY, _metadata_registry, _validate_metadata_destination
from harpy.image._image import add_image
from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview
from harpy.io._cosmx._raster import (
    _assemble_raster,
    _mosaic_block_grid,
    _mosaic_placements,
    _pixel_coordinate_system,
)

_DEFAULT_CHUNKS = (1, 1024, 1024)


@dataclass(frozen=True)
class _CosmxSelectedChannel:
    plane: int
    channel_id: str
    name: str
    output_coordinate: str


def _add_morphology_images(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    channels: Sequence[str] | None = None,
    output_image_name: str = "morphology_image",
    coordinate_system: str = "global",
    flip_x: bool = True,
    flip_y: bool = False,
    chunks: tuple[int, int, int] = _DEFAULT_CHUNKS,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Add one lazy, backed morphology image per CosMx mosaic group.

    The function consumes the FOV selection and geometries from ``preview``;
    it does not repeat discovery or spatial grouping. Each selected TIFF page
    becomes one delayed source task and receives the requested dataset-wide
    axis flips before placement. Disjoint tile slices and lazy zero-filled gap
    blocks are concatenated into a virtual mosaic and materialized only when
    Harpy writes the image element to the already-backed ``SpatialData`` store.

    Each image element receives a root metadata record containing:

    - ``fovs``: source FOV numbers contributing to the mosaic;
    - ``source_origin_px``: upper-left mosaic bound in the pre-group/source
      pixel coordinate system. This origin is subtracted from every FOV
      position so that the mosaic starts at ``(0, 0)``. It is source-geometry
      metadata, not an active SpatialData transformation;
    - ``orientation``: dataset-wide local x/y-axis flips applied before
      placement;
    - ``pixel_size_um``: physical size of one source pixel coordinate unit; and
    - ``channels``: retained acquisition channel IDs and names together with
      their source TIFF planes and output channel coordinates.

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
    flip_x
        Whether to reverse the local TIFF x-axis before mosaic placement. The
        default follows the supported decoded-layout convention. The same value
        must be used for labels and transcript-local coordinates.
    flip_y
        Whether to reverse the local TIFF y-axis before mosaic placement. The
        default follows the supported decoded-layout convention. The same value
        must be used for labels and transcript-local coordinates.
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
    _validate_metadata_destination(sdata, _IMAGES_METADATA_KEY)
    placements = {mosaic.mosaic: _mosaic_placements(preview, mosaic) for mosaic in preview.mosaics}
    element_names = tuple(_image_element_name(output_image_name, mosaic.mosaic) for mosaic in preview.mosaics)
    existing = {name: element_type for element_type, name, _ in sdata.gen_elements() if name in element_names}
    wrong_type = sorted(name for name, element_type in existing.items() if element_type != "images")
    if wrong_type:
        raise ValueError(f"CosMx morphology output names already belong to non-image elements: {wrong_type}.")
    collisions = sorted(existing)
    if collisions and not overwrite:
        raise ValueError(f"CosMx morphology image elements already exist: {collisions}.")

    attrs = deepcopy(sdata.attrs)
    images_metadata = _metadata_registry(attrs, _IMAGES_METADATA_KEY)
    channel_records = [
        {
            "channel_id": channel.channel_id,
            "name": channel.name,
            "source_plane": channel.plane,
            "output_coordinate": channel.output_coordinate,
        }
        for channel in selected_channels
    ]

    for mosaic, element_name in zip(preview.mosaics, element_names, strict=True):
        array = _morphology_mosaic(
            preview,
            mosaic,
            placements=placements[mosaic.mosaic],
            channels=selected_channels,
            flip_x=flip_x,
            flip_y=flip_y,
            chunks=chunks,
        )
        log.info(
            f"CosMx morphology mosaic {mosaic.mosaic} pre-write graph contains "
            f"{len(array.__dask_graph__())} tasks across {len(array.dask.layers)} layers."
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
        images_metadata[element_name] = {
            "fovs": list(mosaic.fovs),
            "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
            "orientation": {"flip_x": flip_x, "flip_y": flip_y},
            "pixel_size_um": preview.manifest.run.pixel_size_um,
            "channels": deepcopy(channel_records),
        }

    sdata.attrs = attrs
    sdata.write_attrs()
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
    if (
        not isinstance(chunks, tuple)
        or len(chunks) != 3
        or any(not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1 for chunk in chunks)
    ):
        raise ValueError(f"CosMx morphology chunks must be three positive integers, found {chunks!r}.")
    if chunks[0] != 1:
        raise ValueError(f"CosMx morphology channel chunk must be 1, found {chunks[0]}.")


def _morphology_mosaic(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
    *,
    placements: dict[int, tuple[int, int]],
    channels: tuple[_CosmxSelectedChannel, ...],
    flip_x: bool,
    flip_y: bool,
    chunks: tuple[int, int, int],
) -> da.Array:
    """Construct a lazy ``(c, y, x)`` morphology mosaic without reading pixels."""
    block_grid = _mosaic_block_grid(
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
                flip_x=flip_x,
                flip_y=flip_y,
            )
            for fov in mosaic.fovs
        }
        channel_arrays.append(_assemble_raster(block_grid, planes=planes, dtype=dtype))

    result = da.stack(channel_arrays, axis=0).rechunk(chunks)
    expected_shape = (len(channels), *mosaic.shape)
    if result.shape != expected_shape:
        raise RuntimeError(f"Constructed CosMx morphology mosaic has shape {result.shape}; expected {expected_shape}.")
    return result


def _lazy_morphology_plane(
    path: Path | None,
    *,
    plane: int,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype,
    flip_x: bool,
    flip_y: bool,
) -> da.Array:
    if path is None:
        raise ValueError("Cannot construct a lazy CosMx morphology plane without a TIFF path.")
    value = delayed(_read_morphology_plane, pure=True)(
        path,
        plane,
        expected_shape,
        expected_dtype.name,
        flip_x=flip_x,
        flip_y=flip_y,
    )
    return da.from_delayed(value, shape=expected_shape, dtype=expected_dtype)


def _read_morphology_plane(
    path: Path,
    plane: int,
    expected_shape: tuple[int, int],
    expected_dtype: str,
    *,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Read, revalidate, and orient one morphology TIFF page.

    The flips map local TIFF axes into the stage-derived mosaic axes without
    changing the preview geometry. They must be resolved once per dataset and
    shared by morphology, labels, and transcript-local coordinates.
    """
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
    if flip_y:
        result = result[::-1, :]
    if flip_x:
        result = result[:, ::-1]
    return result


def _image_element_name(base: str, mosaic: int) -> str:
    return f"{base}_mosaic_{mosaic}"
