from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import tifffile
from dask import delayed
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Identity, Scale

from harpy._metadata import _LABELS_METADATA_KEY, _metadata_registry, _validate_metadata_destination
from harpy.image._image import add_labels
from harpy.io._cosmx._models import (
    _COMPARTMENT_LABELS_PRODUCT,
    _INSTANCE_ID_DTYPE,
    _INSTANCE_LABELS_PRODUCT,
    _CosmxMosaicGeometry,
    _CosmxPreview,
    _instance_id_base,
)
from harpy.io._cosmx._raster import (
    _assemble_raster,
    _mosaic_block_grid,
    _mosaic_placements,
    _pixel_coordinate_system,
)

_DEFAULT_CHUNKS = (1024, 1024)
_COMPARTMENT_CATEGORIES = {
    0: "background",
    1: "nuclear",
    2: "membrane",
    3: "cytoplasmic",
}
_LabelFamily = Literal["instance_labels", "compartment_labels"]


def _add_instance_labels(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    sample_id: str,
    output_labels_name: str = "instance_labels",
    coordinate_system: str = "global",
    flip_x: bool = True,
    flip_y: bool = False,
    chunks: tuple[int, int] = _DEFAULT_CHUNKS,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Add one lazy, globally ID-remapped instance-label raster per mosaic."""
    return _add_label_family(
        sdata,
        preview,
        family=_INSTANCE_LABELS_PRODUCT,
        output_labels_name=output_labels_name,
        coordinate_system=coordinate_system,
        flip_x=flip_x,
        flip_y=flip_y,
        chunks=chunks,
        scale_factors=scale_factors,
        sample_id=sample_id,
        overwrite=overwrite,
    )


def _add_compartment_labels(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    sample_id: str,
    output_labels_name: str = "compartment_labels",
    coordinate_system: str = "global",
    flip_x: bool = True,
    flip_y: bool = False,
    chunks: tuple[int, int] = _DEFAULT_CHUNKS,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Add one lazy semantic compartment-label raster per mosaic."""
    return _add_label_family(
        sdata,
        preview,
        family=_COMPARTMENT_LABELS_PRODUCT,
        output_labels_name=output_labels_name,
        coordinate_system=coordinate_system,
        flip_x=flip_x,
        flip_y=flip_y,
        chunks=chunks,
        scale_factors=scale_factors,
        sample_id=sample_id,
        overwrite=overwrite,
    )


def _add_label_family(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    family: _LabelFamily,
    output_labels_name: str,
    coordinate_system: str,
    flip_x: bool,
    flip_y: bool,
    chunks: tuple[int, int],
    scale_factors: ScaleFactors_t | None,
    sample_id: str,
    overwrite: bool,
) -> SpatialData:
    """Add one lazy label raster per mosaic for a single CosMx label family.

    The function applies the mosaic groups and geometry already established by
    ``preview``. Each source TIFF is decoded lazily, oriented, interpreted
    according to ``family``, and assembled through the same block placements as
    morphology. Instance labels are remapped to globally unique ``uint32`` IDs;
    compartment labels retain their semantic values and source dtype.

    Each label element receives a root metadata record containing:

    - ``fovs``: source FOV numbers contributing to the mosaic;
    - ``sample_id`` and ``mosaic``: the sample identity and the grouping
      mode/effective adjacency tolerance;
    - ``source_origin_px``: upper-left mosaic bound in the pre-group/source
      pixel coordinate system. This origin is subtracted from every FOV
      position so that the mosaic starts at ``(0, 0)``. It is source-geometry
      metadata, not an active SpatialData transformation;
    - ``orientation``: dataset-wide local x/y-axis flips applied before
      placement;
    - ``pixel_size_um``: physical size of one source pixel coordinate unit;
    - ``instance_id_encoding`` for instance labels: background value, complete
      source-dtype range reserved per FOV, and the local-to-global ID formula;
      and
    - ``categories`` for compartment labels: biological meanings assigned to
      the supported integer codes zero through three.

    Parameters
    ----------
    sdata
        Backed SpatialData object receiving the label elements.
    preview
        Validated FOV selection and mosaic geometry shared by all modalities.
    sample_id
        Identifier of the sample that owns the generated elements.
    family
        Instance- or compartment-label source family to ingest.
    output_labels_name
        Base element name; ``_mosaic_<n>`` is appended for each mosaic.
    coordinate_system
        Base name for each mosaic's independent pixel and physical systems.
    flip_x, flip_y
        Dataset-wide local TIFF-axis flips shared with images and transcripts.
    chunks
        Final two-dimensional ``(y, x)`` Dask/Zarr chunks.
    scale_factors
        Optional relative factors used to construct a lazy label pyramid.
    overwrite
        Whether existing label elements with the planned names may be replaced.

    Returns
    -------
    SpatialData
        The input object with one backed label element per mosaic.
    """
    if not sdata.is_backed():
        raise ValueError("CosMx label ingestion requires a backed SpatialData object.")
    if not preview.mosaics:
        raise ValueError("CosMx label ingestion requires at least one selected mosaic.")
    if not output_labels_name:
        raise ValueError("CosMx output labels base name must not be empty.")
    if not coordinate_system:
        raise ValueError("CosMx coordinate-system base name must not be empty.")

    _validate_label_chunks(chunks)
    _validate_metadata_destination(sdata, _LABELS_METADATA_KEY)
    element_names = tuple(_labels_element_name(output_labels_name, mosaic.mosaic) for mosaic in preview.mosaics)
    existing = {name: element_type for element_type, name, _ in sdata.gen_elements() if name in element_names}
    wrong_type = sorted(name for name, element_type in existing.items() if element_type != "labels")
    if wrong_type:
        raise ValueError(f"CosMx label output names already belong to non-label elements: {wrong_type}.")
    collisions = sorted(existing)
    if collisions and not overwrite:
        raise ValueError(f"CosMx label elements already exist: {collisions}.")

    source_dtype_name = (
        preview.manifest.run.instance_labels_dtype
        if family == _INSTANCE_LABELS_PRODUCT
        else preview.manifest.run.compartment_labels_dtype
    )
    if source_dtype_name is None:
        raise ValueError(f"CosMx preview has no dtype for {family}.")
    source_dtype = np.dtype(source_dtype_name)
    output_dtype = _INSTANCE_ID_DTYPE if family == _INSTANCE_LABELS_PRODUCT else source_dtype
    placements = {mosaic.mosaic: _mosaic_placements(preview, mosaic) for mosaic in preview.mosaics}

    attrs = deepcopy(sdata.attrs)
    labels_metadata = _metadata_registry(attrs, _LABELS_METADATA_KEY)
    instance_id_base = _instance_id_base(source_dtype) if family == _INSTANCE_LABELS_PRODUCT else None

    for mosaic, element_name in zip(preview.mosaics, element_names, strict=True):
        array = _label_mosaic(
            preview,
            mosaic,
            placements=placements[mosaic.mosaic],
            family=family,
            source_dtype=source_dtype,
            output_dtype=output_dtype,
            flip_x=flip_x,
            flip_y=flip_y,
            chunks=chunks,
        )
        log.info(
            f"CosMx {family} mosaic {mosaic.mosaic} pre-write graph contains "
            f"{len(array.__dask_graph__())} tasks across {len(array.dask.layers)} layers."
        )
        pixel_coordinate_system = _pixel_coordinate_system(coordinate_system, mosaic.mosaic)
        sdata = add_labels(
            sdata,
            arr=array,
            output_labels_name=element_name,
            dims=("y", "x"),
            chunks=chunks,
            transformations={
                pixel_coordinate_system: Identity(),
                f"{pixel_coordinate_system}_micron": Scale(
                    [preview.manifest.run.pixel_size_um, preview.manifest.run.pixel_size_um],
                    axes=("x", "y"),
                ),
            },
            scale_factors=scale_factors,
            overwrite=overwrite,
        )
        metadata = {
            "fovs": list(mosaic.fovs),
            "sample_id": sample_id,
            "mosaic": {
                "mode": preview.mosaic_mode,
                "adjacency_tolerance_px": preview.adjacency_tolerance_px,
            },
            "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
            "orientation": {"flip_x": flip_x, "flip_y": flip_y},
            "pixel_size_um": preview.manifest.run.pixel_size_um,
        }
        if family == _INSTANCE_LABELS_PRODUCT:
            assert instance_id_base is not None
            metadata["instance_id_encoding"] = {
                "background": 0,
                "base": instance_id_base,
                "formula": "global_id = (fov - 1) * base + local_id",
            }
        else:
            metadata["categories"] = deepcopy(_COMPARTMENT_CATEGORIES)
        labels_metadata[element_name] = metadata

    sdata.attrs = attrs
    sdata.write_attrs()
    return sdata


def _label_mosaic(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
    *,
    placements: dict[int, tuple[int, int]],
    family: _LabelFamily,
    source_dtype: np.dtype,
    output_dtype: np.dtype,
    flip_x: bool,
    flip_y: bool,
    chunks: tuple[int, int],
) -> da.Array:
    """Construct one lazy ``(y, x)`` label mosaic without reading pixels."""
    block_grid = _mosaic_block_grid(
        mosaic,
        placements=placements,
        tile_shape=preview.manifest.run.tile_shape,
    )
    fovs_by_id = preview.manifest.fovs_by_id
    planes = {
        fov: _lazy_label_plane(
            getattr(fovs_by_id[fov], family),
            fov=fov,
            family=family,
            expected_shape=preview.manifest.run.tile_shape,
            source_dtype=source_dtype,
            output_dtype=output_dtype,
            flip_x=flip_x,
            flip_y=flip_y,
        )
        for fov in mosaic.fovs
    }
    result = _assemble_raster(block_grid, planes=planes, dtype=output_dtype).rechunk(chunks)
    if result.shape != mosaic.shape:
        raise RuntimeError(f"Constructed CosMx {family} mosaic has shape {result.shape}; expected {mosaic.shape}.")
    return result


def _lazy_label_plane(
    path: Path | None,
    *,
    fov: int,
    family: _LabelFamily,
    expected_shape: tuple[int, int],
    source_dtype: np.dtype,
    output_dtype: np.dtype,
    flip_x: bool,
    flip_y: bool,
) -> da.Array:
    if path is None:
        raise ValueError(f"Cannot construct a lazy CosMx {family} plane without a TIFF path.")
    value = delayed(_read_label_plane, pure=True)(
        path,
        fov,
        family,
        expected_shape,
        source_dtype.name,
        flip_x=flip_x,
        flip_y=flip_y,
    )
    return da.from_delayed(value, shape=expected_shape, dtype=output_dtype)


def _read_label_plane(
    path: Path,
    fov: int,
    family: _LabelFamily,
    expected_shape: tuple[int, int],
    expected_dtype: str,
    *,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Read, revalidate, orient, and interpret one FOV label TIFF."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        if shape != expected_shape:
            raise ValueError(
                f"CosMx {family} TIFF {path} changed shape after discovery: found {shape}, expected {expected_shape}."
            )
        if dtype != np.dtype(expected_dtype):
            raise ValueError(
                f"CosMx {family} TIFF {path} changed dtype after discovery: found {dtype.name}, "
                f"expected {np.dtype(expected_dtype).name}."
            )
        result = np.asarray(series.asarray())

    if result.shape != expected_shape or result.dtype != dtype:
        raise ValueError(
            f"CosMx {family} pixels in {path} have {(result.shape, result.dtype.name)}; expected "
            f"{(expected_shape, dtype.name)}."
        )
    if flip_y:
        result = result[::-1, :]
    if flip_x:
        result = result[:, ::-1]

    if family == _INSTANCE_LABELS_PRODUCT:
        return _remap_instance_ids(result, fov=fov)
    unexpected = sorted(set(np.unique(result).tolist()) - set(_COMPARTMENT_CATEGORIES))
    if unexpected:
        raise ValueError(
            f"CosMx compartment-label TIFF {path} contains unsupported category values {unexpected}; "
            f"expected a subset of {sorted(_COMPARTMENT_CATEGORIES)}."
        )
    return result


def _remap_instance_ids(values: np.ndarray, *, fov: int) -> np.ndarray:
    """Map one FOV's nonzero local IDs into its reserved ``uint32`` range.

    Zero remains background. Every other value becomes
    ``(fov - 1) * base + local_id``, where ``base`` is the complete number of
    values representable by the source dtype.
    """
    base = _instance_id_base(values.dtype)
    offset = (fov - 1) * base
    result = values.astype(_INSTANCE_ID_DTYPE, copy=True)
    foreground = result != 0
    result[foreground] += np.uint32(offset)
    return result


def _validate_label_chunks(chunks: tuple[int, int]) -> None:
    if (
        not isinstance(chunks, tuple)
        or len(chunks) != 2
        or any(not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1 for chunk in chunks)
    ):
        raise ValueError(f"CosMx label chunks must be two positive integers, found {chunks!r}.")


def _labels_element_name(base: str, mosaic: int) -> str:
    return f"{base}_mosaic_{mosaic}"
