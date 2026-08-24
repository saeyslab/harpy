from __future__ import annotations

import csv
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from dask.utils import parse_bytes
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale

from harpy.io._cosmx._models import _CosmxPreview
from harpy.io._cosmx._raster import _mosaic_placements, _pixel_coordinate_system
from harpy.points._points import add_points

_SOURCE_COLUMNS = ("V1", "CellComp", "codeclass", "target", "x", "y", "z")
_IGNORED_SOURCE_COLUMNS = frozenset({"CellId", "fov"})
_SOURCE_TO_OUTPUT = {
    "V1": "transcript_id",
    "CellComp": "source_compartment",
    "codeclass": "code_class",
    "target": "gene",
    "z": "source_z",
}
_OUTPUT_COLUMNS = ("transcript_id", "source_compartment", "code_class", "gene", "x", "y", "source_z")
_SOURCE_DTYPES = {
    "V1": "int64",
    "CellComp": "string[pyarrow]",
    "codeclass": "string[pyarrow]",
    "target": "string[pyarrow]",
    "x": "float64",
    "y": "float64",
}


def _add_transcript_points(
    sdata: SpatialData,
    preview: _CosmxPreview,
    *,
    output_points_name: str = "transcripts",
    coordinate_system: str = "global",
    flip_x: bool = True,
    flip_y: bool = False,
    blocksize: str | int = "64MB",
    overwrite: bool = False,
) -> SpatialData:
    """Add one backed, out-of-core transcript points element per mosaic.

    Only transcript CSVs selected by ``preview`` are read. Each FOV's local
    coordinates are oriented exactly like its raster tile and translated into
    its mosaic-local canvas. The resulting Dask DataFrames remain lazy until
    SpatialData writes them as partitioned Parquet inside the backing store.

    The vendor ``CellId`` and CSV ``fov`` columns are deliberately excluded at
    parse time. Instance allocation is a downstream spatial operation against
    the final label raster, while FOV routing comes from the validated manifest.

    Each points element receives a root metadata record containing:

    - ``fovs``: source FOV numbers contributing to the mosaic;
    - ``source_origin_px``: upper-left mosaic bound in the pre-group/source
      pixel coordinate system. This origin is subtracted from every FOV
      position so that the output points use mosaic-local coordinates starting
      at ``(0, 0)``. It is source-geometry metadata, not an active SpatialData
      transformation;
    - ``orientation``: dataset-wide local x/y-axis flips applied before each
      FOV placement is added; and
    - ``pixel_size_um``: physical size of one source x/y coordinate unit.

    Parameters
    ----------
    sdata
        Backed SpatialData object receiving the points elements.
    preview
        Validated selection and mosaic geometry shared with raster ingestion.
    output_points_name
        Base points-element name; ``_mosaic_<n>`` is appended per mosaic.
    coordinate_system
        Base name for independent pixel and physical coordinate systems.
    flip_x, flip_y
        Dataset-wide local-coordinate flips. These must match image and label
        ingestion.
    blocksize
        Positive byte count or Dask byte-size string used to partition each
        source CSV.
    overwrite
        Whether existing points elements with the planned names may be replaced.

    Returns
    -------
    SpatialData
        The input object with one backed transcript element per mosaic.
    """
    if not sdata.is_backed():
        raise ValueError("CosMx transcript ingestion requires a backed SpatialData object.")
    if not preview.mosaics:
        raise ValueError("CosMx transcript ingestion requires at least one selected mosaic.")
    if not output_points_name:
        raise ValueError("CosMx output points base name must not be empty.")
    if not coordinate_system:
        raise ValueError("CosMx coordinate-system base name must not be empty.")
    if not isinstance(overwrite, bool):
        raise ValueError(f"CosMx transcript overwrite must be a bool, found {overwrite!r}.")

    _validate_blocksize(blocksize)
    _validate_transcript_metadata_destination(sdata)

    element_names = tuple(_points_element_name(output_points_name, mosaic.mosaic) for mosaic in preview.mosaics)
    existing = {name: element_type for element_type, name, _ in sdata.gen_elements() if name in element_names}
    wrong_type = sorted(name for name, element_type in existing.items() if element_type != "points")
    if wrong_type:
        raise ValueError(f"CosMx transcript output names already belong to non-points elements: {wrong_type}.")
    collisions = sorted(existing)
    if collisions and not overwrite:
        raise ValueError(f"CosMx transcript points elements already exist: {collisions}.")

    sources = _transcript_sources(preview)
    headers = {fov: _read_transcript_header(path) for fov, path in sources.items()}
    gene_categories = _gene_categories(tuple(sources.values()), blocksize=blocksize)

    for mosaic, element_name in zip(preview.mosaics, element_names, strict=True):
        placements = _mosaic_placements(preview, mosaic)
        frames = [
            _read_fov_transcripts(
                sources[fov],
                header=headers[fov],
                placement=placements[fov],
                tile_shape=preview.manifest.run.tile_shape,
                gene_categories=gene_categories,
                flip_x=flip_x,
                flip_y=flip_y,
                blocksize=blocksize,
            )
            for fov in mosaic.fovs
        ]
        points = frames[0] if len(frames) == 1 else dd.concat(frames, axis=0, interleave_partitions=True)
        pixel_coordinate_system = _pixel_coordinate_system(coordinate_system, mosaic.mosaic)
        micron_coordinate_system = f"{pixel_coordinate_system}_micron"
        sdata = add_points(
            sdata,
            ddf=points,
            output_points_name=element_name,
            coordinates={"x": "x", "y": "y"},
            transformations={
                pixel_coordinate_system: Identity(),
                micron_coordinate_system: Scale(
                    [preview.manifest.run.pixel_size_um, preview.manifest.run.pixel_size_um],
                    axes=("x", "y"),
                ),
            },
            overwrite=overwrite,
        )
    attrs = deepcopy(sdata.attrs)
    cosmx = attrs.setdefault("cosmx", {})
    assert isinstance(cosmx, dict)
    transcripts = cosmx.setdefault("transcripts", {})
    assert isinstance(transcripts, dict)
    for mosaic, element_name in zip(preview.mosaics, element_names, strict=True):
        transcripts[element_name] = {
            "fovs": list(mosaic.fovs),
            "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
            "orientation": {"flip_x": flip_x, "flip_y": flip_y},
            "pixel_size_um": preview.manifest.run.pixel_size_um,
        }
    sdata.attrs = attrs
    sdata.write_attrs()
    return sdata


def _validate_blocksize(blocksize: str | int) -> None:
    """Require a positive Dask-compatible CSV partition size."""
    if isinstance(blocksize, bool) or not isinstance(blocksize, (str, int)):
        raise ValueError(
            f"CosMx transcript blocksize must be a positive integer or byte-size string, found {blocksize!r}."
        )
    try:
        size = parse_bytes(blocksize) if isinstance(blocksize, str) else blocksize
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid CosMx transcript blocksize {blocksize!r}.") from error
    if size < 1:
        raise ValueError(f"CosMx transcript blocksize must be positive, found {blocksize!r}.")


def _transcript_sources(preview: _CosmxPreview) -> dict[int, Path]:
    """Resolve selected transcript sources from the manifest without CSV routing."""
    sources = {}
    fovs_by_id = preview.manifest.fovs_by_id
    for fov in preview.included_fovs:
        path = fovs_by_id[fov].transcripts
        if path is None:
            raise ValueError(f"CosMx included FOV {fov} has no transcript source.")
        sources[fov] = path
    return sources


def _read_transcript_header(path: Path) -> tuple[str, ...]:
    """Read and validate one CSV header without parsing transcript rows."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        try:
            header = tuple(next(csv.reader(file)))
        except StopIteration as error:
            raise ValueError(f"CosMx transcript CSV {path} is empty.") from error
    if not header or any(not column for column in header):
        raise ValueError(f"CosMx transcript CSV {path} has an empty column name in {header}.")
    if len(set(header)) != len(header):
        raise ValueError(f"CosMx transcript CSV {path} has duplicate column names in {header}.")
    missing = sorted(set(_SOURCE_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"CosMx transcript CSV {path} is missing required columns {missing}.")

    extras = set(header) - set(_SOURCE_COLUMNS) - _IGNORED_SOURCE_COLUMNS
    collisions = sorted(extras & set(_OUTPUT_COLUMNS))
    if collisions:
        raise ValueError(
            f"CosMx transcript CSV {path} has additional columns that collide with canonical output names: "
            f"{collisions}."
        )
    return header


def _gene_categories(paths: Sequence[Path], *, blocksize: str | int) -> tuple[str, ...]:
    """Discover one deterministic target vocabulary shared by all mosaics."""
    frames = [
        dd.read_csv(
            path,
            usecols=["target"],
            dtype={"target": "string[pyarrow]"},
            encoding="utf-8-sig",
            blocksize=blocksize,
        )["target"]
        for path in paths
    ]
    targets = frames[0] if len(frames) == 1 else dd.concat(frames, axis=0, interleave_partitions=True)
    values = targets.dropna().drop_duplicates().compute().tolist()
    categories = tuple(sorted(str(value) for value in values))
    if not categories:
        raise ValueError("CosMx transcript sources contain no non-null gene targets.")
    return categories


def _read_fov_transcripts(
    path: Path,
    *,
    header: tuple[str, ...],
    placement: tuple[int, int],
    tile_shape: tuple[int, int],
    gene_categories: tuple[str, ...],
    flip_x: bool,
    flip_y: bool,
    blocksize: str | int,
) -> DaskDataFrame:
    """Build a lazy canonical transcript frame for one manifest-routed FOV."""
    retained = [column for column in header if column not in _IGNORED_SOURCE_COLUMNS]
    dtype = {column: value for column, value in _SOURCE_DTYPES.items() if column in retained}
    raw = dd.read_csv(path, usecols=retained, dtype=dtype, encoding="utf-8-sig", blocksize=blocksize)
    meta = _normalize_transcript_partition(
        raw._meta,
        placement=placement,
        tile_shape=tile_shape,
        gene_categories=gene_categories,
        flip_x=flip_x,
        flip_y=flip_y,
        path=path,
    )
    return raw.map_partitions(
        _normalize_transcript_partition,
        placement=placement,
        tile_shape=tile_shape,
        gene_categories=gene_categories,
        flip_x=flip_x,
        flip_y=flip_y,
        path=path,
        meta=meta,
    )


def _normalize_transcript_partition(
    frame: pd.DataFrame,
    *,
    placement: tuple[int, int],
    tile_shape: tuple[int, int],
    gene_categories: tuple[str, ...],
    flip_x: bool,
    flip_y: bool,
    path: Path,
) -> pd.DataFrame:
    """Rename retained fields and map FOV-local coordinates into a mosaic."""
    source_x = frame["x"].to_numpy(dtype=np.float64, na_value=np.nan)
    source_y = frame["y"].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(source_x).all() or not np.isfinite(source_y).all():
        raise ValueError(f"CosMx transcript CSV {path} contains non-finite x or y coordinates.")
    if frame["target"].isna().any():
        raise ValueError(f"CosMx transcript CSV {path} contains null gene targets.")

    tile_height, tile_width = tile_shape
    if not ((0 <= source_x) & (source_x < tile_width)).all() or not ((0 <= source_y) & (source_y < tile_height)).all():
        raise ValueError(
            f"CosMx transcript CSV {path} contains coordinates outside FOV bounds "
            f"0 <= x < {tile_width} and 0 <= y < {tile_height}."
        )
    placement_y, placement_x = placement
    x = tile_width - 1 - source_x if flip_x else source_x
    y = tile_height - 1 - source_y if flip_y else source_y

    result = frame.drop(columns=["x", "y"]).rename(columns=_SOURCE_TO_OUTPUT)
    result["gene"] = result["gene"].astype(pd.CategoricalDtype(categories=gene_categories))
    result["x"] = x + placement_x
    result["y"] = y + placement_y
    extras = [column for column in result.columns if column not in _OUTPUT_COLUMNS]
    return result[[*_OUTPUT_COLUMNS, *extras]]


def _validate_transcript_metadata_destination(sdata: SpatialData) -> None:
    """Validate the root mappings used for transcript metadata."""
    cosmx = sdata.attrs.get("cosmx")
    if cosmx is not None and not isinstance(cosmx, dict):
        raise ValueError("SpatialData attribute 'cosmx' must be a mapping.")
    transcripts = None if cosmx is None else cosmx.get("transcripts")
    if transcripts is not None and not isinstance(transcripts, dict):
        raise ValueError("SpatialData attribute 'cosmx.transcripts' must be a mapping.")


def _points_element_name(base: str, mosaic: int) -> str:
    return f"{base}_mosaic_{mosaic}"
