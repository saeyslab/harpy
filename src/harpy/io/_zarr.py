from __future__ import annotations

from pathlib import Path

import zarr
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01


def _get_backing_zarr_format(sdata: SpatialData) -> int:
    """Return the Zarr format version of a backed SpatialData store."""
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError("`convert_to_zarr_2()` requires a backed SpatialData object.")

    group = zarr.open_group(store=str(sdata.path), mode="r")
    zarr_format = getattr(getattr(group, "metadata", None), "zarr_format", None)
    if zarr_format is None:
        raise ValueError(f"Could not determine the Zarr format of the backing store at '{sdata.path}'.")
    return zarr_format


def convert_to_zarr_2(
    sdata: SpatialData,
    output: str | Path,
    overwrite: bool = False,
) -> SpatialData:
    """
    Convert a backed Zarr v3 SpatialData object into a Zarr v2 SpatialData store.

    Parameters
    ----------
    sdata
        The SpatialData object to convert. It must be backed by a Zarr v3 store.
    output
        Output path for the converted Zarr v2 SpatialData store.
    overwrite
        If True, overwrite the output path if it already exists.

    Returns
    -------
    The converted SpatialData object reloaded from ``output``.
    """
    source_zarr_format = _get_backing_zarr_format(sdata)
    if source_zarr_format != 3:
        raise ValueError(
            f"`convert_to_zarr_2()` expects a SpatialData object backed by Zarr v3, found Zarr v{source_zarr_format}."
        )

    output = Path(output)
    if output == Path(sdata.path):
        raise ValueError("The output path must differ from the input SpatialData backing store.")

    sdata.write(
        output,
        overwrite=overwrite,
        sdata_formats=SpatialDataContainerFormatV01(),
        update_sdata_path=False,
    )
    return read_zarr(output)
