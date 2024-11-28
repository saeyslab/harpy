from __future__ import annotations

from collections.abc import Iterable

import dask.array as da
from dask.array import Array
from spatialdata import SpatialData, bounding_box_query
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation
from xarray import DataArray

from harpy.image._image import (
    _get_spatial_element,
    add_image_layer,
)
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def combine(
    sdata: SpatialData,
    img_layer: str,
    output_layer: str,
    nuc_channels: int | str | Iterable[int | str] | None = None,
    mem_channels: int | str | Iterable[int | str] | None = None,
    crd: tuple[int, int, int, int] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Combines specific channels within an image layer of a SpatialData object.

    When given, `nuc_channels` are aggregated together, as are `mem_channels`.

    Parameters
    ----------
    sdata
        Spatial data object containing the image to be combined.
    img_layer
        The image layer in `sdata` to process.
    output_layer
        The name of the output layer where results will be stored.
    nuc_channels
        Specifies which channel(s) to consider as nuclear channels.
    mem_channels
        Specifies which channel(s) to consider as membrane channels.
    crd
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the combined image added to the specified output layer.
    If `nuc_channels` and `mem_channels` is not None, the nuc channels will be at position 0 and the mem channel at position 1.

    Raises
    ------
    ValueError
        If `output_layer` is not provided.
    ValueError
        If no channels are specified for combining.
    ValueError
        If provided arrays are not 2D or 3D (c, (z) , y, x).

    Notes
    -----
    The function combines specified channels from a SpatialData object's image layer, creating a new image layer.
    The provided channels can be specified as nuclear or membrane channels. If coordinates (crd) are specified, only
    the region within those coordinates will be considered for the combination.

    Examples
    --------
    Sum nuclear channels 0 and 1, and keep membrane channel 2 as is from the image layer "raw_image":

    >>> sdata = combine(sdata, img_layer="raw_image", output_layer="combined_image", nuc_channels=[0,1], mem_channels=2)

    Sum only nuclear channels 0 and 1:

    >>> sdata = combine(sdata, img_layer="raw_image", output_layer="nuc_combined", nuc_channels=[0,1])
    """
    se = _get_spatial_element(sdata, layer=img_layer)

    if crd is not None:
        se_crop = bounding_box_query(
            se,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system="global",
        )
        if se_crop is not None:
            se = se_crop
        else:
            log.warning(
                f"Cropped spatial element using crd '{crd}' is None. Falling back to processing on full dataset."
            )

    def _process_channels(
        channels: int | Iterable[int] | None,
        se: DataArray,
    ) -> Array:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

        all_channels = list(se.c.data)
        ch_indices = [all_channels.index(_ch) for _ch in channels if _ch in all_channels]

        if len(ch_indices) == 0:
            raise ValueError(
                f"No matching channels between provided channels '{channels}' and channels in '{img_layer}':  '{all_channels}'."
            )

        arr = se.isel(c=ch_indices).data
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Array is of dimension {arr.shape}, currently only images with 2 or 3 spatial dimensions are supported."
            )
        arr = arr.sum(axis=0)
        arr = arr[None, ...]
        return arr

    results = []

    if nuc_channels is not None:
        nuc_arr = _process_channels(nuc_channels, se)
        results.append(nuc_arr)

    if mem_channels is not None:
        mem_arr = _process_channels(mem_channels, se)
        results.append(mem_arr)

    if len(results) == 2:
        arr = da.concatenate(results, axis=0)
    elif len(results) == 1:
        arr = results[0]
    else:
        raise ValueError(
            f"Please specify either nuc_channels "
            f"(currently set to: {nuc_channels}) or mem_channels (currently set to {mem_channels})."
        )

    sdata = add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformations=get_transformation(se, get_all=True),
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
