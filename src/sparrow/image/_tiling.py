from typing import List, Optional, Tuple

import cv2
import dask.array as da
import jax.numpy as jnp
import numpy as np
import squidpy as sq
from basicpy import BaSiC
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from sparrow.image._image import (
    _add_image_layer,
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
)
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def tiling_correction(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    tile_size: int = 2144,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    output_layer: str = "tiling_correction",
    overwrite: bool = False,
) -> Tuple[SpatialData, List[np.ndarray]]:
    """
    Function corrects for the tiling effect that occurs in some image data (e.g. resolve data).

    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.

    Parameters
    ----------
    sdata
        The SpatialData object containing the image data to correct.
    img_layer
        The image layer in `sdata` to be corrected for tiling effects. If not provided, the last image layer in `sdata` is used.
    tile_size
        The size of the tiles in the image.
    crd
        Coordinates defining the region of the image to correct. It defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    output_layer
        Name of the image layer where the corrected image will be stored in the `sdata` object.
    overwrite
        If True overwrites the element if it already exists.

    Returns
    -------
    Updated `sdata` object containing the corrected image and a list of flatfield arrays with length equal to the number of channels.

    Raises
    ------
    ValueError
        If the image layer does not contain exactly 2 spatial dimensions.
    ValueError
        If the dimensions of the image layer are not multiples of the given tile size.

    Notes
    -----
    The function integrates the BaSiC algorithm for illumination correction and uses OpenCV's inpainting
    to stitch tiles together. It manages the pre- and post-processing of data, translation of coordinates,
    and addition of corrected image results back to the `sdata` object.
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, layer=img_layer)

    if se.dims != ("c", "y", "x"):
        raise ValueError(
            "Tiling correction is only supported for images with 2 spatial dimensions, "
            f"while provided image layer ({img_layer}) has dimensions {se.ndim}."
        )

    if se.sizes["x"] % tile_size or se.sizes["y"] % tile_size:
        raise ValueError(
            f"Spatial Dimension of image layer '{img_layer}' ({se.shape}) on which to run the "
            f"tilingCorrection is not a multiple of the given tile size ({tile_size})."
        )

    # crd is specified on original uncropped pixel coordinates
    # need to substract possible translation, because we use crd to crop imagecontainer, which does not take
    # translation into account
    if crd:
        crd = _substract_translation_crd(se, crd)

    tx, ty = _get_translation(se)

    result_list = []
    flatfields = []

    for channel in se.c.data:
        channel_idx = list(se.c.data).index(channel)
        ic = sq.im.ImageContainer(se.isel(c=channel_idx), layer=img_layer)

        # Create the tiles
        tiles = ic.generate_equal_crops(size=tile_size, as_array=img_layer)
        tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
        black = np.array([1 if ~np.any(tile - 1) else 0 for tile in tiles])

        # create the masks for inpainting
        i_mask = (
            np.block(
                [
                    list(tiles[i : i + (ic.shape[1] // tile_size)])
                    for i in range(0, len(tiles), ic.shape[1] // tile_size)
                ]
            ).astype(np.uint16)
            == 0
        )

        basic = BaSiC(smoothness_flatfield=1)
        basic.fit(tiles)
        if jnp.isnan(basic._reweight_score).item():
            log.warning(
                f"Basicpy model used for illumination correction for channel '{channel}' did not converge. "
                "Illumination correction will be skipped. Continuing with inpainting. Please consider using a larger image ( more tiles )."
            )
            flatfields.append(None)
            tiles_corrected = tiles
        else:
            flatfields.append(basic.flatfield)
            tiles_corrected = basic.transform(tiles)

        tiles_corrected = np.array(
            [tiles[number] if black[number] == 1 else tile for number, tile in enumerate(tiles_corrected)]
        )

        # Stitch the tiles back together
        i_new = np.block(
            [
                list(tiles_corrected[i : i + (ic.shape[1] // tile_size)])
                for i in range(0, len(tiles_corrected), ic.shape[1] // tile_size)
            ]
        ).astype(np.uint16)

        ic = sq.im.ImageContainer(i_new, layer=img_layer)

        ic.add_img(
            i_mask.astype(np.uint8),
            layer="mask_black_lines",
        )

        if crd:
            x0 = crd[0]
            x_size = crd[1] - crd[0]
            y0 = crd[2]
            y_size = crd[3] - crd[2]
            ic = ic.crop_corner(y=y0, x=x0, size=(y_size, x_size))

        # Perform inpainting
        ic.apply(
            {"0": cv2.inpaint},
            layer=img_layer,
            drop=False,
            channel=0,
            new_layer=output_layer,
            copy=False,
            # chunks=10,
            fn_kwargs={
                "inpaintMask": ic.data.mask_black_lines.squeeze().to_numpy(),
                "inpaintRadius": 55,
                "flags": cv2.INPAINT_NS,
            },
        )

        # result for each channel
        result_list.append(ic[output_layer].data)

    # make one dask array of shape (c,y,x)
    result = da.concatenate(result_list, axis=-1).transpose(3, 0, 1, 2).squeeze(-1)

    if crd:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    sdata = _add_image_layer(
        sdata,
        arr=result,
        output_layer=output_layer,
        chunks=result.chunksize,
        transformations={"global": translation} if translation is not None else None,
        scale_factors=scale_factors,
        c_coords=se.c.data,
        overwrite=overwrite,
    )

    return sdata, flatfields
