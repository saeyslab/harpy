from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._image import _get_spatial_element
from sparrow.image._map import map_image
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import cv2
except ImportError:
    log.warning("'OpenCV (cv2)' not installed, to use 'sp.im.enhance_contrast' please install this library.")


def enhance_contrast(
    sdata: SpatialData,
    img_layer: str | None = None,
    contrast_clip: float | list[float] = 3.5,
    chunks: str | tuple[int, ...] | int | None = 10000,
    depth: tuple[int, ...] | dict[int, int] | int = 3000,
    output_layer: str = "clahe",
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
):
    """
    Enhance the contrast of an image in a SpatialData object.

    Contrast Limited Adaptive Histogram Equalization (CLAHE) is used.
    Compatibility with image layers that have either two or three spatial dimensions (c, (z), y, x).

    Parameters
    ----------
    sdata
        The SpatialData object containing the image to enhance.
    img_layer
        The image layer in `sdata` on which the enhance_contrast function will be applied.
        If not provided, the last image layer in `sdata` is used.
    contrast_clip
        The clip limit for the CLAHE algorithm. Higher values result in stronger contrast enhancement
        but also stronger noise amplification.
        If provided as a list, the length must match the number of channels,
        as the parameter will be used to process the different channels.
    chunks
        Specification for rechunking the data before applying the function.
    depth
        The overlapping depth used in `dask.array.map_overlap`.
        If specified as a tuple or dict, it contains the depth used in 'y' and 'x' dimension.
    output_layer
        The name of the image layer where the enhanced image will be stored.
        The default value is "clahe".
    crd
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    to_coordinate_system
        The coordinate system to which the `crd` is specified. Ignored if `crd` is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True overwrites the element if it already exists.

    Returns
    -------
    An updated `sdata` object with the contrast enhanced image added as a new layer.

    Raises
    ------
    ValueError
        If the dimensions in `img_layer` of `sdata` is not equal to (c,(z),y,x)

    Notes
    -----
    CLAHE is applied to each channel and z-stack (if the image is 3D) of the image separately.
    """

    def _apply_clahe(image: NDArray, contrast_clip: float = 3.5) -> NDArray:
        # input c, z , y, x
        # output c, z, y, x
        # squeeze dimension, and then put it back

        image_dim = image.ndim

        if image_dim == 3:
            if image.shape[0] == 1:
                image = np.squeeze(image, axis=0)
            else:
                raise ValueError("_apply_clahe only accepts c dimension equal to 1.")
        elif image_dim == 4:
            if image.shape[0] == 1 and image.shape[1] == 1:
                image = np.squeeze(image, axis=(0, 1))
            else:
                raise ValueError("_apply_clahe only accepts c and z dimension equal to 1.")
        else:
            raise ValueError("Please provide numpy array containing c,(z),y and x dimension.")

        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))

        # if image.dtype != np.uint16:
        #    image = cv2.normalize(image, None, 0, np.iinfo(np.uint16).max, cv2.NORM_MINMAX).astype(np.uint16)

        image = clahe.apply(image)

        if image_dim == 3:
            image = image[None, ...]
        else:
            image = image[None, None, ...]

        return image

    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, img_layer)

    supported_dtypes = ["uint8", "uint16"]
    if se.dtype not in supported_dtypes:
        raise ValueError(
            f"Contrast enhancing via 'cv2.createCLAHE' is only supported for arrays of dtype '{supported_dtypes}', "
            f"while array is of dtype '{se.dtype}'. Please consider converting to one of these data types first."
        )

    if isinstance(contrast_clip, Iterable):
        assert (
            len(contrast_clip) == len(se.c.data)
        ), f"If 'contrast_clip' is provided as a list, it should match the number of channels in '{se}' ({len(se.c.data)})"
        fn_kwargs = {key: {"contrast_clip": value} for (key, value) in zip(se.c.data, contrast_clip)}
    else:
        fn_kwargs = {"contrast_clip": contrast_clip}

    sdata = map_image(
        sdata,
        img_layer=img_layer,
        output_layer=output_layer,
        func=_apply_clahe,
        fn_kwargs=fn_kwargs,
        chunks=chunks,
        blockwise=True,
        crd=crd,
        to_coordinate_system=to_coordinate_system,
        scale_factors=scale_factors,
        overwrite=overwrite,
        depth=depth,
    )

    return sdata
