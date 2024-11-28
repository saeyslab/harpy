from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def _watershed(
    img: NDArray,
    thresh: float | None = None,
    geq: bool = True,
    channel: int = 0,
) -> NDArray:
    # input is z,y,x,c
    # output is z,y,x,c

    if img.shape[0] != 1:
        raise ValueError("Z dimension not equal to 1 is not supported for watershed segmentation.")
    img = img.squeeze(0)
    img = img[..., channel]

    # taken from https://squidpy.readthedocs.io/en/stable/_modules/squidpy/im/_segment.html#segment
    if thresh is None:
        thresh = threshold_otsu(img)
    mask = (img >= thresh) if geq else (img < thresh)
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=mask)
    local_maxi = np.zeros(distance.shape, dtype=np.bool_)
    local_maxi[tuple(coords.T)] = True

    markers, _ = ndi.label(local_maxi)

    masks = np.asarray(watershed(-distance, markers, mask=mask))
    # add trivial z and channel dimensions
    masks = masks[None, ..., None]

    return masks
