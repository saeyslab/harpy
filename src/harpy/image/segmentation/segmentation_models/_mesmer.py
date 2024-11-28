from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from deepcell.applications import Mesmer

    DEEPCELL_AVAILABLE = True
except ImportError:
    DEEPCELL_AVAILABLE = False

if DEEPCELL_AVAILABLE:
    # download model weights from https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz
    app = Mesmer()
else:
    raise RuntimeError("The deepcell module is not available. Please install it to use this function.")


def _mesmer(
    img: NDArray,
    image_mpp: int = 0.2,
    nuc_channel: int | None = None,
    mem_channel: int | None = None,
    compartment: str = "whole-cell",
) -> NDArray:
    if nuc_channel is None and mem_channel is None:
        raise ValueError("Please specify either nuc_channel, mem_channel or both.")

    if nuc_channel is None or mem_channel is None:
        specified_channel = nuc_channel if nuc_channel is not None else mem_channel
        channel_img = img[..., specified_channel][..., None]
        zeros_arr = np.zeros((1, img.shape[1], img.shape[2], 1))
        img = np.concatenate((channel_img, zeros_arr), axis=-1)

    # If both channels are specified
    else:
        nuc_img = img[..., nuc_channel][..., None]
        mem_img = img[..., mem_channel][..., None]
        img = np.concatenate((nuc_img, mem_img), axis=-1)

    # mesmer want img to be of dimension (batch, y, x, c)
    # we give it input (z, y, x, c), with z dim ==1.

    # Mesmer (sometimes) raises errors when img.shape[1]!=img.shape[2], therefore pad with 0
    pad_y = 0
    pad_x = 0
    if img.shape[2] > img.shape[1]:
        pad_y = (img.shape[2] - img.shape[1]) // 2
        pad_size = ((0, 0), (pad_y, pad_y), (0, 0), (0, 0))
        img = np.pad(img, pad_width=pad_size, mode="constant", constant_values=0)
    elif img.shape[1] > img.shape[2]:
        pad_x = (img.shape[1] - img.shape[2]) // 2
        pad_size = ((0, 0), (0, 0), (pad_x, pad_x), (0, 0))
        img = np.pad(img, pad_width=pad_size, mode="constant", constant_values=0)

    masks = app.predict(img, image_mpp=image_mpp, compartment=compartment)

    if pad_y:
        masks = masks[:, pad_y:-pad_y, :, :]
    if pad_x:
        masks = masks[:, :, pad_x:-pad_x, :]

    # mesmer returns dimension (batch, y,x,c), with in our use case batch==1, and c==1.
    # so it is in correct format (z,y,x,c)
    return masks
