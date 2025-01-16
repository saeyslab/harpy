"""
[Instanseg](https://github.com/instanseg/instanseg), a pytorch based cell and nucleus segmentation pipeline for fluorescent and brightfield microscopy images. More information here:

Goldsborough, T. et al. (2024) ‘A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg’. bioRxiv, p. 2024.09.04.611150. Available at: https://doi.org/10.1101/2024.09.04.611150.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from instanseg import InstanSeg
from numpy.typing import NDArray

from harpy.image.segmentation._utils import _SEG_DTYPE


def _instanseg(
    img: NDArray,
    device: str | None = "cpu",
    instanseg_model: InstanSeg
    | Path
    | str = "instanseg.pt",  # can be a loaded model, or path to instanseg model (.pt file)
    output: Literal["all_outputs", "nuclei", "cells"] = None,
    dtype: type = _SEG_DTYPE,
    pixel_size: float = 0.5,
    **kwargs,
) -> NDArray:
    # input is z,y,x,c
    # output is z,y,x,c
    """
    Perform instanseg segmentation on an image.

    Parameters
    ----------
    img
        The input image as a NumPy array on which instance segmentation will be performed (z,y,x,c).
    device
        The device to run the model on. Can be "cpu", "cuda", or another supported device.
        Default is "cpu".
    instanseg_model
        The InstaSeg model used for segmentation. This can either be a pre-loaded model, or
        a file path to the model (typically a `.pt` file).
    output
        Specifies the output segmentation type. Options are:
            - "cells": segment entire cells,
            - "nuclei": segment only the nuclei,
            - "all_outputs": segment both cells and nuclei.
        If None, will output `all_outputs`.
    dtype
        The data type for the output mask. Default is set by `_SEG_DTYPE`.

    Returns
    -------
    NDArray
        A NumPy array containing the segmented regions as labeled masks (z,y,x,c).
    """
    if img.shape[0] != 1:
        raise ValueError("Z dimension not equal to 1 is not supported for Instanseg segmentation.")
    img = img.squeeze(0)
    # transpose y,x,c to c,y,x
    img = img.transpose(2, 0, 1)

    if not isinstance(instanseg_model, InstanSeg):
        import torch

        # instanseg_model is the path to the torch jit .pt file.
        instanseg_model = torch.load(instanseg_model)
        instanseg_model = InstanSeg(model_type=instanseg_model, device=device)

    if output is None:
        output = "all_outputs"
    labeled_output, _ = instanseg_model.eval_small_image(
        img,
        pixel_size=pixel_size,
        resolve_cell_and_nucleus=True,
        cleanup_fragments=True,
        target=output,
    )

    # we want the c dimension to be the last dimension and the output to be in numpy format
    labeled_output = labeled_output.permute([0, 2, 3, 1]).cpu().numpy().astype(dtype)
    # already has a trivial z dimension (batch) at 0
    # dimension 1 is (nucleus mask (0) and whole cell mask (1))
    if output == "cells":
        labeled_output = labeled_output[..., 1:2]
    elif output == "nuclei":
        labeled_output = labeled_output[..., 0:1]
    elif output == "all_outputs" or output is None:
        labeled_output = labeled_output
    else:
        raise ValueError(
            f"Invalid value for parameter 'output': '{output}'. Expected one of: 'all_outputs', 'nuclei', 'cells', or None."
        )

    return labeled_output
