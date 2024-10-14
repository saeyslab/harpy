"""
[Instanseg](https://github.com/instanseg/instanseg), a pytorch based cell and nucleus segmentation pipeline for fluorescent and brightfield microscopy images. More information here:

Goldsborough, T. et al. (2024) ‘A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg’. bioRxiv, p. 2024.09.04.611150. Available at: https://doi.org/10.1101/2024.09.04.611150.
"""

from pathlib import Path
from typing import Literal

from InstanSeg.utils.augmentations import Augmentations
from numpy.typing import NDArray
from torch.jit import RecursiveScriptModule

from sparrow.image.segmentation._utils import _SEG_DTYPE


def _instanseg(
    img: NDArray,
    device: str | None = "cpu",
    instanseg_model: RecursiveScriptModule
    | Path
    | str = "instanseg.pt",  # can be a loaded model, or path to instanseg model (.pt file)
    output: Literal["whole_cell", "nuclei", "all"] = None,
    dtype: type = _SEG_DTYPE,
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
            - "whole_cell": segment entire cells,
            - "nuclei": segment only the nuclei,
            - "all": segment both cells and nuclei.
        If None, will output `all`.
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

    if device is None:
        from InstanSeg.utils.utils import _choose_device

        device = _choose_device()
    if not isinstance(instanseg_model, RecursiveScriptModule):
        import torch

        # instanseg_model is the path to the torch jit .pt file.
        instanseg_model = torch.jit.load(instanseg_model)
        instanseg_model.to(device)

    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(img, normalize=False)  # this converts the input data to a tensor
    # normalize the input tensor
    input_tensor, _ = Augmenter.normalize(input_tensor, percentile=0.1)

    # Run model
    labeled_output = instanseg_model(
        input_tensor.to(device)[None]
    )  # The labeled_output shape should be 1,1,H,W (nucleus or whole cell) or 1,2,H,W (nucleus and whole cell)

    # we want the c dimension to be the last dimension and the output to be in numpy format
    labeled_output = labeled_output.permute([0, 2, 3, 1]).cpu().numpy().astype(dtype)
    # already has a trivial z dimension (batch) at 0
    # dimension 1 is (nucleus mask (0) and whole cell mask (1))
    if output == "whole_cell":
        labeled_output = labeled_output[..., 1:2]
    elif output == "nuclei":
        labeled_output = labeled_output[..., 0:1]
    elif output == "all" or output is None:
        labeled_output = labeled_output
    else:
        raise ValueError(
            f"Invalid value for parameter 'output': '{output}'. Expected one of: 'whole_cell', 'nuclei', 'all', or None."
        )

    return labeled_output
