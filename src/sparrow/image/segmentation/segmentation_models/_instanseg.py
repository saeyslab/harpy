"""
[Instanseg](https://github.com/instanseg/instanseg), a pytorch based cell and nucleus segmentation pipeline for fluorescent and brightfield microscopy images. More information here:

Goldsborough, T. et al. (2024) ‘A novel channel invariant architecture for the segmentation of cells and nuclei in multiplexed images using InstanSeg’. bioRxiv, p. 2024.09.04.611150. Available at: https://doi.org/10.1101/2024.09.04.611150.
"""

from typing import Literal

from InstanSeg.utils.augmentations import Augmentations
from numpy.typing import NDArray


def _instanseg(
    img: NDArray,
    device: str | None = "cpu",
    instanseg_model=None,
    output: Literal["whole_cell", "nuclei", "all"] = None,
) -> NDArray:
    # input is z,y,x,c
    # output is z,y,x,c

    if img.shape[0] != 1:
        raise ValueError("Z dimension not equal to 1 is not supported for Instanseg segmentation.")
    img = img.squeeze(0)
    # transpose y,x,c to c,y,x
    img = img.transpose(2, 0, 1)

    if device is None:
        from InstanSeg.utils.utils import _choose_device

        device = _choose_device()
    if instanseg_model is None:
        import os

        import torch
        from InstanSeg.utils.utils import download_model

        model_to_download = "fluorescence_nuclei_and_cells"  # or "brightfield_nuclei"
        download_model(model_to_download)
        path_to_torchscript_model = os.environ["INSTANSEG_BIOIMAGEIO_PATH"] + f"{model_to_download}/instanseg.pt"
        # Load the model from torchscript, replace "my_first_instanseg.pt" with the name of your model.
        instanseg_model = torch.jit.load(path_to_torchscript_model)

    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(img, normalize=False)  # this converts the input data to a tensor
    # normalize the input tensor
    input_tensor, _ = Augmenter.normalize(input_tensor, percentile=0.1)

    # Run model
    labeled_output = instanseg_model(
        input_tensor.to(device)[None]
    )  # The labeled_output shape should be 1,1,H,W (nucleus or whole cell) or 1,2,H,W (nucleus and whole cell)

    # we want the c dimension to be the last dimension and the output to be in numpy format
    labeled_output = labeled_output.permute([0, 2, 3, 1]).cpu().numpy().astype("uint32")
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
