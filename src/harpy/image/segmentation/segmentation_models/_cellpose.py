from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA = torch.cuda.is_available()
except ImportError:
    log.warning(
        "Module 'torch' not installed, please install 'torch' if you want to use the callable 'harpy.im.cellpose_callable' as model for 'harpy.im.segment'."
    )
    TORCH_AVAILABLE = False
    CUDA = False

try:
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    log.warning(
        "Module 'cellpose' not installed, please install 'cellpose' (https://github.com/MouseLand/cellpose) if you want to use the callable 'harpy.im.cellpose_callable' as model for 'harpy.im.segment'."
    )
    CELLPOSE_AVAILABLE = False


def cellpose_callable(
    img: NDArray,
    min_size: int = 80,
    cellprob_threshold: int = 0,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    model_type: str = "nuclei",  # ignored if pretrained model is specified.
    pretrained_model: models.CellposeModel | str | Path | None = None,
    channels: list[int] | None = None,
    device: str = "cuda" if CUDA else "cpu",
    z_axis: int = 0,
    channel_axis: int = 3,
    do_3D: bool = False,
    anisotropy: float = 2,
) -> NDArray:
    """
    Perform cell segmentation using the Cellpose model.

    Should be passed to `model` parameter of `harpy.im.segment` for distributed processing.

    Parameters
    ----------
    img
        The input image as a `numpy` array. Dimensions should follow the format (z,y,x,c).
    min_size
        The minimum size (in pixels) of segmented objects. Objects smaller than this will be excluded.
    cellprob_threshold
         all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
    flow_threshold
         flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
    diameter : int, optional
        The estimated diameter of cells (in pixels).
    model_type : str, optional
        The type of model to use for segmentation. Options are "nuclei", "cyto", etc.
        This is ignored if `pretrained_model` is not `None`.
    pretrained_model
        A pretrained Cellpose model to use for segmentation. This can either be a Cellpose model object, or a file path to a saved model. Default is None.
        We recommend passing a Cellpose model object. Otherwise a Cellpose model will be loaded for each call of `cellpose_callable` during distributed processing.
    channels
        List of channels.
        First element of list is the channel to segment.
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to segment grayscale images, input [0,0]. To segment images with cells
        in green and nuclei in blue, input [2,3].
    device
        The device to run the model on. Can be "cpu", "cuda", "mps", or another supported device.
        Default is "cuda" if available, otherwise "cpu".
    z_axis
        The axis representing the z-dimension in the input image. Default is 0.
        Ignored if `do_3D` is `False`.
    channel_axis
        The axis representing the channel dimension in the input image.
    do_3D
        Whether to perform 3D segmentation on the input image.
    anisotropy
        The anisotropy value (ratio of `z`-axis voxel size to `xy` voxel size) for 3D segmentation.
        Ignored if `do_3D` is `False`.

    Returns
    -------
    A `numpy` array containing the segmented regions as labeled masks `(z,y,x,c)`.

    See Also
    --------
    harpy.im.segment : distributed segmentation using `Dask`.
    """
    if channels is None:
        channels = [0, 0]
    if not TORCH_AVAILABLE:
        raise RuntimeError("Module 'torch' is not available. Please install it to use this function.")

    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("Module 'cellpose' is not available. Please install it to use this function.")
    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if pretrained_model is not None:
        if isinstance(pretrained_model, models.CellposeModel):
            model = pretrained_model
        else:
            model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model, device=torch.device(device))
    elif model_type is not None:
        model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    else:
        raise ValueError(
            "Please provide either 'model_type' or 'pretrained_model (i.e. a path to a pretrained model or a loaded Cellpose model of type 'models.CellposeModel')'."
        )
    results = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        z_axis=z_axis,
        channel_axis=channel_axis,
        do_3D=do_3D,
        anisotropy=anisotropy,
    )

    masks = results[0]

    # make sure we always return z,y,x for labels.
    if not do_3D:
        masks = masks[None, ...]

    # add trivial channel dimension, so we return z,y,x,c
    masks = masks[..., None]

    return masks
