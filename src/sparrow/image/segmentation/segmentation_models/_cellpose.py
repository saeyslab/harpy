from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA = torch.cuda.is_available()
except ImportError:
    log.warning(
        "Module 'torch' not installed, please install 'torch' if you want to use the callable 'sparrow.image.segmentation.segmentation_models._cellpose' as model for 'sp.im.segment'."
    )
    TORCH_AVAILABLE = False
    CUDA = False

try:
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    log.warning(
        "Module 'cellpose' not installed, please install 'cellpose' if you want to use the callable 'sparrow.image.segmentation.segmentation_models._cellpose' as model for 'sp.im.segment'."
    )
    CELLPOSE_AVAILABLE = False


def _cellpose(
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
            "Please provide either 'model_type' or 'pretrained_model (i.e. a path to a pretrained model or a loaded Cellpose model of type models.CellposeModel)'."
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
