from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray
from packaging import version

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
    import cellpose
    from cellpose import models

    CELLPOSE_AVAILABLE = True
except ImportError:
    log.warning(
        "Module 'cellpose' not installed, please install 'cellpose' (https://github.com/MouseLand/cellpose) if you want to use the callable 'harpy.im.cellpose_callable' as model for 'harpy.im.segment'."
    )
    CELLPOSE_AVAILABLE = False


def cellpose_callable(
    img: NDArray,
    batch_size: int = 8,
    channels: list[int] | None = None,
    normalize: bool = True,
    invert: bool = False,
    diameter: int = 55,
    flow_threshold: float = 0.6,
    cellprob_threshold: float = 0.0,
    do_3D: bool = False,
    anisotropy: float = 2,
    flow3D_smooth: int = 0,
    stitch_threshold: float = 0,
    min_size: int = 80,
    max_size_fraction: float = 0.4,
    niter: int | None = None,
    pretrained_model: str | Path = "nuclei",
    device: str = "cuda" if CUDA else "cpu",
) -> NDArray:
    """
    Perform cell segmentation using the Cellpose model.

    Should be passed to `model` parameter of `harpy.im.segment` for distributed processing.

    Parameters
    ----------
    img
        The input image as a `numpy` array. Dimensions should follow the format (z,y,x,c).
    batch_size
        Number of 256x256 (cellpose>=4.0) or 224x224 (cellpose<4.0) patches to run simultaneously.
        (can make smaller or bigger depending on GPU/CPU/MPS memory usage). Defaults to 8.
    channels
        List of channels.
        First element of list is the channel to segment.
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to segment grayscale images, input [0,0]. To segment images with cells
        in green and nuclei in blue, input [2,3].
        `channels` is deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used.
    normalize
        If `True`, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel.
        See documentation of `cellpose.models.CellposeModel.eval` for full description.
    invert
        Invert image pixel intensity before running network. Defaults to `False`.
    diameter
        The estimated diameter of cells (in pixels).
    flow_threshold
        Flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
    cellprob_threshold
        All pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
    do_3D
        Whether to perform 3D segmentation on the input image.
    anisotropy
        The anisotropy value (ratio of `z`-axis voxel size to `xy` voxel size) for 3D segmentation.
        (E.g. set to 2.0 if Z is sampled half as dense as X or Y).
        Ignored if `do_3D` is `False`.
    flow3D_smooth
        If `do_3D` and `flow3D_smooth>0`, smooth flows with gaussian filter of this stddev.
    stitch_threshold
        If `stitch_threshold>0.0` and not `do_3D`, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
    min_size
        The minimum size (in pixels) of segmented objects. Objects smaller than this will be excluded.
    max_size_fraction
        Masks larger than `max_size_fraction` of total image size are removed. Default is 0.4.
    niter
        number of iterations for dynamics computation. if `None`, it is set proportional to the diameter. Defaults to `None`.
    pretrained_model
        A pretrained Cellpose model to use for segmentation. This can either be a model type, e.g. `"nuclei"`, `"cyto"`, `"cyto3"` for Cellpose<4.0 or `"cpsam"` for Cellpose >=4.0;
        or a file path to a saved model on disk. Default is `"nuclei"`.
    device
        The device to run the model on. Can be `"cpu"`, `"cuda"`, `"mps"`, or another supported device.
        Default is `"cuda"` if available, otherwise `"cpu"`.


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

    cellpose_version = version.parse(cellpose.version)

    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()

    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model, device=torch.device(device))

    do_3D_segmentation = True
    if do_3D is False and stitch_threshold == 0:
        assert (
            img.shape[0] == 1
        ), f"If 'do_3D' is set to 'False' and 'stitch_threshold' equals 0, we assume z-dimension is '1', but z dimension of provided image is '{img.shape[0]}'."
        do_3D_segmentation = False
        img = img.squeeze(0)

    # add some checks
    if do_3D is False and stitch_threshold != 0:
        if diameter is not None:
            raise ValueError(
                "Specifying the diameter currently causes a bug in pseudo-3D segmentation in Cellpose (do_3D == False and stitch_threshold != 0)."
            )

    if do_3D is True and stitch_threshold != 0:
        raise ValueError(
            "Please either set 'do_3D' to 'True' (3D segmentation) or 'stitch_threshold!=0' (psuedo 3D segmentation), not both."
        )

    common_args = {
        "x": [img],
        "batch_size": batch_size,
        "channels": channels,
        "channel_axis": 3 if do_3D_segmentation else 2,
        "z_axis": 0 if do_3D_segmentation else None,
        "normalize": normalize,
        "invert": invert,
        "rescale": None,  # not supported in harpy.
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "do_3D": do_3D,
        "anisotropy": anisotropy,
        "stitch_threshold": stitch_threshold,
        "min_size": min_size,
        "max_size_fraction": max_size_fraction,
        "niter": niter,
        "augment": False,
        "tile_overlap": 0.1,
        "compute_masks": True,
    }

    # Add version-specific arguments
    if cellpose_version >= version.parse("3.1.1.1"):
        common_args["flow3D_smooth"] = flow3D_smooth
    else:
        common_args["dP_smooth"] = flow3D_smooth

    results = model.eval(**common_args)

    masks = results[0][0]

    # make sure we always return z,y,x for labels.
    if not do_3D_segmentation:
        masks = masks[None, ...]

    # add trivial channel dimension, so we return z,y,x,c
    masks = masks[..., None]

    return masks
