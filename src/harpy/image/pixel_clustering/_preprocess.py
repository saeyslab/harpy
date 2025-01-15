from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable

import dask.array as da
import numpy as np
from dask import compute, persist
from dask.array import Array
from dask_image.ndfilters import gaussian_filter
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element, add_image_layer
from harpy.image._normalize import _nonzero_nonnan_percentile, _nonzero_nonnan_percentile_axis_0
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def pixel_clustering_preprocess(
    sdata: SpatialData,
    img_layer: str | Iterable[str],
    output_layer: str | Iterable[str],
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    q: float | None = 99,
    q_sum: float | None = 5,
    q_post: float = 99.9,
    sigma: float | Iterable[float] | None = 2,
    norm_sum: bool = True,
    cap_max: float | None = None,
    chunks: str | int | tuple[int, ...] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    cast_dtype: type | None = np.float32,
    persist_intermediate: bool = True,
    overwrite: bool = False,
) -> SpatialData:
    """
    Preprocess image layers specified in `img_layer`. Normalizes and blurs the images based on various quantile and gaussian blur parameters. The results are added to `sdata` as specified in `output_layer`.

    Preprocessing function specifically designed for preprocessing images before using `harpy.im.flowsom`.

    Parameters
    ----------
    sdata
        The SpatialData object containing the image data.
    img_layer
        The image layer(s) from `sdata` to process. This can be a single layer or a list of layers, e.g., when multiple fields of view are available.
    output_layer
        The preprocessed images are saved under this layer in `sdata`.
    channels
        Specifies the channels to be included in the processing.
    q
        Quantile used for normalization. If specified, pixel values are normalized by this quantile across the specified channels.
        Each channel is normalized by its own calculated quantile.
    q_sum
        If the sum of the channel values at a pixel is below this quantile, the pixel values across all channels are set to NaN.
    q_post
        Quantile used for normalization after other preprocessing steps (`q`, `q_sum`, `norm_sum` normalization and Gaussian blurring) are performed. If specified, pixel values are normalized by this quantile across the specified channels.
        Each channel is normalized by its own calculated quantile.
    sigma
        Gaussian blur parameter for each channel. Use `0` to omit blurring for specific channels or `None` to skip blurring altogether.
    norm_sum
        If `True`, each channel is normalized by the sum of all channels at each pixel.
    cap_max
        The maximum allowable value for the elements in the resulting preprocessed image layers. If `None`, no capping is applied.
        Typical value would be `1.0` to exclude outliers.
    chunks
        Chunk sizes for processing. If provided as a tuple, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    scale_factors
        Scale factors to apply for multiscale
    persist_intermediate
        If set to `True` will persist all preprocessed elements in `img_layer` in memory.
        If the elements in `img_layer` are large, this could lead to increased ram usage.
        Set to `False` to write to intermediate zarr store instead, which will reduce ram usage, but will increase computation time slightly.
        Persist or writing to intermediate zarr store is needed, otherwise Dask will not be able to optimize the computation graph for the multiple `img_layer` use case.
        Ignored if `sdata` is not backed by a zarr store, or if there is only one element in `img_layer`.
    cast_dtype
        Image data in `img_layer` will be casted to `dtype` before preprocessing starts. If set to None, and input image is of integer type, normalizations will lead to
        data of type `numpy.float64` due to quantile normalizations, leading to increased memory usage.
    overwrite
        If `True`, overwrites existing data in `output_layer`.

    Notes
    -----
    To avoid data leakage:
     - in the single fov case (one image layer provided), to prevent data leakage between channels, one should set `q_sum=None` and `norm_sum=False`, the only normalization that will be performed will then be a division by the `q` and `q_post` quantile values per channel.
     - in the multiple fov case (multiple image layers provided), both `q_sum`, `norm_sum`, `q` and `q_post` should be set to None to prevent data leakage both between channels and between images.

    Returns
    -------
    An updated SpatialData object with the preprocessed image data stored in specified `output_layers`.

    See Also
    --------
    harpy.im.flowsom : flowsom pixel clustering on image layers.

    """
    # setting q_sum =None, and norm_sum=False -> then there will be no data leakage in single fov case.
    img_layer = list(img_layer) if isinstance(img_layer, Iterable) and not isinstance(img_layer, str) else [img_layer]
    output_layer = (
        list(output_layer)
        if isinstance(output_layer, Iterable) and not isinstance(output_layer, str)
        else [output_layer]
    )
    assert len(output_layer) == len(img_layer), (
        "The number of 'output_layer' specified should be the equal to the the number of 'img_layer' specified."
    )

    se_image = _get_spatial_element(sdata, layer=img_layer[0])

    if channels is not None:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]
    else:
        if channels is None:
            channels = se_image.c.data

    _arr_list = []
    _transformations = []
    for i, _img_layer in enumerate(img_layer):
        se_image = _get_spatial_element(sdata, layer=_img_layer)
        _transformations.append(get_transformation(se_image, get_all=True))
        arr = se_image.sel(c=channels).data
        if i == 0:
            _array_dim = arr.ndim
        else:
            assert _array_dim == arr.ndim, (
                "Image layers specified via parameter `img_layer` should all have same number of dimensions."
            )
        if chunks is not None:
            arr = arr.rechunk(chunks)

        to_squeeze = False
        if arr.ndim == 3:
            # add trivial z dimension for 2D case
            arr = arr[:, None, ...]
            to_squeeze = True
        if cast_dtype is not None:
            arr = arr.astype(cast_dtype)
        _arr_list.append(arr)

    if q is not None:
        results_arr_percentile = []
        for _arr in _arr_list:
            # 1) calculate percentiles (excluding nan and 0 from calculation)
            arr_percentile = _nonzero_nonnan_percentile_axis_0(_arr, q=q, dtype=_arr.dtype)
            results_arr_percentile.append(arr_percentile)
        arr_percentile = da.stack(results_arr_percentile, axis=0)
        arr_percentile_mean = da.mean(arr_percentile, axis=0)  # mean over all images
        # arr_percentile_mean has shape (#n_channels,)
        # for multiple img_layer, in ark one uses np.mean( arr_percentile ) as the percentile to normalize

    # 2) calculate norm sum percentile for img_layer
    # now normalize by percentile (percentile for each channel separate)
    if q is not None:
        for i in range(len(_arr_list)):
            _arr_list[i] = _arr_list[i] / da.asarray(arr_percentile_mean[..., None, None, None])

    # sum over all channels for each image
    _arr_sum_list = []
    for _arr in _arr_list:
        _arr_sum_list.append(da.sum(_arr, axis=0))
    # norm_sum_percentile = da.percentile(arr_norm_sum.flatten(), q=q_norm_sum)
    # in ark_analysis, np.quantile is used, which uses 0's for quantile computation, so equivalent would be da.percentile, not sure if we should also use it instead of _nonzeropercentile
    if q_sum is not None:
        results_norm_sum_percentile = []
        for _arr_sum in _arr_sum_list:
            # using da.percentile reproduces exactly results of ark, but nonzero_nonnan feels like a better choice (i.e. case where there is a lot of zero in image)
            # norm_sum_percentile = da.percentile(_arr_sum.flatten(), q=q_sum)
            norm_sum_percentile = _nonzero_nonnan_percentile(_arr_sum, q=q_sum, dtype=_arr_sum.dtype)
            results_norm_sum_percentile.append(norm_sum_percentile)
        # pixel_thresh_val in ark analysis, if multiple images, we take average over all norm_sum_percentile for all images, and we use that value later on.
        norm_sum_percentile = da.stack(results_norm_sum_percentile, axis=0)
        norm_sum_percentile = da.mean(norm_sum_percentile, axis=0)

    # 3) gaussian blur
    if sigma is not None:
        sigma = list(sigma) if isinstance(sigma, Iterable) else [sigma] * len(channels)
        assert len(sigma) == len(channels), (
            f"If 'sigma' is provided as a list, it should match the number of channels in '{se_image}', or the number of channels provided via the 'channels' parameter '{channels}'."
        )
        # gaussian blur for each image separately
        for i in range(len(_arr_list)):
            _arr_list[i] = _gaussian_blur(_arr_list[i], sigma=sigma)
        # recompute the sum over all channels
        _arr_sum_list = []
        for _arr in _arr_list:
            _arr_sum_list.append(da.sum(_arr, axis=0))

    # sanity check
    assert len(_arr_sum_list) == len(_arr_list) == len(img_layer)
    results_arr_percentile_post_norm = []
    for i in range(len(_arr_list)):
        # sanity update to make sure that if pixel at certain location in a channel is nan, all pixels at that location for all channels are set to nan.
        # use fact that sum is nan if one of the values is nan along that axis -> this is necesarry for map_blocks of the _sampling function
        _arr_list[i] = da.where(~da.isnan(_arr_sum_list[i]), _arr_list[i], np.nan)

        # 4) normalize
        # set pixel values for which sum over all channels are below norm_sum_percentile to NaN
        if q_sum is not None:
            _arr_list[i] = da.where(_arr_sum_list[i] > norm_sum_percentile, _arr_list[i], np.nan)
        if norm_sum:
            # recompute the sum (previous step puts all pixel positions below threshold to nan), discard the nans for the sum
            _arr_sum = da.nansum(_arr_list[i], axis=0)
            # for each pixel position, divide by its sum over all channels, if sum is 0 (i.e. if all channels give zero at this pixel position, set it to nan)
            _arr_list[i] = da.where(_arr_sum > 0, _arr_list[i] / _arr_sum, np.nan)

        if q_post is not None:
            arr_percentile_post_norm = _nonzero_nonnan_percentile_axis_0(
                _arr_list[i], q=q_post, dtype=_arr_list[i].dtype
            )
            results_arr_percentile_post_norm.append(arr_percentile_post_norm)

    if q_post is not None:
        arr_percentile_post_norm = da.stack(results_arr_percentile_post_norm, axis=0)
        arr_percentile_post_norm_mean = da.mean(
            arr_percentile_post_norm, axis=0
        )  # arr_percentil_post_mean is of shape (#n_channels,)

    # Now normalize each image layer by arr_percentile_post_norm and add to spatialdata object
    if q_post is not None:
        for i in range(len(_arr_list)):
            _arr_list[i] = _arr_list[i] / da.asarray(arr_percentile_post_norm_mean[..., None, None, None])

    if cap_max is not None:
        for i in range(len(_arr_list)):
            _arr_list[i] = da.clip(_arr_list[i], None, cap_max)

    # need to let dask do optimization of the computation graph in case there are multiple images
    # otherwise will recaclulate whole preprocessing every time we add an image layer to the sdata zarr store
    # should only do this if there is more than one image
    clean_up = False
    if len(_arr_list) > 1:
        if sdata.is_backed() and not persist_intermediate:
            clean_up = True
            _uuid = uuid.uuid4()
            for i, _arr in enumerate(_arr_list):
                _intermediate_zarr_store = os.path.join(os.path.dirname(sdata.path), f"{i}_{_uuid}.zarr")
                log.info(f"Preparing to write to intermediate zarr store {_intermediate_zarr_store}.")
                _arr.to_zarr(_intermediate_zarr_store, compute=False)
            # write to intermediate zarr store, and let dask optimize computation graph
            compute(_arr_list)
            # load them dask array back lazily
            for i in range(len(_arr_list)):
                _arr_list[i] = da.from_zarr(os.path.join(os.path.dirname(sdata.path), f"{i}_{_uuid}.zarr"))
        else:
            _arr_list = persist(*_arr_list)

    # save the preprocessed images, in this way we get the preprocessed images from which we sample
    for i in range(len(_arr_list)):
        sdata = add_image_layer(
            sdata,
            arr=_arr_list[i].squeeze(1) if to_squeeze else _arr_list[i],
            output_layer=output_layer[i],
            transformations=_transformations[i],
            scale_factors=scale_factors,
            c_coords=channels,
            overwrite=overwrite,
        )

    if clean_up:
        # clean up the intermediate zarr store.
        for i in range(len(_arr_list)):
            _intermediate_zarr_store = os.path.join(os.path.dirname(sdata.path), f"{i}_{_uuid}.zarr")
            log.info(f"Removing intermediate zarr store {_intermediate_zarr_store}")
            shutil.rmtree(_intermediate_zarr_store)

    return sdata


def _gaussian_blur(
    arr: Array,
    sigma: list[float],
) -> Array:
    results_gaussian = []
    for _c_arr_norm, _sigma in zip(arr, sigma):
        # run gaussian filter on each z stack independently, otherwise issues with depth when having few z-stacks in gaussian_filter
        results_gaussian_z = []
        for _z_c_arr_norm in _c_arr_norm:
            _z_c_arr_norm = gaussian_filter(_z_c_arr_norm, sigma=_sigma)
            results_gaussian_z.append(_z_c_arr_norm)
        results_gaussian.append(da.stack(results_gaussian_z, axis=0))
    return da.stack(results_gaussian, axis=0)
