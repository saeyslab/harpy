from typing import Iterable

import dask.array as da
import numpy as np
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import _get_spatial_element, add_image_layer
from sparrow.image._map import map_image


def normalize(
    sdata: SpatialData,
    img_layer: str,
    output_layer: str,
    q_min: float | list[float] = 5.0,
    q_max: float | list[float] = 95.0,
    eps: float = 1e-20,
    internal_method: str = "tdigest",
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Normalize the intensity of an image layer in a SpatialData object using specified percentiles.

    The normalization can be applied globally or individually to each channel, depending on whether `q_min` and `q_max`
    are provided as single values or as lists. This allows for flexible intensity scaling across multiple channels.

    Parameters
    ----------
    sdata
        SpatialData object.
    img_layer
        The image layer in `sdata` to normalize.
    output_layer
        The name of the output layer where the normalized image will be stored.
    q_min
        The lower percentile for normalization. If provided as a list, the length
        must match the number of channels.
    q_max
        The upper percentile for normalization. If provided as a list, the length
        must match the number of channels.
    eps : float, optional
        A small epsilon value added to the denominator to avoid division by zero. Default is 1e-20.
    internal_method : str, optional
        The method dask uses for computing percentiles. Default is "tdigest". Can be "dask" or "tdigest".
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the element if it already exists.

    Returns
    -------
    The `sdata` object with the normalized image added.

    Raises
    ------
    ValueError
        If `q_min` and `q_max` are provided as lists and their lengths do not match the number of channels.

    Examples
    --------
    Normalize using a single percentile range for all channels:

    >>> sdata = normalize(sdata, img_layer='my_image', output_layer='normalized_image', q_min=5, q_max=95)

    Normalize using different percentile ranges for each channel:

    >>> sdata = normalize(sdata, img_layer='my_image', output_layer='normalized_image', q_min=[5, 10, 15], q_max=[95, 90, 85])
    """
    se = _get_spatial_element(sdata, img_layer)

    # if q_min is Iterable, we apply q_min, q_max normalization to each channel individually
    if isinstance(q_min, Iterable):
        if not isinstance(q_max, Iterable):
            raise ValueError("'q_min' must be an iterable if `q_max` is an iterable.")
        assert (
            len(q_min) == len(q_max) == len(se.c.data)
        ), f"If 'q_min' and 'q_max' is provided as a list, it should match the number of channels in '{se}' ({len(se.c.data)})"
        fn_kwargs = {
            key: {"q_min": q_min_value, "q_max": q_max_value, "eps": eps, "internal_method": internal_method}
            for (key, q_min_value, q_max_value) in zip(se.c.data, q_min, q_max)
        }
        sdata = map_image(
            sdata,
            img_layer=img_layer,
            output_layer=output_layer,
            func=_normalize,
            fn_kwargs=fn_kwargs,
            blockwise=False,
            scale_factors=scale_factors,
            overwrite=overwrite,
        )

    else:
        arr = _normalize(se.data, q_min=q_min, q_max=q_max, eps=eps, internal_method=internal_method)
        sdata = add_image_layer(
            sdata,
            arr=arr,
            output_layer=output_layer,
            transformations=get_transformation(se, get_all=True),
            scale_factors=scale_factors,
            c_coords=se.c.data,
            overwrite=overwrite,
        )

    return sdata


def _normalize(
    arr: da.Array, q_min: float, q_max: float, eps: float = 1e-20, internal_method: str = "tdigest", dtype=np.float32
) -> da.Array:
    mi = _nonzero_nonnan_percentile(arr, q=q_min, internal_method=internal_method, dtype=dtype)
    ma = _nonzero_nonnan_percentile(arr, q=q_max, internal_method=internal_method, dtype=dtype)
    eps = da.asarray(eps, dtype=dtype)

    arr = (arr - mi) / (ma - mi + eps)

    return da.clip(arr, 0, 1)


def _nonzero_nonnan_percentile(
    array: da.Array, q: float, internal_method: str = "tdigest", dtype=np.float32
) -> da.Array:
    """Computes the percentile of a dask array excluding all zeros and nans."""
    array = array.flatten()
    non_zero_non_nan_mask = (array != 0) & (~da.isnan(array))

    array = da.compress(non_zero_non_nan_mask, array)

    return da.percentile(array, q=q, internal_method=internal_method).astype(dtype)[0]


def _nonzero_nonnan_percentile_axis_0(arr: da.Array, q: float, internal_method: str = "tdigest", dtype=np.float32):
    results_percentile = []
    for i in range(arr.shape[0]):
        arr_percentile = _nonzero_nonnan_percentile(arr[i], q=q, internal_method=internal_method, dtype=dtype)
        results_percentile.append(arr_percentile)
    return da.stack(results_percentile, axis=0)
