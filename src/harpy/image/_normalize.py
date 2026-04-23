from __future__ import annotations

from collections.abc import Iterable

import dask.array as da
import numpy as np
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element, add_image
from harpy.image._map import map_image


def normalize(
    sdata: SpatialData,
    image_name: str,
    output_image_name: str,
    p_min: float | list[float] = 5.0,
    p_max: float | list[float] = 95.0,
    eps: float = 1e-20,
    internal_method: str = "tdigest",
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Normalize the intensity of an image element in a SpatialData object using specified percentiles.

    The normalization can be applied globally or individually to each channel, depending on whether `p_min` and `p_max`
    are provided as single values or as lists. This allows for flexible intensity scaling across multiple channels.

    Parameters
    ----------
    sdata
        SpatialData object.
    image_name
        The image element in `sdata` to normalize.
    output_image_name
        The name of the output element where the normalized image will be stored.
    p_min
        The lower percentile for normalization. If provided as a list, the length
        must match the number of channels.
    p_max
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
        If `p_min` and `p_max` are provided as lists and their lengths do not match the number of channels.

    Examples
    --------
    Normalize using a single percentile range for all channels:

    >>> sdata = normalize(sdata, image_name='my_image', output_image_name='normalized_image', p_min=5, p_max=95)

    Normalize using different percentile ranges for each channel:

    >>> sdata = normalize(sdata, image_name='my_image', output_image_name='normalized_image', p_min=[5, 10, 15], p_max=[95, 90, 85])
    """
    se = _get_spatial_element(sdata, image_name)

    # if p_min is Iterable, we apply p_min, p_max normalization to each channel individually
    if isinstance(p_min, Iterable):
        if not isinstance(p_max, Iterable):
            raise ValueError("'p_min' must be an iterable if `p_max` is an iterable.")
        assert len(p_min) == len(p_max) == len(se.c.data), (
            f"If 'p_min' and 'p_max' is provided as a list, it should match the number of channels in '{se}' ({len(se.c.data)})"
        )
        fn_kwargs = {
            key: {"p_min": p_min_value, "p_max": p_max_value, "eps": eps, "internal_method": internal_method}
            for (key, p_min_value, p_max_value) in zip(se.c.data, p_min, p_max, strict=True)
        }
        sdata = map_image(
            sdata,
            image_name=image_name,
            output_image_name=output_image_name,
            func=_normalize,
            fn_kwargs=fn_kwargs,
            blockwise=False,
            scale_factors=scale_factors,
            overwrite=overwrite,
        )

    else:
        arr = _normalize(se.data, p_min=p_min, p_max=p_max, eps=eps, internal_method=internal_method)
        sdata = add_image(
            sdata,
            arr=arr,
            output_image_name=output_image_name,
            transformations=get_transformation(se, get_all=True),
            scale_factors=scale_factors,
            c_coords=se.c.data,
            overwrite=overwrite,
        )

    return sdata


def _normalize(
    arr: da.Array, p_min: float, p_max: float, eps: float = 1e-20, internal_method: str = "tdigest", dtype=np.float32
) -> da.Array:
    mi = _nonzero_nonnan_percentile(arr, q=p_min, internal_method=internal_method, dtype=dtype)
    ma = _nonzero_nonnan_percentile(arr, q=p_max, internal_method=internal_method, dtype=dtype)
    eps = da.asarray(eps, dtype=dtype)

    arr = (arr - mi) / (ma - mi + eps)

    return da.clip(arr, 0, 1)


def _nonzero_nonnan_percentile(
    array: da.Array, q: float, internal_method: str = "tdigest", dtype=np.float32
) -> da.Array:
    """Computes the percentile of a dask array excluding all zeros and nans."""
    array = array.ravel()
    non_zero_non_nan_mask = (array != 0) & (~da.isnan(array))

    array = da.compress(non_zero_non_nan_mask, array)

    result = da.percentile(array, q=q, internal_method=internal_method).astype(dtype)
    # Dask returns a 0-D array for scalar q in newer versions and a length-1
    # array in older versions; support both without indexing errors.
    if result.ndim == 0:
        return result
    return result[0]


def _nonzero_nonnan_percentile_axis_0(arr: da.Array, q: float, internal_method: str = "tdigest", dtype=np.float32):
    results_percentile = []
    for i in range(arr.shape[0]):
        arr_percentile = _nonzero_nonnan_percentile(arr[i], q=q, internal_method=internal_method, dtype=dtype)
        results_percentile.append(arr_percentile)
    return da.stack(results_percentile, axis=0)
