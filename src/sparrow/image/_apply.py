from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping

import xarray as xr
from dask.array import Array
from numpy.typing import NDArray
from spatialdata import SpatialData, bounding_box_query, map_raster
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import (
    _add_image_layer,
    _get_spatial_element,
)
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def map_channels_zstacks(
    sdata: SpatialData,
    img_layer: str,
    output_layer: str,
    func: Callable[..., NDArray | Array] | Mapping[str, Any],
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    chunks: str | int | tuple[int, ...] | None = None,
    depth: tuple[int, int] | dict[int, int] | int | None = None,
    blockwise: bool = True,
    crd: tuple[int, int, int, int] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Apply a specified function to an image layer of a SpatialData object.

    Function will be applied to each channel and z stack separately. Input and output dimension of `func` should be (1,1,y,x).

    Parameters
    ----------
    sdata
        Spatial data object containing the image to be processed.
    img_layer
        The image layer in `sdata` to process.
    output_layer
        The name of the output layer where results will be stored.
    func
        The Callable to apply to the image.
        Can also be a Mapping if different Callable should be applied to different z_slices and/or channels
        e.g. { 'channel1': function, 'channel2': function2 ... } or { 'channel': { 'z_slice': function ... } ... }.
        If a Mapping is specified, and fn_kwargs is specified then the Mapping should match
        the Mapping specified in fn_kwargs.
    fn_kwargs
        Keyword arguments to pass to `func`.
        If different fn_kwargs should be passed to different z_slices and or channels, one can specify e.g.
        {'channel1': fn_kwargs1, 'channel2': fn_kwargs2 ... } or { 'channel': { 'z_slice': fn_kwargs ... } ... }.
    chunks
        Specification for rechunking the data before applying the function.
    depth
        The overlapping depth used in `dask.array.map_overlap`. If not `None`, `dask.array.map_overlap` will be used, else `dask.array.map_blocks`.
        If specified as a tuple or dict, it contains the depth used in 'y' and 'x' dimension.
    blockwise
        If `True`, `func` will be distributed with `dask.array.map_overlap` or `dask.array.map_blocks`,
        otherwise `func` is applied to the full data. If `False`, `depth` is ignored.
    crd
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the processed image added to the specified output layer.

    Raises
    ------
    ValueError
        If depth is a Tuple, and does not match (y,x).

    Notes
    -----
    This function is designed for processing images stored in a SpatialData object using Dask for potential
    parallelism and out-of-core computation. Depending on the `chunks` parameter and other settings,
    it can leverage dask's map_overlap or map_blocks functions to apply the desired image processing function.

    Examples
    --------
    Apply a custom function `my_function` to all channels of an image layer using different parameters for each channel
    (we assume sdata[ "raw_image" ] has 2 channels, 0 and 1, and has dimensions c,y,x ):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ 0: { "parameter": 2 }, 1: { "parameter": 3 } }
    >>> sdata = apply(sdata, img_layer="raw_image", output_layer="processed_image", func=my_function, fn_kwargs=fn_kwargs,)

    Apply the same function to all channels of the image with the same parameters:

    >>> fn_kwargs={ "parameter": 2 }
    >>> sdata = apply(sdata, img_layer="raw_image", output_layer="processed_image", func=my_function, fn_kwargs=fn_kwargs,)

    Apply a custom function `my_function` to all z slices of an image layer using different parameters for each z slice
    (we assume sdata[ "raw_image" ] has 2 z slices at 0.5 and 1.5, and has dimensions c,z,y,x ):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ 0.5: { "parameter": 2 }, 1.5: { "parameter": 3 } }
    >>> sdata = apply(sdata, img_layer="raw_image", output_layer="processed_image", func=my_function, fn_kwargs=fn_kwargs,)

    Apply a custom function `my_function` to all z slices and channels of an image layer using different parameters for each z slice
    and channel.
    (we assume sdata[ "raw_image" ] has 2 channels 0 and 1, and 2 z slices at 0.5 and 1.5, and has dimensions c,z,y,x ):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ 0: {  0.5: { "parameter": 2 }, 1.5: { "parameter": 3 } }, 1: {  0.5: { "parameter": 4 }, 1.5: { "parameter": 5 } }  }
    >>> sdata = apply(sdata, img_layer="raw_image", output_layer="processed_image", func=my_function, fn_kwargs=fn_kwargs,)
    """
    se = _get_spatial_element(sdata, img_layer)

    if crd is not None:
        se_crop = bounding_box_query(
            se,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system="global",
        )
        if se_crop is not None:
            se = se_crop
        else:
            log.warning(
                f"Cropped spatial element using crd '{crd}' is None. Falling back to processing on full dataset."
            )

    if chunks is not None:
        se = se.chunk(chunks)

    to_squeeze = False
    # iterate over c and z
    if "z" not in se.dims:
        se = se.expand_dims(z=1).transpose("c", "z", "y", "x")
        to_squeeze = True

    fn_kwargs, func = _precondition(
        func=func, fn_kwargs=fn_kwargs, combine_c=False, combine_z=False, channels=se.c.data, z_slices=se.z.data
    )

    # Adjusting depth here prevents rechunking in map_overlap via the ensure_minimum_chunksize function.
    # This way we do not have unexpected chunk sizes/memory use as result.
    if depth is not None:

        def adjust_depth(depth, chunksize, depth_dim):
            original_depth = depth[depth_dim]
            if chunksize[depth_dim] < original_depth:
                depth[depth_dim] = chunksize[depth_dim] // 4
                log.warning(
                    f"The overlapping depth '{original_depth}' is larger than your chunksize '{chunksize[depth_dim]}' along dimension '{depth_dim}'. "
                    f"Setting depth to 'chunks//4 ({depth[depth_dim]})'"
                )

        chunksize = se.data.chunksize
        if not isinstance(depth, int):
            if len(depth) != se.ndim - 2:
                raise ValueError(
                    f"Depth ({depth}) is provided for {len(depth)} dimensions. "
                    f"Please (only) provide depth for 'y' and 'x' dimension."
                )
            depth = {0: 0, 1: 0, 2: depth[0], 3: depth[1]}
        else:
            depth = {0: 0, 1: 0, 2: depth, 3: depth}
        for dim in range(se.data.ndim):
            adjust_depth(depth, chunksize, dim)

    # process each c and z stack independently
    result = []
    for channel in se.c.data:
        result_z = []
        for z_slice in se.z.data:
            _func = func[channel][z_slice]
            _fn_kwargs = fn_kwargs[channel][z_slice]
            se_c_z = se.sel(c=[channel], z=[z_slice])
            se_c_z = map_raster(
                se_c_z,
                func=_func,
                func_kwargs=_fn_kwargs,
                blockwise=blockwise,
                dims=("c", "z", "y", "x"),
                depth=depth,
            )
            result_z.append(se_c_z)
        result.append(xr.concat(result_z, dim="z"))
    se_result = xr.concat(result, dim="c")

    if to_squeeze:
        se_result = se_result.isel(z=0)

    # rechunk, otherwise could have issues with irregular chunking when saving to zarr
    se_result = se_result.chunk(se_result.data.chunksize)

    sdata = _add_image_layer(
        sdata,
        arr=se_result.data,
        output_layer=output_layer,
        chunks=se_result.data.chunksize,
        transformations=get_transformation(se_result, get_all=True),
        scale_factors=scale_factors,
        c_coords=se_result.c.data,
        overwrite=overwrite,
    )

    return sdata


def _precondition(
    fn_kwargs: dict[Any:Any],
    func: Callable | dict[Any:Any],
    combine_c: bool,
    combine_z: bool,
    channels: list[str] | list[int],
    z_slices: list[float],
):
    def collect_keys_fn_kwargs(d):
        if d == {}:
            return
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value, (dict)):
                result = collect_keys_fn_kwargs(value)
                if result is not None:
                    keys.append(result)
            else:
                return
        return keys

    def collect_keys_func(d):
        if d == {}:
            return
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value, (dict)):
                result = collect_keys_func(value)
                if result is not None:
                    keys.append(result)
            else:
                if not isinstance(value, Callable):
                    raise ValueError("Should specify callable if a dict is provided.")
        return keys

    def flatten_list(nested_list):
        result = []
        for element in nested_list:
            if isinstance(element, list):
                result.extend(flatten_list(element))
            else:
                result.append(element)
        return result

    def make_mapping(keys, value):
        result = {}
        previous_item = None
        for i, item in enumerate(keys):
            if i % 2 == 0:
                if isinstance(item, list):
                    raise ValueError(
                        "'keys' should be a list, with only at uneven positions a list, e.g. [ 0, [0.5, 1.5], 1, [0.5, 1.5] ]."
                    )
            if isinstance(item, list):
                sub_dict = {k: value for k in item}
                result[previous_item] = sub_dict
            else:
                result[item] = value
                previous_item = item
        return result

    if fn_kwargs == {}:
        if isinstance(func, Callable):
            if not combine_z:
                if fn_kwargs == {}:
                    fn_kwargs = {key: fn_kwargs for key in z_slices}
            if not combine_c:
                fn_kwargs = {key: fn_kwargs for key in channels}
        else:
            keys_func = collect_keys_func(func)
            fn_kwargs = make_mapping(keys_func, {})

    keys_fn_kwargs = collect_keys_fn_kwargs(fn_kwargs)
    if isinstance(func, Callable):
        if keys_fn_kwargs is not None:
            # if callable, and parameters are specified for specific channels/z-stacks make a mapping of func
            func = make_mapping(keys_fn_kwargs, func)

    # sanity checks
    if keys_fn_kwargs is not None:
        keys_func = collect_keys_func(func)
        assert keys_fn_kwargs == keys_func, "should specify same keys in 'fn_kwargs' and 'func'"
        keys = set(flatten_list(keys_fn_kwargs))
        # now also do sanity check on the channels and z slices.
        if combine_c and channels is not None:
            if keys.intersection(set(channels)):
                raise ValueError(
                    "Keys in 'fn_kwargs' can not have intersection with names of 'channel' if 'combine_c' is set to True."
                )
        if combine_z and z_slices is not None:
            if keys.intersection(set(z_slices)):
                raise ValueError(
                    "Keys in 'fn_kwargs' can not have intersection with names of 'z_slices' if 'combine_z' is set to True."
                )
        # we do not allow channel/z_slices keys specified in fn_kwargs/func that are not in channels/z_slices
        if not combine_z or not combine_c:
            if not combine_c and channels is not None:
                keys = keys - set(channels)
            if not combine_z and z_slices is not None:
                keys = keys - set(z_slices)
            if keys:
                raise ValueError(
                    f"Some keys in 'fn_kwargs/func' where specified that "
                    f"where not found in specified channels ({channels}) or z_slices ({z_slices})."
                )

    # fix keys
    if not combine_c:
        if not (set(fn_kwargs.keys()).issubset(set(channels)) or set(channels).issubset(set(fn_kwargs.keys()))):
            if not combine_z:
                if not (set(fn_kwargs.keys()).issubset(set(z_slices)) or set(z_slices).issubset(fn_kwargs.keys())):
                    # in this case func should be a callable
                    # if not isinstance( func, Callable ):
                    #    raise ValueError
                    log.info(
                        f"'combine_z' is False, but not all 'z-slices' spefified in 'fn_kwargs'/'func' ({fn_kwargs}/{func}). "
                        f"Specifying z-slices ({z_slices})."
                    )
                    fn_kwargs = {key: fn_kwargs for key in z_slices}
                    func = {key: func for key in z_slices}
            log.info(
                f"'combine_c' is False, but not all channels spefified in 'fn_kwargs'/'func' ({fn_kwargs}/{func}). "
                f"Specifying channels ({channels})."
            )
            fn_kwargs = {key: fn_kwargs for key in channels}
            func = {key: func for key in channels}

        # case where we are subset of channels, but want to add z dimension
        elif not combine_z:
            for _, value in fn_kwargs.items():
                if (
                    not (set(value.keys()).issubset(set(z_slices)) or set(z_slices).issubset(set(value.keys())))
                    or value == {}
                ):
                    log.info(
                        f"'combine_z' is False, but not all 'z-slices' spefified in 'fn_kwargs'/'func' ({fn_kwargs}/{func}). "
                        f"Specifying z-slices ({z_slices})."
                    )
                    fn_kwargs = {key: {z_slice: _value for z_slice in z_slices} for key, _value in fn_kwargs.items()}
                    func = {key: {z_slice: _value for z_slice in z_slices} for key, _value in func.items()}
                    break
    elif not combine_z:
        if not (set(fn_kwargs.keys()).issubset(set(z_slices)) or set(z_slices).issubset(set(fn_kwargs.keys()))):
            log.info(
                f"'combine_z' is False, but not all 'z-slices' spefified in 'fn_kwargs'/'func' ({fn_kwargs}/{func}). "
                f"Specifying z-slices ({z_slices})."
            )
            fn_kwargs = {key: fn_kwargs for key in z_slices}
            func = {key: func for key in z_slices}
    else:
        # sanity check
        assert isinstance(func, Callable)
        assert (
            keys_fn_kwargs is None
        ), f"'combine_z' and 'combine_c' are both set to True, but it seems 'fn_kwargs' ({fn_kwargs}) contains specific parameters for different channels and z slices."

    return fn_kwargs, func
