from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import dask.array as da
from dask.array import Array
from dask.array.overlap import coerce_depth
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from sparrow.image._image import _add_image_layer
from sparrow.image._utils import (
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
)
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def apply(
    sdata: SpatialData,
    func: Callable[..., NDArray | Array] | Mapping[str, Any],
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    img_layer: str | None = None,
    output_layer: str | None = None,
    channel: int | Iterable[int] | str | Iterable[str] | None = None,
    z_slice: float | Iterable[float] | None = None,
    combine_c=True,
    combine_z=True,
    chunks: str | int | tuple[int, ...] | None = None,
    output_chunks: tuple[tuple[int, ...], ...] | None = None,
    crd: tuple[int, int, int, int] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Apply a specified function to an image layer of a SpatialData object.

    Parameters
    ----------
    sdata
        Spatial data object containing the image to be processe.
    func
        The Callable to apply to the image.
        Can also be a Mapping if different Callable should be applied different z_slices and/or channels
        e.g. { 'channel1': function, 'channel2': function2 ... } or { 'channel': { 'z_slice': function ... } ... }.
        If a Mapping is specified, and fn_kwargs is specified then the Mapping should match
        the Mapping specified in fn_kwargs.
    fn_kwargs
        Keyword arguments to pass to `func`.
        If different fn_kwargs should be passed to different z_slices and or channels, one can specify e.g.
        {'channel1': fn_kwargs1, 'channel2': fn_kwargs2 ... } or { 'channel': { 'z_slice': fn_kwargs ... } ... }.
    img_layer
        The image layer in `sdata` to process. If not provided, the last image layer in `sdata` is used.
    output_layer
        The name of the output layer where results will be stored. This must be specified.
    channel
        Specifies which channel(s) to run `func` on.
        If None, the `func` is run on all channels if `func` is a Callable,
        and if `func` or `fn_kwargs` is a Mapping, it will run on the specfied channels in `func`/`fn_kwargs` if provided.
    z_slice
        Specifies which z_slice(s) to run `func` on.
        If None, the `func` is run on all z_slices if `func` is a Callable,
        and if `func` or `fn_kwargs` is a Mapping, in will run on the specfied z slices in `func`/`fn_kwargs` if provided.
    combine_c
        If False, each channel is processed indepentently,
        i.e. input to `func`/map_blocks/map_overlap will be of shape (1,(z),y,x).
        If set to True, input to the Callable will depend on the chunk parameter specified for the c dimension.
    combine_z
        If False, each z slice is processed indepentently,
        i.e. input to `func`/map_blocks/map_overlap will be of shape (c,1,y,x).
        If set to True, input to the Callable will depend on the chunk parameter specified for the z dimension.
        Ignored when `img_layer` does not contain a z dimension.
    chunks
        Specification for rechunking the data before applying the function.
        If specified, dask's map_overlap or map_blocks is used depending on the occurence of the "depth" parameter in kwargs.
        If chunks is a Tuple, they should contain desired chunk size for c, (z), y, x.
    output_chunks
        Chunk shape of resulting blocks if the function does not preserve
        shape. If not provided, the resulting array is assumed to have the same
        block structure as the first input array.
        Ignored when chunks is None. Passed to map_overlap/map_blocks as `chunks`.
        E.g. ( (3,), (256,) , (256,)  ).
    crd
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to dask's map_overlap or map_blocks
        depending of the occurence of "depth" in kwargs.

    Returns
    -------
    The `sdata` object with the processed image added to the specified output layer.

    Raises
    ------
    ValueError
        If `output_layer` is not provided.
    ValueError
        If chunks is a Tuple, and do not match (c,(z),y,x).
    ValueError
        If depth is a Tuple, and do not match (c,(z),y,x).

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
    >>> sdata = apply(sdata, my_function, fn_kwargs=fn_kwargs, img_layer="raw_image", output_layer="processed_image", combine_c=False)

    Apply the same function to only the first channel of the image:

    >>> fn_kwargs={ 0: { "parameter": 2 } }
    >>> sdata = apply(sdata, my_function, fn_kwargs=fn_kwargs, img_layer="raw_image", output_layer="processed_image", combine_c=False)

    Apply the same function to all channels of the image with the same parameters:

    >>> fn_kwargs={ "parameter": 2 }
    >>> sdata = apply(sdata, my_function, fn_kwargs=fn_kwargs, img_layer="raw_image", output_layer="processed_image", combine_c=False)

    In the above example, setting combine_c to True results in `my_function` receiving an array with the shape c,y,x,
    whereas setting it to False leads to my_function being supplied with an array shaped as 1,y,x.

    Apply a custom function `my_function` and `my_function2` to channel 0, respectively channel 1 of an image layer
    (we assume sdata[ "raw_image" ] has 2 channels, 0 and 1, and has dimensions c,y,x ):

    >>> def my_function1( image ):
    ...    return image*2
    >>> def my_function2( image ):
    ...    return image+2

    >>> func={ 0: function1, 1: function2 }
    >>> sdata = apply(sdata, func, img_layer="raw_image", output_layer="processed_image", combine_c=False)

    Apply a custom function `my_function` to all z slices of an image layer using different parameters for each z slice
    (we assume sdata[ "raw_image" ] has 2 z slices at 0.5 and 1.5, and has dimensions c,z,y,x ):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ 0.5: { "parameter": 2 }, 1.5: { "parameter": 3 } }
    >>> sdata = apply(sdata, my_function, fn_kwargs=fn_kwargs, img_layer="raw_image", output_layer="processed_image", combine_z=False)

    Apply a custom function `my_function` to all z slices and channels of an image layer using different parameters for each z slice
    and channel.
    (we assume sdata[ "raw_image" ] has 2 channels 0 and 1, and 2 z slices at 0.5 and 1.5, and has dimensions c,z,y,x ):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ 0: {  0.5: { "parameter": 2 }, 1.5: { "parameter": 3 } }, 1: {  0.5: { "parameter": 4 }, 1.5: { "parameter": 5 } }  }
    >>> sdata = apply(sdata, my_function, fn_kwargs=fn_kwargs, img_layer="raw_image", output_layer="processed_image", combine_c=False, combine_z=False)
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    if output_layer is None:
        raise ValueError("Please specify a name for the output layer.")

    def apply_func(
        func: Callable[..., NDArray | Array],
        arr: NDArray | Array,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> Array:
        if chunks is None:
            # if dask array, we want to rechunk,
            # because taking a crop could have caused irregular chunks
            if isinstance(arr, Array):
                arr = arr.rechunk(arr.chunksize)
            arr = func(arr, **fn_kwargs)
            arr = da.asarray(arr)
            # func could have cause irregular chunking
            return arr.rechunk(arr.chunksize)
        if not isinstance(chunks, (int, str)):
            if len(chunks) != arr.ndim:
                raise ValueError(
                    f"Chunks ({chunks}) are provided for {len(chunks)} dimensions. "
                    f"Please (only) provide chunks for {arr.ndim} dimensions."
                )
        if output_chunks is not None:
            kwargs["chunks"] = output_chunks
        arr = da.asarray(arr).rechunk(chunks)
        if "depth" in kwargs:
            kwargs.setdefault("boundary", "reflect")

            # Adjusting depth here prevents rechunking in map_overlap via the ensure_minimum_chunksize function.
            # This way we do not have unexpected chunk sizes/memory use as result.
            def adjust_depth(depth, chunksize, depth_dim):
                original_depth = depth[depth_dim]
                if chunksize[depth_dim] < original_depth:
                    depth[depth_dim] = chunksize[depth_dim] // 4
                    log.warning(
                        f"The overlapping depth '{original_depth}' is larger than your chunksize '{chunksize[depth_dim]}' along dimension '{depth_dim}'. "
                        f"Setting depth to 'chunks//4 ({depth[depth_dim]})'"
                    )

            chunksize = arr.chunksize
            depth = kwargs["depth"]
            if not isinstance(depth, int):
                if len(depth) != arr.ndim:
                    raise ValueError(
                        f"Depth ({depth}) is provided for {len(depth)} dimensions. "
                        f"Please (only) provide depth for {arr.ndim} dimensions."
                    )
            depth = coerce_depth(arr.ndim, depth)

            for dim in range(arr.ndim):
                adjust_depth(depth, chunksize, dim)

            kwargs["depth"] = depth

            arr = da.map_overlap(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        else:
            arr = da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        return arr.rechunk(chunks)

    # get spatial element
    se = _get_spatial_element(sdata, layer=img_layer)

    # here you specify the channels the function is applied on
    if channel is not None:
        channel = list(channel) if isinstance(channel, Iterable) and not isinstance(channel, str) else [channel]
    else:
        channel = se.c.data

    # here you specify the z-stacks the function is applied on
    if "z" in se.dims:
        if z_slice is not None:
            z_slice = list(z_slice) if isinstance(z_slice, Iterable) and not isinstance(z_slice, str) else [z_slice]
        else:
            z_slice = se.z.data
    else:
        z_slice = None
        log.info(
            f"combine_z flag was set to False, while layer '{img_layer}' does not contain a 'z' dimension, "
            "setting 'combine_z' to True for proper processing."
        )
        # trivial combine_z if there is not z dimension
        combine_z = True

    fn_kwargs, func = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=combine_c,
        combine_z=combine_z,
        channels=channel,
        z_slices=z_slice,
    )

    if crd is not None:
        crd = _substract_translation_crd(se, crd)

    if not combine_c:
        # process channels dimensions individually, so get channel names from fn_kwargs, because could have less channels
        channel = list(fn_kwargs.keys())
        if not combine_z:
            result = []
            for key_c, _fn_kwargs_c in fn_kwargs.items():
                func_c = func[key_c]
                result_z = []
                for key_z, _fn_kwargs_c_z in _fn_kwargs_c.items():
                    func_c_z = func_c[key_z]
                    arr = se.sel(z=[key_z], c=[key_c]).data
                    if crd is not None:
                        arr = arr[:, :, crd[2] : crd[3], crd[0] : crd[1]]
                    # TODO decide if we want to squeeze here, and then put them back in next step.
                    # but then we have to be carefull if chunks was passed as tuple.
                    arr = apply_func(func=func_c_z, arr=arr, fn_kwargs=_fn_kwargs_c_z)
                    # arr of size (1,1,y,x)
                    result_z.append(arr[0, 0, ...])
                result.append(da.stack(result_z, axis=0))
            arr = da.stack(result, axis=0)
        else:
            # this combines z. Thus all z dimensions are send to apply func
            result = []
            for key_c, _fn_kwargs_c in fn_kwargs.items():
                func_c = func[key_c]
                if z_slice is not None:
                    arr = se.sel(z=z_slice, c=[key_c]).data
                else:
                    arr = se.sel(c=[key_c]).data
                if crd is not None:
                    if arr.ndim == 3:
                        arr = arr[:, crd[2] : crd[3], crd[0] : crd[1]]
                    elif arr.ndim == 4:
                        arr = arr[:, :, crd[2] : crd[3], crd[0] : crd[1]]
                arr = apply_func(func=func_c, arr=arr, fn_kwargs=_fn_kwargs_c)
                # apply func expects c,z,y,x
                # apply_func returns 1, z, y, x, so we take z,y,x and do a stack over all results to get our c-dimension back
                result.append(arr[0, ...])
            arr = da.stack(result, axis=0)
    else:
        if not combine_z:
            result = []
            for key_z, _fn_kwargs_z in fn_kwargs.items():
                func_z = func[key_z]
                arr = se.sel(z=[key_z], c=channel).data
                if crd is not None:
                    arr = arr[:, :, crd[2] : crd[3], crd[0] : crd[1]]
                arr = apply_func(func=func_z, arr=arr, fn_kwargs=_fn_kwargs_z)
                # apply func returns c,1,y,x, so we take c,y,x and do stack over results to get z-dimension back
                result.append(arr[:, 0, ...])
            arr = da.stack(result, axis=1)

        else:  # case where combine_z is True, and combine_c is True
            # -> just regular fn_kwargs and func passed to apply, but on channels and z_slice that are specified
            if z_slice is not None:
                arr = se.sel(c=channel, z=z_slice).data
            else:
                arr = se.sel(c=channel).data
            if crd is not None:
                if arr.ndim == 3:
                    arr = arr[:, crd[2] : crd[3], crd[0] : crd[1]]
                elif arr.ndim == 4:
                    arr = arr[:, :, crd[2] : crd[3], crd[0] : crd[1]]
            arr = apply_func(func=func, arr=arr, fn_kwargs=fn_kwargs)

    tx, ty = _get_translation(se)

    if crd is not None:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    sdata = _add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformation=translation,
        scale_factors=scale_factors,
        c_coords=channel,
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
