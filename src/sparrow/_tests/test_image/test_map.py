from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from spatialdata import SpatialData

from sparrow.image._map import _precondition, map_image


def _multiply(image: NDArray, parameter: Any):
    image = parameter * image
    return image


def _add(image: NDArray, parameter_add: Any):
    image = image + parameter_add
    return image


def test_precondition():
    fn_kwargs = {"parameter": 4}
    func = _multiply

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=True,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == fn_kwargs
    assert func_post == func

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {"parameter": 4}, 1: {"parameter": 4}}
    assert func_post == {0: _multiply, 1: _multiply}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=True,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0.5: {"parameter": 4}, 1.5: {"parameter": 4}}
    assert func_post == {0.5: _multiply, 1.5: _multiply}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
        1: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
    }
    assert func_post == {
        0: {0.5: _multiply, 1.5: _multiply},
        1: {0.5: _multiply, 1.5: _multiply},
    }

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
    }
    assert func_post == {
        0: {0.5: _multiply, 1.5: _multiply},
    }

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {
        0: {
            0.5: {"parameter": 4},
        },
        1: {
            0.5: {"parameter": 4},
        },
    }
    assert func_post == {
        0: {
            0.5: _multiply,
        },
        1: {
            0.5: _multiply,
        },
    }


def test_precondition_multiple_func():
    fn_kwargs = {0: {"parameter": 4}, 1: {"parameter_add": 10}}
    func = {0: _multiply, 1: _add}

    with pytest.raises(ValueError):
        _, _ = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=True,
            combine_z=True,
            channels=[0, 1],
            z_slices=[0.5, 1.5],
        )

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {"parameter": 4}, 1: {"parameter_add": 10}}
    assert func_post == {0: _multiply, 1: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
        1: {0.5: {"parameter_add": 10}, 1.5: {"parameter_add": 10}},
    }
    assert func_post == {0: {0.5: _multiply, 1.5: _multiply}, 1: {0.5: _add, 1.5: _add}}

    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply, 1.5: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
        1: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
    }
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}

    # if conflicts in channels/z_slices specified in fn_kwargs/func and channels/z_slices,
    # then raise a ValueError to prevent unwanted behaviour.
    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply, 1.5: _add}

    with pytest.raises(ValueError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )

    # fn_kwargs and func should match with keys, if func is a mapping
    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply}

    with pytest.raises(AssertionError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )


def test_precondition_empty_fn_kwargs():
    fn_kwargs = {}
    func = _multiply

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}}, 1: {0.5: {}}}
    assert func_post == {0: {0.5: _multiply}, 1: {0.5: _multiply}}

    fn_kwargs = {}
    func = {0.5: _multiply, 1.5: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}, 1.5: {}}, 1: {0.5: {}, 1.5: {}}}
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}

    fn_kwargs = {}
    func = {0: {0.5: _multiply}, 1: {0.5: _add}}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}}, 1: {0.5: {}}}
    assert func_post == {0: {0.5: _multiply}, 1: {0.5: _add}}

    fn_kwargs = {}
    func = {0.5: _multiply, 1.5: _add}

    # if keys specified they should be in channels or z_slices, otherwise raise ValueError.
    with pytest.raises(ValueError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )

    fn_kwargs = {}
    func = {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _add, 1.5: _add}}
    # keys for channels can not be specified if combine_c is set to True.
    with pytest.raises(ValueError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=True,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5, 1.5],
        )


def test_apply(sdata_multi_c: SpatialData):
    fn_kwargs = {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 8}},
        1: {0.5: {"parameter": 10}, 1.5: {"parameter": 20}},
    }
    func = _multiply

    sdata_multi_c = map_image(
        sdata_multi_c,
        img_layer="combine_z",
        output_layer="combine_z_apply",
        func=func,
        fn_kwargs=fn_kwargs,
        chunks=212,
        overwrite=True,
    )

    res = sdata_multi_c["combine_z"].sel(c=0, z=0.5).compute()
    res2 = sdata_multi_c["combine_z_apply"].sel(c=0, z=0.5).compute()
    assert np.array_equal(res * fn_kwargs[0][0.5]["parameter"], res2)

    res = sdata_multi_c["combine_z"].sel(c=0, z=1.5).compute()
    res2 = sdata_multi_c["combine_z_apply"].sel(c=0, z=1.5).compute()
    assert np.array_equal(res * fn_kwargs[0][1.5]["parameter"], res2)

    res = sdata_multi_c["combine_z"].sel(c=1, z=0.5).compute()
    res2 = sdata_multi_c["combine_z_apply"].sel(c=1, z=0.5).compute()
    assert np.array_equal(res * fn_kwargs[1][0.5]["parameter"], res2)

    res = sdata_multi_c["combine_z"].sel(c=1, z=1.5).compute()
    res2 = sdata_multi_c["combine_z_apply"].sel(c=1, z=1.5).compute()
    assert np.array_equal(res * fn_kwargs[1][1.5]["parameter"], res2)
