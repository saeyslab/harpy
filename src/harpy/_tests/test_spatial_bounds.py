from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import harpy as hp
from harpy._spatial_bounds import _normalize_spatial_bounds


def test_spatial_bounds_names_axes_and_freezes_normalized_bounds():
    x = [0, 1000]
    crop = hp.SpatialBounds(x=x, y=np.array([200, 800]))
    assert crop.x == (0.0, 1000.0)
    assert crop.y == (200.0, 800.0)
    assert crop.z is None
    assert crop.as_tuple() == (0.0, 1000.0, 200.0, 800.0)
    assert all(isinstance(value, float) for value in crop.as_tuple())
    x[0] = 999
    assert crop.x == (0.0, 1000.0)
    with pytest.raises(FrozenInstanceError):
        crop.x = (10, 20)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize(
    "bounds",
    [
        (),
        (1,),
        (0, 1, 2),
        1,
        "12",
        (None, 2),
        (0, np.nan),
        (-np.inf, 0),
        (0, np.inf),
        (2, 1),
        (1, 1),
    ],
)
def test_spatial_bounds_rejects_invalid_bounds_at_construction(axis, bounds):
    kwargs = {"x": (0, 10), "y": (0, 10), axis: bounds}
    with pytest.raises(ValueError, match=f"axis '{axis}'"):
        hp.SpatialBounds(**kwargs)


def test_spatial_bounds_requires_explicit_xy_keywords():
    with pytest.raises(TypeError):
        hp.SpatialBounds((0, 1), (0, 1))
    with pytest.raises(TypeError):
        hp.SpatialBounds(x=(0, 1))
    with pytest.raises(TypeError):
        hp.SpatialBounds(x=(0, 1), y=(0, 1), t=(0, 1))


@pytest.mark.parametrize(
    "value",
    [
        (0, 1, 2),
        (0, 1, 2, 3, 4),
        (0, np.nan, 0, 1),
        (0, 1, 2, 2),
        (0, 1, 0, 1, 2, 2),
        (0, 1, 0, 1, 0, np.inf),
        "0123",
    ],
)
def test_tuple_crops_use_the_same_validation(value):
    with pytest.raises(ValueError, match="crd"):
        _normalize_spatial_bounds(value)


def test_normalized_bounds_are_identical_for_both_forms():
    assert _normalize_spatial_bounds(None) is None
    assert _normalize_spatial_bounds((0, 1000, 200, 800)) == hp.SpatialBounds(x=(0, 1000), y=(200, 800))
    assert _normalize_spatial_bounds((0, 1000, 200, 800, 9, 11)) == hp.SpatialBounds(
        x=(0, 1000), y=(200, 800), z=(9, 11)
    )


def test_optional_z_bounds_are_immutable_and_included_in_tuple():
    z = [9, 11]
    bounds = hp.SpatialBounds(x=(0, 1000), y=(200, 800), z=z)
    z[0] = 100
    assert bounds.z == (9.0, 11.0)
    assert bounds.as_tuple() == (0.0, 1000.0, 200.0, 800.0, 9.0, 11.0)
    assert _normalize_spatial_bounds(bounds) is bounds
    with pytest.raises(FrozenInstanceError):
        bounds.z = None
