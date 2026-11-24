import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Identity, Translation

from harpy.metadata._coordinate_systems import _coordinate_system_records, _element_coordinate_system
from harpy.metadata._registry import _COORDINATE_SYSTEMS_METADATA_KEY, _HARPY_METADATA_KEY, _METADATA_REGISTRIES

RECORDS = {
    "s1_global_1": {"sample_id": "s1", "mosaic": 1, "microns_per_unit": 0.12},
    "s1_global_2": {"sample_id": "s1", "mosaic": 2, "microns_per_unit": 0.12},
}


def _labels(transformations):
    return Labels2DModel.parse(np.zeros((4, 4), dtype=np.uint32), dims=("y", "x"), transformations=transformations)


def _sdata(records=RECORDS):
    """Elements of two mosaics, of none, and of both, with the records of the two mosaics."""
    image = Image2DModel.parse(np.zeros((1, 8, 8)), dims=("c", "y", "x"), transformations={"s1_global_1": Identity()})
    labels = {
        # Labels segmented on a crop of the image, also placed in a coordinate system without metadata.
        "crop_labels": _labels({"s1_global_1": Translation([2, 2], axes=("y", "x")), "aligned": Identity()}),
        "unrecorded_labels": _labels({"global": Identity()}),
        "two_mosaic_labels": _labels({"s1_global_1": Identity(), "s1_global_2": Identity()}),
    }
    points = PointsModel.parse(pd.DataFrame({"x": [1.0], "y": [2.0]}), transformations={"s1_global_2": Identity()})
    table = TableModel.parse(
        AnnData(obs=pd.DataFrame({"region": pd.Categorical(["crop_labels"]), "instance_id": [1]}, index=["0"])),
        region="crop_labels",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(images={"image": image}, labels=labels, points={"points": points}, tables={"table": table})
    if records is not None:
        sdata.attrs[_HARPY_METADATA_KEY] = {"metadata_version": 1, _COORDINATE_SYSTEMS_METADATA_KEY: records}
    return sdata


def test_coordinate_systems_is_a_registry():
    assert _COORDINATE_SYSTEMS_METADATA_KEY in _METADATA_REGISTRIES


@pytest.mark.parametrize(
    ("element_name", "expected"),
    [
        ("image", "s1_global_1"),
        ("crop_labels", "s1_global_1"),
        ("points", "s1_global_2"),
        ("unrecorded_labels", None),
    ],
)
def test_element_coordinate_system(element_name, expected):
    assert _element_coordinate_system(_sdata(), element_name) == expected


def test_element_coordinate_system_without_registry():
    sdata = _sdata(records=None)
    assert _coordinate_system_records(sdata) == {}
    assert _element_coordinate_system(sdata, "image") is None


def test_element_in_two_coordinate_systems_raises():
    with pytest.raises(
        ValueError, match=r"several coordinate systems with Harpy metadata, \['s1_global_1', 's1_global_2'\]"
    ):
        _element_coordinate_system(_sdata(), "two_mosaic_labels")


def test_table_raises():
    with pytest.raises(ValueError, match="'table' is a table, which has no transformations"):
        _element_coordinate_system(_sdata(), "table")


def test_unknown_element_raises():
    with pytest.raises(ValueError, match="'missing' is not an element of sdata"):
        _element_coordinate_system(_sdata(), "missing")


def test_registry_that_is_not_a_mapping_raises():
    sdata = _sdata(records=["s1_global_1"])
    with pytest.raises(ValueError, match="'harpy.coordinate_systems' must be a mapping"):
        _element_coordinate_system(sdata, "image")
