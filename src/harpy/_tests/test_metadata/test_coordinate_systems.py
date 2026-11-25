from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Identity, Translation

import harpy as hp
from harpy.metadata import (
    get_coordinate_system_metadata,
    get_element_coordinate_system,
    validate_coordinate_system_metadata,
)
from harpy.metadata._coordinate_systems import _coordinate_system_records
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
def test_get_element_coordinate_system(element_name, expected):
    assert get_element_coordinate_system(_sdata(), element_name) == expected


def test_get_element_coordinate_system_without_registry():
    sdata = _sdata(records=None)
    assert _coordinate_system_records(sdata) == {}
    assert get_element_coordinate_system(sdata, "image") is None


def test_get_element_coordinate_system_rejects_an_element_in_two_coordinate_systems():
    with pytest.raises(
        ValueError, match=r"several coordinate systems with Harpy metadata, \['s1_global_1', 's1_global_2'\]"
    ):
        get_element_coordinate_system(_sdata(), "two_mosaic_labels")


def test_get_element_coordinate_system_rejects_a_table():
    with pytest.raises(ValueError, match="'table' is a table, which has no transformations"):
        get_element_coordinate_system(_sdata(), "table")


def test_get_element_coordinate_system_rejects_an_unknown_element():
    with pytest.raises(ValueError, match="'missing' is not an element of sdata"):
        get_element_coordinate_system(_sdata(), "missing")


def test_registry_that_is_not_a_mapping_raises():
    sdata = _sdata(records=["s1_global_1"])
    with pytest.raises(ValueError, match="'harpy.coordinate_systems' must be a mapping"):
        get_element_coordinate_system(sdata, "image")


def test_public_functions_are_exported_at_the_top_level():
    assert hp.get_coordinate_system_metadata is get_coordinate_system_metadata
    assert hp.get_element_coordinate_system is get_element_coordinate_system
    assert hp.validate_coordinate_system_metadata is validate_coordinate_system_metadata
    assert {
        "get_coordinate_system_metadata",
        "get_element_coordinate_system",
        "validate_coordinate_system_metadata",
    } <= set(hp.__all__)


def test_validate_coordinate_system_metadata_accepts_valid_metadata():
    sdata = _sdata()
    del sdata["two_mosaic_labels"]
    # Labels on a crop, also in a coordinate system without metadata, and labels in no recorded one, are valid.
    validate_coordinate_system_metadata(sdata)


def test_validate_coordinate_system_metadata_accepts_a_store_without_metadata():
    # Without metadata, no element lies in a coordinate system with metadata, so none is ambiguous.
    validate_coordinate_system_metadata(_sdata(records=None))


def test_validate_coordinate_system_metadata_rejects_an_element_in_two_coordinate_systems():
    with pytest.raises(
        ValueError,
        match=r"several coordinate systems with Harpy metadata: 'two_mosaic_labels' to \['s1_global_1', 's1_global_2'\]",
    ):
        validate_coordinate_system_metadata(_sdata())


def test_validate_coordinate_system_metadata_rejects_metadata_of_a_missing_coordinate_system():
    sdata = _sdata(records=RECORDS | {"s1_global_9": {"sample_id": "s1", "mosaic": 9}})
    del sdata["two_mosaic_labels"]
    with pytest.raises(
        ValueError,
        match=r"harpy\.coordinate_systems\['s1_global_9'\] records coordinate system 's1_global_9', to which no "
        r"element of sdata has a transformation",
    ):
        validate_coordinate_system_metadata(sdata)


def test_validate_coordinate_system_metadata_rejects_metadata_that_is_not_a_mapping():
    with pytest.raises(ValueError, match="must map each coordinate system to a mapping, found list for 's1_global_1'"):
        validate_coordinate_system_metadata(_sdata(records={"s1_global_1": [1, 2]}))


def test_get_coordinate_system_metadata():
    records = {
        "s1_global_1": {
            "sample_id": "s1",
            "mosaic": 1,
            "microns_per_unit": 0.12,
            "fovs": [1, 2],
            "fov_bounds": {"1": {"y0": 0, "x0": 0, "y1": 8, "x1": 8}, "2": {"y0": 0, "x0": 8, "y1": 8, "x1": 16}},
        },
        "s1_global_2": {
            "sample_id": "s1",
            "mosaic": 2,
            "microns_per_unit": 0.12,
            "fovs": [3],
            "fov_bounds": {"3": {"y0": 0, "x0": 0, "y1": 8, "x1": 8}},
            "acquisition_timestamp": "20240101_100000_S2",
        },
    }
    sdata = _sdata(records=records)
    attrs = deepcopy(sdata.attrs)

    metadata = get_coordinate_system_metadata(sdata)

    assert metadata.index.name == "coordinate_system"
    assert metadata.index.tolist() == ["s1_global_1", "s1_global_2"]
    # One column per field, in the order the fields first occur; the dtypes are pandas' own.
    assert metadata.columns.tolist() == [
        "sample_id",
        "mosaic",
        "microns_per_unit",
        "fovs",
        "fov_bounds",
        "acquisition_timestamp",
    ]
    assert metadata["mosaic"].dtype == np.int64
    assert metadata["microns_per_unit"].dtype == np.float64
    assert metadata["sample_id"].tolist() == ["s1", "s1"]
    assert metadata["mosaic"].tolist() == [1, 2]
    # Nested values are kept as objects, not flattened; a field that one coordinate system lacks is missing.
    assert metadata.loc["s1_global_1", "fov_bounds"] == records["s1_global_1"]["fov_bounds"]
    assert metadata.loc["s1_global_2", "fovs"] == [3]
    assert pd.isna(metadata.loc["s1_global_1", "acquisition_timestamp"])
    # The nested values are copies: changing them leaves sdata.attrs unchanged.
    metadata.loc["s1_global_1", "fov_bounds"]["1"]["x1"] = 99
    metadata.loc["s1_global_1", "fovs"].append(4)
    assert sdata.attrs == attrs


def test_get_coordinate_system_metadata_without_registry():
    metadata = get_coordinate_system_metadata(_sdata(records=None))

    assert metadata.empty
    assert metadata.index.name == "coordinate_system"
    assert metadata.columns.tolist() == []


def test_get_coordinate_system_metadata_returns_the_fields_as_recorded():
    # Metadata of two writers with different fields: no field is added, and a lacking field is missing.
    records = {"s1_global_1": {"sample_id": "s1", "microns_per_unit": 0.5}, "s1_global_2": {"section": "b"}}

    metadata = get_coordinate_system_metadata(_sdata(records=records))

    assert metadata.columns.tolist() == ["sample_id", "microns_per_unit", "section"]
    assert metadata.loc["s1_global_1", "microns_per_unit"] == 0.5
    assert pd.isna(metadata.loc["s1_global_1", "section"])
    assert pd.isna(metadata.loc["s1_global_2", "sample_id"])


def test_get_coordinate_system_metadata_rejects_metadata_that_is_not_a_mapping():
    with pytest.raises(ValueError, match="must map each coordinate system to a mapping, found list for 's1_global_1'"):
        get_coordinate_system_metadata(_sdata(records={"s1_global_1": [1, 2]}))
