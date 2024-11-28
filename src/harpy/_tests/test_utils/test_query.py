import os

import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils._query import bounding_box_query


@pytest.mark.parametrize("crd", [[2000, 3000, 3000, 4000], None])
def test_bounding_box_query(sdata_transcripts, tmpdir, crd):
    labels_layer = "segmentation_mask"

    sdata_transcripts_queried = bounding_box_query(
        sdata_transcripts,
        labels_layer=labels_layer,
        to_coordinate_system=None,
        crd=crd,
        output=os.path.join(tmpdir, "sdata_queried.zarr"),
    )
    assert isinstance(sdata_transcripts_queried, SpatialData)
    for _table_name in [*sdata_transcripts_queried.tables]:
        adata = sdata_transcripts_queried.tables[_table_name]
        ids = adata[adata.obs[_REGION_KEY] == labels_layer].obs[_INSTANCE_KEY].values
        labels_queried = da.unique(sdata_transcripts_queried.labels[labels_layer].data).compute()
        labels_queried = labels_queried[labels_queried != 0]

        assert np.all(np.isin(ids, labels_queried))


def test_bounding_box_query_no_annotate(sdata_transcripts, tmpdir):
    labels_layer = "segmentation_mask_expanded"

    sdata_transcripts_queried = bounding_box_query(
        sdata_transcripts,
        labels_layer=labels_layer,
        to_coordinate_system=None,
        crd=None,
        output=os.path.join(tmpdir, "sdata_queried.zarr"),
    )
    assert isinstance(sdata_transcripts_queried, SpatialData)
    assert labels_layer in [*sdata_transcripts_queried.labels]
    # because "segmentation_mask_expanded" does not annotate any tables, no tables will be present in resulting spatialdata object.
    assert not sdata_transcripts_queried.tables


@pytest.mark.parametrize("backed", [True, False])
def test_bounding_box_query_multiple_coordinate_systems(sdata_transcripts_mul_coord, tmpdir, backed):
    labels_layer = [
        "labels_a1_1",
        "labels_a1_2",
    ]
    to_coordinate_system = [
        "a1_1",
        "a1_2",
    ]
    crd = [
        (600, 1000, 2300, 3000),
        (700, 1000, 2300, 2500),
    ]

    sdata_transcripts_queried = bounding_box_query(
        sdata_transcripts_mul_coord,
        labels_layer=labels_layer,
        to_coordinate_system=to_coordinate_system,
        crd=crd,
        output=os.path.join(tmpdir, "sdata_queried.zarr") if backed else None,
    )

    assert isinstance(sdata_transcripts_queried, SpatialData)
    for _labels_layer in labels_layer:
        for _table_name in [*sdata_transcripts_queried.tables]:
            adata = sdata_transcripts_queried.tables[_table_name]
            ids = adata[adata.obs[_REGION_KEY] == _labels_layer].obs[_INSTANCE_KEY].values
            labels_queried = da.unique(sdata_transcripts_queried.labels[_labels_layer].data).compute()
            labels_queried = labels_queried[labels_queried != 0]

            assert np.all(np.isin(ids, labels_queried))


def test_bounding_box_query_multiple_coordinate_systems_crd_none(sdata_transcripts_mul_coord, tmpdir):
    labels_layer = [
        "labels_a1_1",
        "labels_a1_2",
    ]
    to_coordinate_system = [
        "a1_1",
        "a1_2",
    ]
    crd = [
        (
            0,
            50,
            0,
            50,
        ),  # Query will be empty for this crd for labels_layer 'labels_a1_1' -> so all elements that are annoted by labels_a1_1  will be removed from resulting sdata tables.
        None,  # keep all elements annotated by 'labels_a1_2'
    ]

    sdata_transcripts_queried = bounding_box_query(
        sdata_transcripts_mul_coord,
        labels_layer=labels_layer,
        to_coordinate_system=to_coordinate_system,
        crd=crd,
        output=os.path.join(tmpdir, "sdata_queried.zarr"),
    )

    assert isinstance(sdata_transcripts_queried, SpatialData)
    _labels_layer = "labels_a1_1"
    assert _labels_layer not in sdata_transcripts_queried.labels
    for _table_name in [*sdata_transcripts_queried.tables]:
        # check that all elements that are annoted by labels_a1_1  will be removed from resulting sdata tables.
        assert _labels_layer not in sdata_transcripts_queried.tables[_table_name].obs[_REGION_KEY].values
        assert (
            _labels_layer
            not in sdata_transcripts_queried.tables[_table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        )
    _labels_layer = "labels_a1_2"
    for _table_name in [*sdata_transcripts_queried.tables]:
        adata = sdata_transcripts_queried.tables[_table_name]
        ids = adata[adata.obs[_REGION_KEY] == _labels_layer].obs[_INSTANCE_KEY].values
        labels_queried = da.unique(sdata_transcripts_queried.labels[_labels_layer].data).compute()
        labels_queried = labels_queried[labels_queried != 0]

        assert np.all(np.isin(ids, labels_queried))
