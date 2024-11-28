import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel
from spatialdata.transformations import Identity, get_transformation

from harpy.datasets.transcriptomics import (
    merscope_example,
    merscope_segmentation_masks_example,
    visium_hd_example,
    xenium_example,
)
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def test_visium_hd_example():
    sdata = visium_hd_example(bin_size=16)
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full Xenium run experiment to the OS cache.")
def test_xenium_example():
    sdata = xenium_example(output=None)

    assert "transcripts_global" in sdata.points
    # harpy only supports points layers with identity transformation defined on them.
    assert get_transformation(sdata["transcripts_global"], to_coordinate_system="global") == Identity()
    assert "table_global" in sdata.tables
    assert "cell_labels_global" in sdata.labels
    assert "nucleus_labels_global" in sdata.labels

    # check that table is annotated by cell_labels_global
    assert ["cell_labels_global"] == sdata["table_global"].obs[_REGION_KEY].cat.categories.to_list()
    # check that instance and region key in table are the harpy instance and region keys
    assert sdata.tables["table_global"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] == _REGION_KEY
    assert sdata.tables["table_global"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] == _INSTANCE_KEY

    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full Merscope run experiment to the OS cache.")
def test_merscope_example():
    sdata = merscope_example(output=None, transcripts=False)
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full Merscope run experiment to the OS cache.")
def test_merscope_segmentation_mask_example():
    sdata = merscope_segmentation_masks_example()
    assert isinstance(sdata, SpatialData)
