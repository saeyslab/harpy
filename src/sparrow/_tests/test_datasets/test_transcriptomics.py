import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel
from spatialdata.transformations import Identity, get_transformation

from sparrow.datasets.transcriptomics import visium_hd_example, xenium_example
from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY


def test_visium_hd_example():
    sdata = visium_hd_example(bin_size=16)
    assert isinstance(sdata, SpatialData)


# skip this unit test, as it downloads a full xenium run experiment in default cache of the os
@pytest.mark.skip
def test_xenium_example():
    sdata = xenium_example(output=None)

    assert "transcripts_global" in sdata.points
    # sparrow only supports points layers with identity transformation defined on them.
    assert get_transformation(sdata["transcripts_global"], to_coordinate_system="global") == Identity()
    assert "table_global" in sdata.tables
    assert "cell_labels_global" in sdata.labels
    assert "nucleus_labels_global" in sdata.labels

    # check that table is annotated by cell_labels_global
    assert ["cell_labels_global"] == sdata["table_global"].obs[_REGION_KEY].cat.categories.to_list()
    # check that instance and region key in table are the sparrow instance and region keys
    assert sdata.tables["table_global"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] == _REGION_KEY
    assert sdata.tables["table_global"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] == _INSTANCE_KEY

    assert isinstance(sdata, SpatialData)
