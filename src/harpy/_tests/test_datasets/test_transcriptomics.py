import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel
from spatialdata.transformations import Identity, get_transformation

from harpy.datasets.transcriptomics import (
    merscope_mouse_liver,
    merscope_mouse_liver_segmentation_mask,
    visium_hd_example,
    xenium_human_lung_cancer,
    xenium_human_ovarian_cancer,
)
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

# Do not forget to set the pooch cache dir when running these unit tests on an hpc,
# otherwise data will be downloaded in the default cache of the os e.g:
# export HARPY_POOCH_CACHE=/data/groups/technologies/spatial.catalyst/Arne/pooch_cache


@pytest.mark.skip(reason="This test downloads a Visium HD run experiment to the OS cache.")
def test_visium_hd_example():
    sdata = visium_hd_example(bin_size=16)
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full Xenium run experiment to the OS cache.")
def test_xenium_human_lung_cancer(tmp_path):
    sdata = xenium_human_lung_cancer(output=tmp_path / "sdata.zarr")
    assert sdata.is_backed()

    assert "transcripts_global" in sdata.points
    # check that transcripts in "global" have the identity transformation defined on them.
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


@pytest.mark.skip(reason="This test downloads a full Xenium run experiment to the OS cache.")
def test_xenium_human_ovarian_cancer(tmp_path):
    sdata = xenium_human_ovarian_cancer(output=tmp_path / "sdata.zarr")
    assert sdata.is_backed()

    to_coordinate_system = "global_ROI1"

    assert f"transcripts_{to_coordinate_system}" in sdata.points
    # check that transcripts in to_coordinate_system have the identity transformation defined on them.
    assert (
        get_transformation(sdata[f"transcripts_{to_coordinate_system}"], to_coordinate_system=to_coordinate_system)
        == Identity()
    )
    assert f"cell_labels_{to_coordinate_system}" in sdata.labels
    assert f"nucleus_labels_{to_coordinate_system}" in sdata.labels
    assert f"table_{to_coordinate_system}" in sdata.tables

    assert sdata["table_global_ROI1"].shape == (406611, 5101)
    # check annotation of the table by cell labels
    assert [f"cell_labels_{to_coordinate_system}"] == sdata[f"table_{to_coordinate_system}"].obs[
        _REGION_KEY
    ].cat.categories.to_list()
    assert [f"cell_labels_{to_coordinate_system}"] == sdata.tables[f"table_{to_coordinate_system}"].uns[
        TableModel.ATTRS_KEY
    ][TableModel.REGION_KEY]
    assert (
        sdata.tables[f"table_{to_coordinate_system}"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        == _REGION_KEY
    )
    assert (
        sdata.tables[f"table_{to_coordinate_system}"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        == _INSTANCE_KEY
    )


@pytest.mark.skip(reason="This test downloads a full Merscope run experiment to the OS cache.")
def test_merscope_example(tmp_path):
    sdata = merscope_mouse_liver(output=tmp_path / "sdata.zarr", transcripts=False)
    assert sdata.is_backed()
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full Merscope run experiment to the OS cache.")
def test_merscope_segmentation_mask_example():
    sdata = merscope_mouse_liver_segmentation_mask()
    assert isinstance(sdata, SpatialData)
