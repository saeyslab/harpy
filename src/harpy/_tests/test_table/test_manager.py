import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._table import add_table_layer
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table_layer(sdata_transcripts: SpatialData, recwarn, is_backed):
    assert sdata_transcripts.is_backed()

    if not is_backed:
        sdata_transcripts.path = None

    adata = sdata_transcripts["table_transcriptomics"]

    sdata_transcripts = add_table_layer(
        sdata_transcripts,
        adata=adata,
        output_layer="table_transcriptomics",
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=True,
    )

    assert (
        _INSTANCE_KEY == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    )
    assert (
        _REGION_KEY == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    )

    assert ["segmentation_mask"] == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][
        TableModel.REGION_KEY
    ]

    userwarning_msg = f"The table is annotating {sdata_transcripts['table_transcriptomics'].obs[_REGION_KEY].cat.categories.to_list()[0]}, which is not present in the SpatialData object."

    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table_layer_change_region_instance_keys(sdata_transcripts: SpatialData, recwarn, is_backed):
    assert sdata_transcripts.is_backed()

    if not is_backed:
        sdata_transcripts.path = None

    adata = sdata_transcripts["table_transcriptomics"]

    # test if we can update the name of the instance and region keys.
    new_instance_key = "instance_key_test"
    new_region_key = "region_key_test"
    adata.obs.rename(
        columns={_REGION_KEY: new_region_key, _INSTANCE_KEY: new_instance_key},
        inplace=True,
    )
    # need to pop the spatialdata_attrs, otherwise harpy will complain that new region key does not match the old region key
    adata.uns.pop(TableModel.ATTRS_KEY, None)

    sdata_transcripts = add_table_layer(
        sdata_transcripts,
        adata=adata,
        output_layer="table_transcriptomics",
        instance_key=new_instance_key,
        region_key=new_region_key,
        region=adata.obs[new_region_key].cat.categories.to_list(),
        overwrite=True,
    )

    assert (
        new_instance_key
        == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    )
    assert (
        new_region_key
        == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    )

    assert ["segmentation_mask"] == sdata_transcripts["table_transcriptomics"].uns[TableModel.ATTRS_KEY][
        TableModel.REGION_KEY
    ]

    userwarning_msg = f"The table is annotating {sdata_transcripts['table_transcriptomics'].obs[new_region_key].cat.categories.to_list()[0]}, which is not present in the SpatialData object."

    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table_layer_not_annotating(sdata_transcripts: SpatialData, is_backed):
    assert sdata_transcripts.is_backed()

    if not is_backed:
        sdata_transcripts.path = None

    adata = sdata_transcripts["table_transcriptomics"]

    sdata_transcripts = add_table_layer(
        sdata_transcripts,
        adata=adata,
        output_layer="table_transcriptomics",
        region=None,  # table is not annotating a region
        overwrite=True,
    )

    assert TableModel.ATTRS_KEY not in sdata_transcripts["table_transcriptomics"].uns
