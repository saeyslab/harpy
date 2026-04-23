import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._manager import _cast_stringdtype_uns_sdata
from harpy.table._table import add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table(sdata_transcripts: SpatialData, recwarn, is_backed):
    assert sdata_transcripts.is_backed()

    if not is_backed:
        sdata_transcripts.path = None

    adata = sdata_transcripts["table_transcriptomics"]

    sdata_transcripts = add_table(
        sdata_transcripts,
        adata=adata,
        output_table_name="table_transcriptomics",
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

    sdata_transcripts = add_table(
        sdata_transcripts,
        adata=adata,
        output_table_name="table_transcriptomics",
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

    sdata_transcripts = add_table(
        sdata_transcripts,
        adata=adata,
        output_table_name="table_transcriptomics",
        region=None,  # table is not annotating a region
        overwrite=True,
    )

    assert TableModel.ATTRS_KEY not in sdata_transcripts["table_transcriptomics"].uns


def test_add_new_backed_table_layer_does_not_warn_about_missing_regions(sdata_transcripts: SpatialData, recwarn):
    adata = sdata_transcripts["table_transcriptomics"].copy()

    sdata_transcripts = add_table(
        sdata_transcripts,
        adata=adata,
        output_table_name="table_transcriptomics_copy",
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=False,
    )

    assert "table_transcriptomics_copy" in sdata_transcripts.tables

    userwarning_msg = (
        f"The table is annotating {adata.obs[_REGION_KEY].cat.categories.to_list()[0]!r}, "
        "which is not present in the SpatialData object."
    )
    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


def _string_dtype_array(values: list[str]) -> np.ndarray:
    if not hasattr(np, "dtypes") or not hasattr(np.dtypes, "StringDType"):
        pytest.skip("NumPy StringDType is not available in this NumPy version.")
    return np.asarray(values, dtype=np.dtypes.StringDType())


def test_cast_stringdtype_uns_sdata_keeps_u7_when_values_fit(sdata_transcripts_no_backed: SpatialData):
    key = "category_labels"
    sdata_transcripts_no_backed["table_transcriptomics"].uns[key] = _string_dtype_array(["#112233", "#aabbcc"])

    _cast_stringdtype_uns_sdata(sdata_transcripts_no_backed, target_dtype="U7")

    values = sdata_transcripts_no_backed["table_transcriptomics"].uns[key]
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.dtype("U7")
    assert values.tolist() == ["#112233", "#aabbcc"]


def test_cast_stringdtype_uns_sdata_leaves_top_level_string_lists_untouched(sdata_transcripts_no_backed: SpatialData):
    key = "metadata"
    sdata_transcripts_no_backed["table_transcriptomics"].uns[key] = ["#11223344", "darkgreen"]

    _cast_stringdtype_uns_sdata(sdata_transcripts_no_backed, target_dtype="U7")

    values = sdata_transcripts_no_backed["table_transcriptomics"].uns[key]
    assert isinstance(values, list)
    assert values == ["#11223344", "darkgreen"]
