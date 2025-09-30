import pytest
from spatialdata import SpatialData

from harpy.table._table import add_table_layer
from harpy.utils._keys import _REGION_KEY


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
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=True,
    )

    userwarning_msg = f"The table is annotating {adata.obs[_REGION_KEY].cat.categories.to_list()[0]}, which is not present in the SpatialData object."

    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)
