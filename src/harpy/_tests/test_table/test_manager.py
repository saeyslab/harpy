from spatialdata import SpatialData

from harpy.table._table import add_table_layer
from harpy.utils._keys import _REGION_KEY


def test_add_table_layer(sdata_transcripts: SpatialData, recwarn):
    assert sdata_transcripts.is_backed()

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
