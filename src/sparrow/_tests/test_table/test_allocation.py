import numpy as np
import pytest
from spatialdata import SpatialData

from sparrow.table._allocation import allocate


def test_allocation(sdata_transcripts: SpatialData):
    assert sdata_transcripts.is_backed()

    sdata_transcripts = allocate(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        output_layer="table_transcriptomics_recompute",
        chunks=20000,
        append=False,
        overwrite=True,
    )

    assert "table_transcriptomics_recompute" in [*sdata_transcripts.tables]
    assert sdata_transcripts["table_transcriptomics_recompute"].shape == (649, 98)

    np.array_equal(
        sdata_transcripts["table_transcriptomics_recompute"].X.toarray(),
        sdata_transcripts["table_transcriptomics"].X.toarray(),
    )


def test_allocation_append(sdata_transcripts: SpatialData):
    assert sdata_transcripts.is_backed()

    sdata_transcripts = allocate(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        output_layer="table_transcriptomics",
        chunks=20000,
        append=False,
        overwrite=True,
    )

    assert "table_transcriptomics" in [*sdata_transcripts.tables]
    assert sdata_transcripts["table_transcriptomics"].shape == (649, 98)

    sdata_transcripts = allocate(
        sdata_transcripts,
        labels_layer="segmentation_mask_expanded",
        output_layer="table_transcriptomics",
        chunks=20000,
        append=True,  # append to existing table
        overwrite=True,
    )

    assert "table_transcriptomics" in [*sdata_transcripts.tables]
    assert sdata_transcripts["table_transcriptomics"].shape == (1302, 98)


def test_allocation_overwrite(sdata_transcripts: SpatialData):
    with pytest.raises(
        ValueError,
        match=r'Attempting to overwrite \'sdata\.tables\["table_transcriptomics"\]\', but overwrite is set to False. Set overwrite to True to overwrite the \.zarr store.',
    ):
        # unit test with append to True, and overwrite to False, which should not be allowed
        sdata_transcripts = allocate(
            sdata_transcripts,
            labels_layer="segmentation_mask",
            output_layer="table_transcriptomics",
            chunks=20000,
            append=False,
            overwrite=False,
        )
