import numpy as np
import pytest
from spatialdata import SpatialData

from sparrow.table._allocation import allocate, bin_counts
from sparrow.utils._keys import _INSTANCE_KEY, _SPATIAL


def test_allocation(sdata_transcripts: SpatialData):
    assert sdata_transcripts.is_backed()

    sdata_transcripts = allocate(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        output_layer="table_transcriptomics_recompute",
        chunks=1000,
        append=False,
        overwrite=True,
    )

    assert "table_transcriptomics_recompute" in [*sdata_transcripts.tables]
    assert sdata_transcripts["table_transcriptomics_recompute"].shape == (649, 96)

    assert np.array_equal(
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
    assert sdata_transcripts["table_transcriptomics"].shape == (649, 96)

    sdata_transcripts = allocate(
        sdata_transcripts,
        labels_layer="segmentation_mask_expanded",
        output_layer="table_transcriptomics",
        chunks=20000,
        append=True,  # append to existing table
        overwrite=True,
    )

    assert "table_transcriptomics" in [*sdata_transcripts.tables]
    assert sdata_transcripts["table_transcriptomics"].shape == (1302, 96)


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


def test_bin_counts(
    sdata_bin,
):
    table_layer_bins = "square_002um"
    labels_layer = (
        "square_labels_32"  # custom grid to bin the counts of table_layer_bins, can be any segmentation mask.
    )
    table_layer = "table_custom_bin_32"
    output_table_layer = f"{table_layer}_reproduce"

    # check that barcodes are unique in table_layer_bins of sdata_bin
    assert sdata_bin.tables[table_layer_bins].obs.index.is_unique

    sdata_bin = bin_counts(
        sdata_bin,
        table_layer=table_layer_bins,
        labels_layer=labels_layer,
        output_layer=output_table_layer,
        overwrite=True,
        append=False,
    )

    assert np.array_equal(
        sdata_bin[table_layer].obs[_INSTANCE_KEY].values, sdata_bin[output_table_layer].obs[_INSTANCE_KEY].values
    )

    assert np.array_equal(sdata_bin[table_layer].var_names, sdata_bin[output_table_layer].var_names)

    matrix1 = sdata_bin[table_layer].X
    matrix2 = sdata_bin[output_table_layer].X

    assert (matrix1 != matrix2).nnz == 0

    assert np.array_equal(sdata_bin[table_layer].obsm[_SPATIAL], sdata_bin[output_table_layer].obsm[_SPATIAL])
