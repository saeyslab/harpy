import numpy as np
import pytest
from spatialdata import SpatialData

from sparrow.table._allocation import allocate, bin
from sparrow.utils._keys import _INSTANCE_KEY


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


def test_bin(sdata_bin, filtered_feature_matrix):
    adata = filtered_feature_matrix

    points_layer = "barcodes_location_subset"
    name_barcode_id = "barcode"
    table_layer = "table_custom_bin_32_subset"
    output_table_layer = f"{table_layer}_reproduce"

    df_barcodes_location = sdata_bin.points[points_layer][name_barcode_id].compute()

    # check that barcodes are unique in adata.obs.index and df_barcodes_location
    assert adata.obs.index.is_unique
    assert df_barcodes_location.values.shape == np.unique(df_barcodes_location.values.shape)
    # and check that there is a match between barcodes in adata.obs.index and 'name_barcode_id' column of points layer.
    intersection = np.intersect1d(df_barcodes_location.values, adata.obs.index)
    assert intersection.size > 0

    sdata_bin = bin(
        sdata_bin,
        adata=adata,
        points_layer=points_layer,
        labels_layer="square_labels",
        output_layer=output_table_layer,
        name_barcode_id=name_barcode_id,  # name of barcode in the points layer 'barcodes_location'
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

    assert np.allclose(
        sdata_bin[table_layer].obsm["spatial"], sdata_bin[output_table_layer].obsm["spatial"], rtol=0, atol=1e-5
    )
