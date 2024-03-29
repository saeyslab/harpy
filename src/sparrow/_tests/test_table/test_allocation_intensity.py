import dask.array as da
import numpy as np
import pytest
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.image.segmentation._align_masks import align_labels_layers
from sparrow.table._allocation_intensity import (
    _calculate_intensity,
    allocate_intensity,
)
from sparrow.table._regionprops import add_regionprop_features


def test_integration_allocate_intensity(sdata_multi_c: SpatialData):
    # integration test for process of aligning masks, allocate intensities and add regionprop features to
    # sdata.tables["table_intensities"].obs

    sdata_multi_c = align_labels_layers(
        sdata_multi_c,
        labels_layer_1="masks_nuclear",
        labels_layer_2="masks_whole",
        output_labels_layer="masks_nuclear_aligned",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_nuclear_aligned" in sdata_multi_c.labels

    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        chunks=100,
        append=False,
        overwrite=True,
    )

    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_nuclear_aligned",
        output_layer="table_intensities",
        chunks=100,
        append=True,
        overwrite=True,
    )

    sdata_multi_c = add_regionprop_features(sdata_multi_c, labels_layer="masks_whole", table_layer="table_intensities")

    sdata_multi_c = add_regionprop_features(
        sdata_multi_c, labels_layer="masks_nuclear_aligned", table_layer="table_intensities"
    )

    assert isinstance(sdata_multi_c, SpatialData)

    assert isinstance(sdata_multi_c.tables["table_intensities"], AnnData)

    assert sdata_multi_c.tables["table_intensities"].shape == (1299, 22)


def test_allocate_intensity(sdata_multi_c: SpatialData):
    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        chunks=100,
        append=False,
        overwrite=True,
    )

    assert isinstance(sdata_multi_c, SpatialData)

    assert isinstance(sdata_multi_c.tables["table_intensities"], AnnData)


def test_allocate_intensity_overwrite(sdata_multi_c: SpatialData):
    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        append=False,
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        match=r"Attempting to overwrite sdata\.tables\[table_intensities\], but overwrite is set to False. Set overwrite to True to overwrite the \.zarr store.",
    ):
        # unit test with append to True, and overwrite to False, which should not be allowed
        sdata_multi_c = allocate_intensity(
            sdata_multi_c,
            img_layer="raw_image",
            labels_layer="masks_nuclear_aligned",
            output_layer="table_intensities",
            append=True,
            overwrite=False,
        )


def test_calculate_intensity():
    chunk_size = (2, 2)

    mask_dask_array = da.from_array(
        np.array([[3, 0, 0, 0], [0, 3, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]),
        chunks=chunk_size,
    )

    float_dask_array = da.from_array(
        np.array(
            [
                [0.5, 1.5, 2.5, 3.5],
                [4.5, 5.5, 6.5, 7.5],
                [8.5, 9.5, 10.5, 11.5],
                [12.5, 13.5, 14.5, 15.5],
            ]
        ),
        chunks=chunk_size,
    )

    sum_of_chunks = _calculate_intensity(
        mask_dask_array=mask_dask_array,
        float_dask_array=float_dask_array,
        chunks=chunk_size,
    )

    # intensity from label==0, label==1 and label==3
    expected_result = np.array([[51.5], [70.5], [6.0]])

    assert np.array_equal(sum_of_chunks, expected_result)
