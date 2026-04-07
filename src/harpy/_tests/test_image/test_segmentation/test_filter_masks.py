import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData

from harpy.image._image import add_labels_layer, get_dataarray
from harpy.image.segmentation._filter_masks import (
    filter_labels_layer,
)


def test_filter_labels_layers(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = filter_labels_layer(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        min_size=100,
        max_size=1000,
        chunks=256,
        output_labels_layer="masks_whole_filtered",
        output_shapes_layer="masks_whole_filtered_boundaries",
        overwrite=True,
    )

    assert "masks_whole_filtered" in sdata_multi_c_no_backed.labels
    assert (
        len(da.unique(sdata_multi_c_no_backed.labels["masks_whole"].data).compute())
        - len(da.unique(sdata_multi_c_no_backed.labels["masks_whole_filtered"].data).compute())
        == 55
    )
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_filter_labels_layer_uses_global_label_size_across_chunks() -> None:
    sdata = SpatialData()
    labels = da.from_array(np.array([[1, 1, 1, 1, 2, 2]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=labels, output_layer="labels", overwrite=True)

    sdata = filter_labels_layer(
        sdata,
        labels_layer="labels",
        min_size=3,
        max_size=10,
        chunks=2,
        output_labels_layer="labels_filtered",
        overwrite=True,
    )

    # Label 1 spans two chunks but has global size 4, so it is kept. Label 2
    # has global size 2 and is filtered out.
    result = get_dataarray(sdata, layer="labels_filtered").data.compute()
    expected = np.array([[1, 1, 1, 1, 0, 0]], dtype=np.uint32)
    assert np.array_equal(result, expected)


def test_filter_labels_layer_raises_for_invalid_size_bounds() -> None:
    sdata = SpatialData()
    labels = da.from_array(np.array([[1, 1], [0, 2]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=labels, output_layer="labels", overwrite=True)

    with pytest.raises(ValueError, match="'min_size' must be <= 'max_size'"):
        filter_labels_layer(
            sdata,
            labels_layer="labels",
            min_size=5,
            max_size=4,
            output_labels_layer="labels_filtered",
            overwrite=True,
        )
