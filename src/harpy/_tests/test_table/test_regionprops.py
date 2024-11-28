from anndata import AnnData
from spatialdata import SpatialData

from sparrow.table._allocation_intensity import allocate_intensity
from sparrow.table._regionprops import add_regionprop_features


def test_allocate_intensity(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = allocate_intensity(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        chunks=100,
        append=False,
        overwrite=True,
    )
    sdata_multi_c_no_backed = add_regionprop_features(
        sdata_multi_c_no_backed, labels_layer="masks_whole", table_layer="table_intensities"
    )

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    assert isinstance(sdata_multi_c_no_backed.tables["table_intensities"], AnnData)
