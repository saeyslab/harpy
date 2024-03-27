from anndata import AnnData
from spatialdata import SpatialData

from sparrow.table._allocation_intensity import allocate_intensity
from sparrow.table._regionprops import add_regionprop_features


def test_allocate_intensity(sdata_multi_c):
    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        chunks=100,
        append=False,
        overwrite=True,
    )
    sdata_multi_c = add_regionprop_features(sdata_multi_c, labels_layer="masks_whole", table_layer="table_intensities")

    assert isinstance(sdata_multi_c, SpatialData)

    assert isinstance(sdata_multi_c.tables["table_intensities"], AnnData)
