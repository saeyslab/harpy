from spatialdata import SpatialData

from harpy.image._combine import combine


def test_combine(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = combine(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        output_layer="combine",
        nuc_channels=[15, 14],
        mem_channels=[0, 6, 11],
        overwrite=True,
    )

    assert "combine" in sdata_multi_c_no_backed.images
    assert isinstance(sdata_multi_c_no_backed, SpatialData)
