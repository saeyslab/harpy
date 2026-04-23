from spatialdata import SpatialData

from harpy.image._contrast import enhance_contrast


def test_enhance_contrast(sdata_multi_c_no_backed: SpatialData):
    """
    Test enhance_contrast on 3D image with 2 channels.
    """
    sdata_multi_c_no_backed = enhance_contrast(
        sdata_multi_c_no_backed,
        image_name="combine_z_16bit",
        output_image_name="preprocessed_contrast",
        chunks=(1, 1, 200, 200),
        overwrite=True,
    )

    assert "preprocessed_contrast" in sdata_multi_c_no_backed.images
    assert isinstance(sdata_multi_c_no_backed, SpatialData)
