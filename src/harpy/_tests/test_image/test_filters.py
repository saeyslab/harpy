from spatialdata import SpatialData

from harpy.image._filters import gaussian_filtering, min_max_filtering


def test_min_max_filtering(sdata_multi_c_no_backed: SpatialData):
    """
    Test min max filtering on 3D image with 2 channels.
    """
    sdata_multi_c_no_backed = min_max_filtering(
        sdata_multi_c_no_backed,
        image_name="combine_z",
        output_image_name="preprocessed_min_max",
        overwrite=True,
    )

    assert "preprocessed_min_max" in sdata_multi_c_no_backed.images
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_gaussian_filtering(sdata_multi_c_no_backed: SpatialData):
    """
    Test gaussian filtering on 3D image with 2 channels.
    """
    sdata_multi_c_no_backed = gaussian_filtering(
        sdata_multi_c_no_backed,
        image_name="combine_z",
        output_image_name="preprocessed_gaussian",
        overwrite=True,
    )

    assert "preprocessed_gaussian" in sdata_multi_c_no_backed.images
    assert isinstance(sdata_multi_c_no_backed, SpatialData)
