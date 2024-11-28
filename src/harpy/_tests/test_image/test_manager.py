import pytest
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element, add_image_layer, add_labels_layer


@pytest.mark.parametrize(
    "scale_factors",
    [
        (None),  # Standard test case
        ([2, 2, 2, 2]),  # Multi-scale test case
    ],
)
def test_add_image_layer_backed(
    sdata_multi_c,
    scale_factors,
):
    name = "raw_image"
    new_name = f"{name}_processed"

    # Modify the data
    arr = sdata_multi_c[name].data
    arr = arr + 1

    # Add the image layer
    sdata_multi_c = add_image_layer(
        sdata_multi_c,
        arr=arr,
        output_layer=new_name,
        scale_factors=scale_factors,
        overwrite=True,
    )

    # Assertions for the backed status and presence of the layer
    assert sdata_multi_c.is_backed()
    assert new_name in [*sdata_multi_c.images]

    # Check if the layer contains non-zero elements
    assert sdata_multi_c[new_name].any().compute()

    # Verify materialization of dask graph layers
    se = _get_spatial_element(sdata_multi_c, layer=new_name)
    for _, layer in se.data.__dask_graph__().layers.items():
        assert layer.is_materialized()


# no backed
@pytest.mark.parametrize(
    "scale_factors",
    [
        (None),  # Standard test case
        ([2, 2, 2, 2]),  # Multi-scale test case
    ],
)
def test_add_image_layer_no_backed(sdata_multi_c, scale_factors):
    name = "raw_image"
    new_name = f"{name}_processed"

    # create an sdata that is not backed
    sdata_no_backed = SpatialData()

    sdata_no_backed = add_image_layer(
        sdata_no_backed,
        arr=sdata_multi_c[name].data,
        output_layer=name,
        scale_factors=scale_factors,
        overwrite=True,
    )

    assert not sdata_no_backed.is_backed()

    # now do a a computation graph, and add result to sdata_no_backed
    se = _get_spatial_element(sdata_no_backed, layer=name)
    arr = se.data
    arr = arr + 1

    sdata_no_backed = add_image_layer(
        sdata_no_backed,
        arr=arr,
        output_layer=new_name,
        scale_factors=scale_factors,
        overwrite=True,
    )

    assert new_name in [*sdata_no_backed.images]

    # check if if contains non zero elements
    assert sdata_no_backed[new_name].any().compute()

    se = _get_spatial_element(sdata_no_backed, layer=new_name)

    for _, layer in se.data.__dask_graph__().layers.items():
        assert layer.is_materialized()


# labels
def test_add_labels_layer_backed(sdata_multi_c):
    name = "masks_whole"
    new_name = f"{name}_processed"

    arr = sdata_multi_c[name].data
    arr = arr + 1
    sdata_multi_c = add_labels_layer(
        sdata_multi_c,
        arr=arr,
        output_layer=new_name,
        overwrite=True,
    )

    # add test to check if mask_whole_processed is non zero

    assert sdata_multi_c.is_backed()
    assert new_name in [*sdata_multi_c.labels]

    # check if it contains non-zero elements.
    assert sdata_multi_c[new_name].any().compute()

    for _, layer in sdata_multi_c[new_name].data.__dask_graph__().layers.items():
        assert layer.is_materialized()


# no backed
def test_add_labels_layer_no_backed(sdata_multi_c):
    name = "masks_whole"
    new_name = f"{name}_processed"

    # create an sdata that is not backed
    sdata_no_backed = SpatialData()

    sdata_no_backed = add_labels_layer(
        sdata_no_backed,
        arr=sdata_multi_c[name].data,
        output_layer=name,
        overwrite=True,
    )

    assert not sdata_no_backed.is_backed()

    # now add a computation graph, and add result to sdata_no_backed
    arr = sdata_no_backed[name].data
    arr = arr + 1

    sdata_no_backed = add_labels_layer(sdata_no_backed, arr=arr, output_layer=new_name)

    assert new_name in [*sdata_no_backed.labels]

    assert sdata_no_backed[new_name].any().compute()

    for _, layer in sdata_no_backed[new_name].data.__dask_graph__().layers.items():
        assert layer.is_materialized()
