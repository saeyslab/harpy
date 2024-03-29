from spatialdata import SpatialData

from sparrow.image._image import _add_image_layer, _add_label_layer


# images
def test_add_image_layer_backed(sdata_multi_c):
    name = "raw_image"
    new_name = f"{name}_processed"

    arr = sdata_multi_c[name].data
    arr = arr + 1
    sdata_multi_c = _add_image_layer(
        sdata_multi_c,
        arr=arr,
        output_layer=new_name,
        overwrite=True,
    )

    # add test to check if raw_image_processed is non zero

    assert sdata_multi_c.is_backed()
    assert new_name in [*sdata_multi_c.images]

    # check if it contains non-zero elements.
    assert sdata_multi_c[new_name].any().compute()

    for name, layer in sdata_multi_c[new_name].data.__dask_graph__().layers.items():
        if name.startswith("from-zarr"):
            assert not layer.is_materialized()
        else:
            assert layer.is_materialized()


# no backed
def test_add_image_layer_no_backed(sdata_multi_c):
    name = "raw_image"
    new_name = f"{name}_processed"

    # create an sdata that is not backed
    sdata_no_backed = SpatialData()

    sdata_no_backed = _add_image_layer(
        sdata_no_backed,
        arr=sdata_multi_c[name].data,
        output_layer=name,
        overwrite=True,
    )

    assert not sdata_no_backed.is_backed()

    # now do a a computation graph, and add result to sdata_no_backed
    arr = sdata_no_backed[name].data
    arr = arr + 1

    sdata_no_backed = _add_image_layer(sdata_no_backed, arr=arr, output_layer=new_name)

    assert new_name in [*sdata_no_backed.images]

    # check if if contains non zero elements
    assert sdata_no_backed[new_name].any().compute()

    for name, layer in sdata_no_backed[new_name].data.__dask_graph__().layers.items():
        if name.startswith("from-zarr"):
            assert not layer.is_materialized()
        else:
            assert layer.is_materialized()


# labels
def test_add_labels_layer_backed(sdata_multi_c):
    name = "masks_whole"
    new_name = f"{name}_processed"

    arr = sdata_multi_c[name].data
    arr = arr + 1
    sdata_multi_c = _add_label_layer(
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

    for name, layer in sdata_multi_c[new_name].data.__dask_graph__().layers.items():
        if name.startswith("from-zarr"):
            assert not layer.is_materialized()
        else:
            assert layer.is_materialized()


# no backed
def test_add_labels_layer_no_backed(sdata_multi_c):
    name = "masks_whole"
    new_name = f"{name}_processed"

    # create an sdata that is not backed
    sdata_no_backed = SpatialData()

    sdata_no_backed = _add_label_layer(
        sdata_no_backed,
        arr=sdata_multi_c[name].data,
        output_layer=name,
        overwrite=True,
    )

    assert not sdata_no_backed.is_backed()

    # now add a computation graph, and add result to sdata_no_backed
    arr = sdata_no_backed[name].data
    arr = arr + 1

    sdata_no_backed = _add_label_layer(sdata_no_backed, arr=arr, output_layer=new_name)

    assert new_name in [*sdata_no_backed.labels]

    assert sdata_no_backed[new_name].any().compute()

    for name, layer in sdata_no_backed[new_name].data.__dask_graph__().layers.items():
        if name.startswith("from-zarr"):
            assert not layer.is_materialized()
        else:
            assert layer.is_materialized()
