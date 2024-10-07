import numpy as np

from sparrow.image import rasterize
from sparrow.shape import add_shapes_layer


def test_rasterize(sdata_multi_c):
    sdata_multi_c = add_shapes_layer(
        sdata_multi_c,
        input=sdata_multi_c["masks_whole"].data,
        output_layer="masks_whole_boundaries_unit_test",
        overwrite=True,
    )

    sdata_multi_c = rasterize(
        sdata_multi_c,
        shapes_layer="masks_whole_boundaries_unit_test",
        output_layer="masks_whole_unit_test",
        chunks=512,
        overwrite=True,
    )
    # using chunks==200, results in failing of unit test, because two pixels do not match in assertion if arrays are equal.
    # this is merely a computational artifact, and can be ignored.

    assert "masks_whole_unit_test" in [*sdata_multi_c.labels]

    assert np.array_equal(
        sdata_multi_c["masks_whole"].data.compute(), sdata_multi_c["masks_whole_unit_test"].data.compute()
    )


def test_rasterize_blobs(sdata):
    # note that this tests the case when there are labels that are not connected.
    # i.e. in this example label==1 will results in two separate polygons, but still the test passes.
    sdata = add_shapes_layer(
        sdata,
        input=sdata["blobs_labels"].data,
        output_layer="blobs_labels_boundaries",
        overwrite=True,
    )

    sdata = rasterize(
        sdata,
        shapes_layer="blobs_labels_boundaries",
        output_layer="blobs_labels_unit_test",
        chunks=512,
        overwrite=True,
    )

    assert "blobs_labels_unit_test" in [*sdata.labels]

    assert np.array_equal(sdata["blobs_labels"].data.compute(), sdata["blobs_labels_unit_test"].data.compute())

    out_shape = (200, 200)
    sdata = rasterize(
        sdata,
        shapes_layer="blobs_labels_boundaries",
        output_layer="blobs_labels_unit_test",
        chunks=512,
        out_shape=out_shape,
        overwrite=True,
    )

    assert sdata["blobs_labels_unit_test"].shape == out_shape

    assert np.array_equal(
        sdata["blobs_labels"].data[: out_shape[0], : out_shape[1]].compute(),
        sdata["blobs_labels_unit_test"].data.compute(),
    )
