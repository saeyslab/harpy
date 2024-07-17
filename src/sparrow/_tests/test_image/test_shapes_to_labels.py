import numpy as np

from sparrow.image import add_labels_layer_from_shapes_layer
from sparrow.shape import add_shapes_layer


def test_add_labels_layer_from_shapes_layer(sdata_multi_c):
    sdata_multi_c = add_shapes_layer(
        sdata_multi_c,
        input=sdata_multi_c["masks_whole"].data,
        output_layer="masks_whole_boundaries_unit_test",
        overwrite=True,
    )

    sdata_multi_c = add_labels_layer_from_shapes_layer(
        sdata_multi_c,
        shapes_layer="masks_whole_boundaries_unit_test",
        output_layer="masks_whole_unit_test",
        overwrite=True,
    )

    assert "masks_whole_unit_test" in [*sdata_multi_c.labels]

    assert np.array_equal(
        sdata_multi_c["masks_whole"].data.compute(), sdata_multi_c["masks_whole_unit_test"].data.compute()
    )
