import numpy as np
import pytest
from spatialdata import SpatialData

from harpy.image.segmentation._grid import add_grid_labels_layer


@pytest.mark.parametrize("hex_size", [5, 13, 24, 27, 30])
def test_add_grid_labels_layer_hexagon(hex_size):
    sdata = SpatialData()

    output_shapes_layer = "hexagonal_shapes"
    output_labels_layer = "hexagonal_labels"
    shape = (1100, 1000)
    offset = (80, 70)

    sdata = add_grid_labels_layer(
        sdata,
        shape=shape,
        offset=offset,  # not recommended to add offset, better to add offset via a translation
        size=hex_size,
        output_shapes_layer=output_shapes_layer,
        output_labels_layer=output_labels_layer,
        grid_type="hexagon",
    )

    assert output_shapes_layer in [*sdata.shapes]
    assert output_labels_layer in [*sdata.labels]

    assert sdata[output_labels_layer].shape == tuple(a + b for a, b in zip(shape, offset))
    array_labels = sdata[output_labels_layer].data.compute()
    unique_labels = np.unique(array_labels)
    unique_labels = unique_labels[unique_labels != 0]
    assert np.array_equal(unique_labels, np.array(sdata[output_shapes_layer].index))

    # check that we fill the grid completely with hexagons.
    assert np.where(array_labels > 0)[0].min() == offset[0]
    assert np.where(array_labels > 0)[1].min() == offset[1]
    assert (sdata[output_labels_layer].shape[0] - np.where(array_labels > 0)[0].max()) <= 2 * hex_size
    assert (sdata[output_labels_layer].shape[1] - np.where(array_labels > 0)[1].max()) <= np.sqrt(3) * hex_size


@pytest.mark.parametrize("square_size", [5, 13, 24, 27, 30])
def test_add_grid_labels_layer_square(square_size):
    sdata = SpatialData()

    output_shapes_layer = "hexagonal_shapes"
    output_labels_layer = "hexagonal_labels"
    shape = (1100, 1000)
    offset = (80, 70)

    sdata = add_grid_labels_layer(
        sdata,
        shape=shape,
        offset=offset,
        size=square_size,
        output_shapes_layer=output_shapes_layer,
        output_labels_layer=output_labels_layer,
        grid_type="square",
    )

    assert output_shapes_layer in [*sdata.shapes]
    assert output_labels_layer in [*sdata.labels]

    assert sdata[output_labels_layer].shape == tuple(a + b for a, b in zip(shape, offset))
    array_labels = sdata[output_labels_layer].data.compute()
    unique_labels = np.unique(array_labels)
    unique_labels = unique_labels[unique_labels != 0]
    assert np.array_equal(unique_labels, np.array(sdata[output_shapes_layer].index))

    # check that we fill the grid completely with hexagons.
    assert np.where(array_labels > 0)[0].min() == offset[0]
    assert np.where(array_labels > 0)[1].min() == offset[1]
    assert (sdata[output_labels_layer].shape[0] - np.where(array_labels > 0)[0].max()) <= square_size
    assert (sdata[output_labels_layer].shape[1] - np.where(array_labels > 0)[1].max()) <= square_size
    assert (array_labels[offset[0] : -square_size, offset[1] : -square_size] != 0).all()
