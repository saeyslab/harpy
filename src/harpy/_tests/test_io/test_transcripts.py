import numpy as np
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Affine, Identity, Scale, get_transformation

from harpy.io._transcripts import read_transcripts


@pytest.mark.parametrize("backed", [False, True])
def test_read_transcripts(path_transcripts, backed, tmp_path):
    sdata = SpatialData()
    if backed:
        sdata.write(tmp_path / "sdata.zarr")
        sdata = read_zarr(sdata.path)

    pixel_size = 0.138

    to_coordinate_system = "a1_1_pixel"
    to_micron_coordinate_system = "a1_1_micron"
    output_layer = "transcripts_a1_1"

    sdata = read_transcripts(
        sdata,
        path_count_matrix=path_transcripts,
        column_x=0,
        column_y=1,
        column_gene=3,
        delimiter="\t",
        header=None,
        output_layer=output_layer,
        to_coordinate_system=to_coordinate_system,
        to_micron_coordinate_system=to_micron_coordinate_system,
        pixel_size=pixel_size,  # size of pixels in micron
        transform_matrix=None,  # transcripts are already in pixels, and registered with the images
        overwrite=True,
    )

    assert output_layer in sdata.points

    assert Identity() == get_transformation(sdata.points[output_layer], to_coordinate_system=to_coordinate_system)
    # check correct transformation is defined on the spatial element
    affine_matrix = Scale(axes=("x", "y"), scale=[pixel_size, pixel_size]).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    assert np.array_equal(
        affine_matrix,
        get_transformation(
            sdata.points[output_layer], to_coordinate_system=to_micron_coordinate_system
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )


@pytest.mark.parametrize("backed", [False, True])
def test_read_transcripts_transform_matrix(path_transcripts, backed, tmp_path):
    sdata = SpatialData()
    if backed:
        sdata.write(tmp_path / "sdata.zarr")
        sdata = read_zarr(sdata.path)

    pixel_size = 0.138

    to_coordinate_system = "a1_1_pixel"
    to_micron_coordinate_system = "a1_1_micron"
    output_layer = "transcripts_a1_1"

    transform_matrix = np.array([[1 / pixel_size, 0, 0], [0, 1 / pixel_size, 0], [0, 0, 1]])
    # Note: transform_matrix is a dummy matrix, the transcripts are already in pixels

    sdata = read_transcripts(
        sdata,
        path_count_matrix=path_transcripts,
        column_x=0,
        column_y=1,
        column_gene=3,
        delimiter="\t",
        header=None,
        output_layer=output_layer,
        to_coordinate_system=to_coordinate_system,
        to_micron_coordinate_system=to_micron_coordinate_system,
        pixel_size=None,
        transform_matrix=transform_matrix,  # do the dummy transform from 'micron' to 'pixels'
        overwrite=True,
    )

    assert output_layer in sdata.points
    assert Identity() == get_transformation(sdata.points[output_layer], to_coordinate_system=to_coordinate_system)
    affine_matrix = (
        Affine(matrix=transform_matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        .inverse()
        .to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    )
    assert np.array_equal(
        affine_matrix,
        get_transformation(
            sdata.points[output_layer], to_coordinate_system=to_micron_coordinate_system
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )


@pytest.mark.parametrize("backed", [False, True])
def test_read_transcripts_raises(path_transcripts, backed, tmp_path):
    sdata = SpatialData()
    if backed:
        sdata.write(tmp_path / "sdata.zarr")
        sdata = read_zarr(sdata.path)

    pixel_size = 0.138

    to_coordinate_system = "a1_1_pixel"
    to_micron_coordinate_system = "a1_1_micron"
    output_layer = "transcripts_a1_1"

    transform_matrix = np.array([[1 / pixel_size, 0, 0], [0, 1 / pixel_size, 0], [0, 0, 1]])
    # Note: transform_matrix is a dummy matrix, the transcripts are already in pixels

    with pytest.raises(
        ValueError,
        match="The transform matrix from micron to pixels is not equal to the identity matrix, which implies transcripts are in micron coordinates",
    ):
        sdata = read_transcripts(
            sdata,
            path_count_matrix=path_transcripts,
            column_x=0,
            column_y=1,
            column_gene=3,
            delimiter="\t",
            header=None,
            output_layer=output_layer,
            to_coordinate_system=to_coordinate_system,
            to_micron_coordinate_system=to_micron_coordinate_system,
            pixel_size=pixel_size,  # both pixel size and transform matrix specified
            transform_matrix=transform_matrix,
            overwrite=True,
        )
