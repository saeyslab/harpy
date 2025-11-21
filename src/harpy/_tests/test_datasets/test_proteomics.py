import importlib

import numpy as np
import pytest
from spatialdata import SpatialData, get_pyramid_levels
from spatialdata.transformations import Identity, Scale, get_transformation

from harpy.datasets.proteomics import macsima_colorectal_carcinoma, macsima_example, macsima_tonsil, mibi_example
from harpy.image._image import get_dataarray


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_macsima_tonsil():
    sdata = macsima_tonsil()
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_mibi_example():
    sdata = mibi_example()
    assert isinstance(sdata, SpatialData)
    assert len([*sdata.images]) != 0


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_macsima_example():
    sdata = macsima_example()
    assert isinstance(sdata, SpatialData)
    assert len([*sdata.images]) != 0


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_macsima_colorectal_carcinoma_full():
    chunks = (10, 4096, 4096)
    scale_factors = [2, 2, 2, 2]
    sdata = macsima_colorectal_carcinoma(
        subset=False,  # full macsima run
        c_subset=None,
        remove_dapi=True,
        image_models_kwargs={
            "chunks": chunks,
            "scale_factors": scale_factors,
        },
        output=None,
    )
    assert "REAscreen_IO_CRC_1" in sdata.images
    assert len(get_pyramid_levels(sdata["REAscreen_IO_CRC_1"])) == len(scale_factors) + 1
    se = get_dataarray(sdata, layer="REAscreen_IO_CRC_1")
    assert se.data.chunksize == chunks


@pytest.mark.parametrize("backed", [True, False])
@pytest.mark.parametrize("remove_dapi", [True, False])
@pytest.mark.parametrize("chunks", [(2, 2000, 2000)])
@pytest.mark.skipif(
    not importlib.util.find_spec("bioio") or not importlib.util.find_spec("bioio_ome_tiff"),
    reason="requires the bioio and bioio-ome-tiff libraries",
)
def test_macsima_colorectal_carcinoma(backed, remove_dapi, chunks, tmp_path):
    scale_factors = [2, 2, 2, 2]
    sdata = macsima_colorectal_carcinoma(
        subset=True,
        c_subset=None,
        remove_dapi=remove_dapi,
        image_models_kwargs={
            "chunks": chunks,
            "scale_factors": scale_factors,
        },
        output=tmp_path / "sdata.zarr" if backed else None,
    )

    assert sdata.is_backed() == backed
    assert "REAscreen_IO_CRC_1" in sdata.images
    assert len(get_pyramid_levels(sdata["REAscreen_IO_CRC_1"])) == len(scale_factors) + 1
    se = get_dataarray(sdata, layer="REAscreen_IO_CRC_1")
    assert len(se.c.data) == (4 if remove_dapi else 5)
    assert se.data.chunksize == chunks
    # check transformations
    transformations = get_transformation(sdata["REAscreen_IO_CRC_1"], get_all=True)
    assert "global_1" in transformations.keys()
    assert "global_1_micron" in transformations.keys()
    assert transformations["global_1"] == Identity()
    assert np.array_equal(
        transformations["global_1_micron"].to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x")),
        Scale(axes=("y", "x"), scale=[0.17, 0.17]).to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x")),
    )


@pytest.mark.skipif(
    not importlib.util.find_spec("bioio") or not importlib.util.find_spec("bioio_ome_tiff"),
    reason="requires the bioio and bioio-ome-tiff libraries",
)
def test_macsima_colorectal_carcinoma_c_subset():
    scale_factors = [2, 2, 2, 2]
    chunks = (2, 2000, 2000)
    sdata = macsima_colorectal_carcinoma(
        subset=True,
        c_subset=["DAPI", "CD15"],
        remove_dapi=True,
        image_models_kwargs={
            "chunks": chunks,
            "scale_factors": scale_factors,
        },
        output=None,
    )
    assert "REAscreen_IO_CRC_1" in sdata.images
    assert len(get_pyramid_levels(sdata["REAscreen_IO_CRC_1"])) == len(scale_factors) + 1
    se = get_dataarray(sdata, layer="REAscreen_IO_CRC_1")
    assert np.array_equal(np.array(["0_DAPI_1_DAPI", "4_CD15_1_CD15__VIMC6"], dtype="<U20"), se.c.data)
    assert se.data.chunksize == chunks
    # check transformations
    transformations = get_transformation(sdata["REAscreen_IO_CRC_1"], get_all=True)
    assert "global_1" in transformations.keys()
    assert "global_1_micron" in transformations.keys()
    assert transformations["global_1"] == Identity()
    assert np.array_equal(
        transformations["global_1_micron"].to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x")),
        Scale(axes=("y", "x"), scale=[0.17, 0.17]).to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x")),
    )
