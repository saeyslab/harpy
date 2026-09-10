import importlib.util
import os

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe import DataFrame
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

from harpy.image._image import _get_spatial_element
from harpy.image.segmentation._segmentation import segment, segment_points
from harpy.image.segmentation.segmentation_models._baysor import _dummy
from harpy.image.segmentation.segmentation_models._cellpose import cellpose_callable
from harpy.points._points import add_points


def _channel_selection_model(
    img: np.ndarray,
    expected_channel_values: tuple[int, ...],
    channels: list[int],
) -> np.ndarray:
    assert channels == [1, 0]
    assert img.shape[-1] == len(expected_channel_values)
    for channel, expected_value in enumerate(expected_channel_values):
        assert np.all(img[..., channel] == expected_value)
    return np.zeros((*img.shape[:-1], 1), dtype=np.uint32)


def _channel_selection_sdata(scale_factors=None) -> SpatialData:
    image = da.stack([da.full((8, 8), value, chunks=(8, 8), dtype=np.uint8) for value in (1, 2, 3)])
    return SpatialData(
        images={
            "image": Image2DModel.parse(
                image,
                dims=("c", "y", "x"),
                c_coords=["first", "second", "third"],
                scale_factors=scale_factors,
            )
        }
    )


@pytest.mark.parametrize(
    ("scale_factors", "image_channels", "expected_channel_values"),
    [
        (None, None, (1, 2, 3)),
        ([2], "second", (2,)),
        ([2], ["third", "first"], (3, 1)),
    ],
)
def test_segment_selects_image_channels_and_preserves_model_channels(
    scale_factors,
    image_channels,
    expected_channel_values,
):
    sdata = _channel_selection_sdata(scale_factors=scale_factors)

    sdata = segment(
        sdata,
        image_name="image",
        image_channels=image_channels,
        model=_channel_selection_model,
        output_labels_name="labels",
        output_shapes_name=None,
        trim=True,
        depth=1,
        channels=[1, 0],
        expected_channel_values=expected_channel_values,
    )

    assert "labels" in sdata.labels


@pytest.mark.parametrize("image_channels", [[], ["missing"]])
def test_segment_rejects_invalid_image_channels(image_channels):
    sdata = _channel_selection_sdata()

    with pytest.raises(ValueError, match="image_channels|Image channels"):
        segment(
            sdata,
            image_name="image",
            image_channels=image_channels,
            model=_channel_selection_model,
            output_labels_name="labels",
            output_shapes_name=None,
            channels=[1, 0],
            expected_channel_values=(1,),
        )


@pytest.mark.skipif(not importlib.util.find_spec("cellpose"), reason="requires the cellpose library")
def test_segment(sdata_multi_c_no_backed: SpatialData):
    import torch

    with dask.config.set(scheduler="processes"):
        sdata_multi_c_no_backed = segment(
            sdata_multi_c_no_backed,
            image_name="combine",
            model=cellpose_callable,
            output_labels_name="masks_cellpose",
            output_shapes_name="masks_cellpose_boundaries",
            trim=False,
            chunks=50,
            overwrite=True,
            depth=30,
            crd=[10, 110, 0, 100],
            scale_factors=[2, 2, 2, 2],
            diameter=50,
            cellprob_threshold=-4,
            flow_threshold=0.9,
            pretrained_model="nuclei",
            do_3D=False,
            channels=[1, 0],
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )

        assert "masks_cellpose" in sdata_multi_c_no_backed.labels
        assert "masks_cellpose_boundaries" in sdata_multi_c_no_backed.shapes
        assert isinstance(sdata_multi_c_no_backed, SpatialData)


@pytest.mark.skipif(not importlib.util.find_spec("cellpose"), reason="requires the cellpose library")
def test_segment_pseudo_3D(sdata_multi_c_no_backed: SpatialData):
    import torch

    with dask.config.set(scheduler="processes"):
        sdata_multi_c_no_backed = segment(
            sdata_multi_c_no_backed,
            image_name="combine_z",
            model=cellpose_callable,
            output_labels_name="masks_cellpose_3D",
            output_shapes_name="masks_cellpose_3D_boundaries",
            trim=False,
            chunks=(50, 50),
            overwrite=True,
            depth=(20, 20),
            crd=[50, 80, 10, 70],
            scale_factors=[2],
            diameter=None,  # specifying diameter is bugged in cellpose>=4.0.6 when running pseudo 3D segmentation.
            cellprob_threshold=-4,
            flow_threshold=0.9,
            pretrained_model="nuclei",
            channels=[1, 0],
            do_3D=False,  # pseudo 3D
            stitch_threshold=0.5,  # pseudo 3D, we stitch in 3D
            anisotropy=1,
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )

        assert "masks_cellpose_3D" in sdata_multi_c_no_backed.labels
        assert isinstance(sdata_multi_c_no_backed, SpatialData)


@pytest.mark.skip(reason="Skipping: 3D segmentation test is time-consuming.")
@pytest.mark.skipif(not importlib.util.find_spec("cellpose"), reason="requires the cellpose library")
def test_segment_3D(sdata_multi_c_no_backed: SpatialData):
    import torch

    with dask.config.set(scheduler="processes"):
        sdata_multi_c_no_backed = segment(
            sdata_multi_c_no_backed,
            image_name="combine_z",
            model=cellpose_callable,
            output_labels_name="masks_cellpose_3D",
            output_shapes_name="masks_cellpose_3D_boundaries",
            trim=False,
            chunks=(50, 50),
            overwrite=True,
            depth=(20, 20),
            crd=[50, 80, 10, 70],
            scale_factors=[2],
            diameter=20,
            cellprob_threshold=-4,
            flow_threshold=0.9,
            pretrained_model="nuclei",
            channels=[1, 0],
            do_3D=True,  #  full 3D
            stitch_threshold=0.0,  # we segment in full 3D
            anisotropy=1,
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )

        assert "masks_cellpose_3D" in sdata_multi_c_no_backed.labels
        assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_segment_points(sdata_multi_c_no_backed: SpatialData):
    data = {"x": [10], "y": [10], "gene": ["dummy_gene"]}

    # Create the DataFrame
    df = pd.DataFrame(data)

    ddf = dd.from_pandas(df, npartitions=1)

    coordinates = {"x": "x", "y": "y"}

    sdata_multi_c_no_backed = add_points(
        sdata_multi_c_no_backed,
        ddf=ddf,
        output_points_name="transcripts",
        coordinates=coordinates,
        overwrite=False,
    )

    assert isinstance((sdata_multi_c_no_backed.points["transcripts"]), DataFrame)

    sdata_multi_c_no_backed = segment_points(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        points_name="transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        output_labels_name="masks_whole_copy_dummy",
        output_shapes_name="masks_whole_copy_dummy_boundaries",
        chunks=256,
        crd=None,
    )

    output_labels_name = ["masks_whole_copy_dummy_1", "masks_whole_copy_dummy_2"]
    output_shapes_name = ["masks_whole_copy_dummy_boundaries_1", "masks_whole_copy_dummy_boundaries_2"]
    # test multi channel support for output labels dimension.
    sdata_multi_c_no_backed = segment_points(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        points_name="transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        c_dim=2,
        output_labels_name=output_labels_name,
        output_shapes_name=output_shapes_name,
        labels_name_align=output_labels_name[0],
        chunks=256,
        iou_depth=[3, 4],  # this will be used when aligning labels
        crd=None,
    )

    for _output_labels_name in output_labels_name:
        assert _output_labels_name in sdata_multi_c_no_backed.labels
    for _output_shapes_name in output_shapes_name:
        assert _output_shapes_name in sdata_multi_c_no_backed.shapes


@pytest.mark.skipif(not importlib.util.find_spec("instanseg"), reason="requires the instanseg library")
def test_segment_instanseg(sdata_multi_c_no_backed: SpatialData):
    import torch
    from instanseg import InstanSeg

    from harpy.image.segmentation.segmentation_models._instanseg import instanseg_callable

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    _ = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1, device=device)

    path_model = os.path.join(
        os.environ.get("INSTANSEG_BIOIMAGEIO_PATH"), "fluorescence_nuclei_and_cells/0.1.1/instanseg.pt"
    )

    output_labels_name = ["labels_nuclei_instanseg", "labels_cells_instanseg"]
    output_shapes_name = ["shapes_nuclei_instanseg", "shapes_cells_instanseg"]
    with dask.config.set(scheduler="processes"):
        sdata_multi_c_no_backed = segment(
            sdata_multi_c_no_backed,
            image_name="combine",
            model=instanseg_callable,
            output_labels_name=output_labels_name,
            output_shapes_name=output_shapes_name,
            labels_name_align="labels_cells_instanseg",
            trim=False,
            chunks=50,
            overwrite=True,
            depth=30,
            crd=[10, 110, 0, 100],
            scale_factors=[2, 2, 2, 2],
            device=device,
            instanseg_model=path_model,
            output="all_outputs",
        )

    for _output_labels_name in output_labels_name:
        assert _output_labels_name in sdata_multi_c_no_backed.labels
    for _output_shapes_name in output_shapes_name:
        assert _output_shapes_name in sdata_multi_c_no_backed.shapes

    for _output_labels_name in output_labels_name:
        se = _get_spatial_element(sdata_multi_c_no_backed, element_name=output_labels_name[0])
        assert da.any(se.data).compute()
