import importlib.util

import dask.array as da
import dask.dataframe as dd
import pandas as pd
import pytest
from dask.dataframe import DataFrame
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.image.segmentation._segmentation import segment, segment_points
from harpy.image.segmentation.segmentation_models._baysor import _dummy
from harpy.image.segmentation.segmentation_models._cellpose import cellpose_callable
from harpy.points._points import add_points_layer


@pytest.mark.skipif(not importlib.util.find_spec("cellpose"), reason="requires the cellpose library")
def test_segment(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = segment(
        sdata_multi_c_no_backed,
        img_layer="combine",
        model=cellpose_callable,
        output_labels_layer="masks_cellpose",
        output_shapes_layer="masks_cellpose_boundaries",
        trim=False,
        chunks=50,
        overwrite=True,
        depth=30,
        crd=[10, 110, 0, 100],
        scale_factors=[2, 2, 2, 2],
        diameter=20,
        cellprob_threshold=-4,
        flow_threshold=0.9,
        model_type="nuclei",
        do_3D=False,
        channels=[1, 0],
    )

    assert "masks_cellpose" in sdata_multi_c_no_backed.labels
    assert "masks_cellpose_boundaries" in sdata_multi_c_no_backed.shapes
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


@pytest.mark.skipif(not importlib.util.find_spec("cellpose"), reason="requires the cellpose library")
def test_segment_3D(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = segment(
        sdata_multi_c_no_backed,
        img_layer="combine_z",
        model=cellpose_callable,
        output_labels_layer="masks_cellpose_3D",
        output_shapes_layer="masks_cellpose_3D_boundaries",
        trim=False,
        chunks=(50, 50),
        overwrite=True,
        depth=(20, 20),
        crd=[50, 80, 10, 70],
        scale_factors=[2],
        diameter=20,
        cellprob_threshold=-4,
        flow_threshold=0.9,
        model_type="nuclei",
        channels=[1, 0],
        do_3D=True,
        anisotropy=1,
    )

    assert "masks_cellpose_3D" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_segment_points(sdata_multi_c_no_backed: SpatialData):
    data = {"x": [10], "y": [10], "gene": ["dummy_gene"]}

    # Create the DataFrame
    df = pd.DataFrame(data)

    ddf = dd.from_pandas(df, npartitions=1)

    coordinates = {"x": "x", "y": "y"}

    sdata_multi_c_no_backed = add_points_layer(
        sdata_multi_c_no_backed,
        ddf=ddf,
        output_layer="transcripts",
        coordinates=coordinates,
        overwrite=False,
    )

    assert isinstance((sdata_multi_c_no_backed.points["transcripts"]), DataFrame)

    sdata_multi_c_no_backed = segment_points(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        points_layer="transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        output_labels_layer="masks_whole_copy_dummy",
        output_shapes_layer="masks_whole_copy_dummy_boundaries",
        chunks=256,
        crd=None,
    )

    output_labels_layer = ["masks_whole_copy_dummy_1", "masks_whole_copy_dummy_2"]
    output_shapes_layer = ["masks_whole_copy_dummy_boundaries_1", "masks_whole_copy_dummy_boundaries_2"]
    # test multi channel support for output labels dimension.
    sdata_multi_c_no_backed = segment_points(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        points_layer="transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        c_dim=2,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        labels_layer_align=output_labels_layer[0],
        chunks=256,
        iou_depth=[3, 4],  # this will be used when aligning labels
        crd=None,
    )

    for _output_labels_layer in output_labels_layer:
        assert _output_labels_layer in sdata_multi_c_no_backed.labels
    for _output_shapes_layer in output_shapes_layer:
        assert _output_shapes_layer in sdata_multi_c_no_backed.shapes


@pytest.mark.skipif(not importlib.util.find_spec("instanseg"), reason="requires the instanseg library")
def test_segment_instanseg(sdata_multi_c_no_backed: SpatialData):
    from instanseg import InstanSeg

    from harpy.image.segmentation.segmentation_models._instanseg import instanseg_callable

    instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1, device="cpu")

    output_labels_layer = ["labels_nuclei_instanseg", "labels_cells_instanseg"]
    output_shapes_layer = ["shapes_nuclei_instanseg", "shapes_cells_instanseg"]
    sdata_multi_c_no_backed = segment(
        sdata_multi_c_no_backed,
        img_layer="combine",
        model=instanseg_callable,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        labels_layer_align="labels_cells_instanseg",
        trim=False,
        chunks=50,
        overwrite=True,
        depth=30,
        crd=[10, 110, 0, 100],
        scale_factors=[2, 2, 2, 2],
        instanseg_model=instanseg_fluorescence,
        output="all_outputs",
    )

    for _output_labels_layer in output_labels_layer:
        assert _output_labels_layer in sdata_multi_c_no_backed.labels
    for _output_shapes_layer in output_shapes_layer:
        assert _output_shapes_layer in sdata_multi_c_no_backed.shapes

    for _output_labels_layer in output_labels_layer:
        se = _get_spatial_element(sdata_multi_c_no_backed, layer=output_labels_layer[0])
        assert da.any(se.data).compute()
