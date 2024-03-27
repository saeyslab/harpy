import os

import pytest

from sparrow.plot._plot import plot_image, plot_labels, plot_shapes
from sparrow.shape._shape import _add_shapes_layer


def test_plot_labels(sdata_multi_c, tmp_path):
    plot_labels(
        sdata_multi_c,
        labels_layer="masks_nuclear",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    plot_labels(
        sdata_multi_c,
        labels_layer=["masks_nuclear_aligned", "masks_whole"],
        output=os.path.join(tmp_path, "labels_all"),
        crd=[100, 200, 100, 200],
    )


def test_plot_image(sdata_multi_c, tmp_path):
    plot_image(
        sdata_multi_c,
        img_layer="raw_image",
        channel=[0, 1],
        output=os.path.join(tmp_path, "raw_image"),
    )


def test_plot_shapes(sdata_multi_c, tmp_path):
    sdata_multi_c = _add_shapes_layer(
        sdata_multi_c,
        input=sdata_multi_c.labels["masks_whole"].data,
        output_layer="masks_whole_boundaries",
        overwrite=True,
    )

    # plot an obs column
    plot_shapes(
        sdata_multi_c,
        img_layer="combine",
        shapes_layer="masks_whole_boundaries",
        column="area",
        table_layer="table_intensities",
        region="masks_whole",
        output=os.path.join(tmp_path, "shapes_masks_whole_area"),
    )

    # plot a .var column
    plot_shapes(
        sdata_multi_c,
        img_layer="raw_image",
        shapes_layer="masks_whole_boundaries",
        channel=1,
        column="1",
        table_layer="table_intensities",
        region="masks_whole",
        output=os.path.join(tmp_path, "shapes_masks_whole_channel_1"),
    )

    with pytest.raises(
        ValueError,
        match=r"'sdata.tables\[table_intensities\]' contains more than one region in 'sdata.tables\[table_intensities\].obs\[ fov_labels \]', please specify 'region'. Choose from the list '\['masks_nuclear_aligned', 'masks_whole'\]",
    ):
        plot_shapes(
            sdata_multi_c,
            img_layer="combine",
            shapes_layer="masks_whole_boundaries",
            column="area",
            table_layer="table_intensities",
            region=None,
            output=os.path.join(tmp_path, "shapes_masks_whole"),
        )
