import os

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import scanpy as sc

from harpy.image._image import add_image_layer, add_labels_layer
from harpy.plot._plot import plot_image, plot_labels, plot_shapes
from harpy.plot._sanity import sanity
from harpy.shape._shape import add_shapes_layer


def test_plot_labels(sdata_multi_c_no_backed, tmp_path):
    plot_labels(
        sdata_multi_c_no_backed,
        labels_layer="masks_nuclear",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    plot_labels(
        sdata_multi_c_no_backed,
        labels_layer=["masks_nuclear_aligned", "masks_whole"],
        output=os.path.join(tmp_path, "labels_all"),
        crd=[100, 200, 100, 200],
    )


def test_plot_image(sdata_multi_c_no_backed, tmp_path):
    plot_image(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        channel=[0, 1],
        output=os.path.join(tmp_path, "raw_image"),
    )

    # dpi + colorbar
    plot_image(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        channel=[0, 1],
        fig_kwargs={"dpi": 10},
        colorbar=True,
        output=os.path.join(tmp_path, "raw_image_dpi"),
    )


def test_plot_shapes(sdata_multi_c_no_backed, tmp_path):
    # plot a .obs column
    plot_shapes(
        sdata_multi_c_no_backed,
        img_layer="combine",
        shapes_layer="masks_whole_boundaries",
        column="area",
        table_layer="table_intensities",
        region="masks_whole",
        output=os.path.join(tmp_path, "shapes_masks_whole_area"),
    )

    # plot a .var column
    plot_shapes(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        shapes_layer="masks_whole_boundaries",
        channel=1,
        column="1",
        table_layer="table_intensities",
        region="masks_whole",
        output=os.path.join(tmp_path, "shapes_masks_whole_channel_1"),
    )

    # plot a .obs column with dpi + legend False
    plot_shapes(
        sdata_multi_c_no_backed,
        img_layer="combine",
        shapes_layer="masks_whole_boundaries",
        column="area",
        table_layer="table_intensities",
        region="masks_whole",
        fig_kwargs={"dpi": 10},
        legend=False,
        output=os.path.join(tmp_path, "shapes_masks_whole_area_dpi"),
    )

    with pytest.raises(
        ValueError,
        match=r"'sdata.tables\[table_intensities\]' contains more than one region in 'sdata.tables\[table_intensities\].obs\[ fov_labels \]', please specify 'region'. Choose from the list '\['masks_nuclear_aligned', 'masks_whole'\]",
    ):
        plot_shapes(
            sdata_multi_c_no_backed,
            img_layer="combine",
            shapes_layer="masks_whole_boundaries",
            column="area",
            table_layer="table_intensities",
            region=None,
            output=os.path.join(tmp_path, "shapes_masks_whole"),
        )


def test_plot_shapes_transcriptomics(sdata_transcripts_no_backed, tmp_path):
    plot_shapes(
        sdata_transcripts_no_backed,
        img_layer="raw_image",
        shapes_layer="segmentation_mask_boundaries",
        table_layer="table_transcriptomics",
        region="segmentation_mask",
        output=os.path.join(tmp_path, "shapes_segmentation_mask"),
    )

    # plot a .obs column
    plot_shapes(
        sdata_transcripts_no_backed,
        img_layer="raw_image",
        shapes_layer="segmentation_mask_boundaries",
        column="cell_ID",
        table_layer="table_transcriptomics",
        region="segmentation_mask",
        output=os.path.join(tmp_path, "shapes_segmentation_mask_cell_ID"),
    )

    # plot a .var column
    plot_shapes(
        sdata_transcripts_no_backed,
        img_layer="raw_image",
        shapes_layer="segmentation_mask_boundaries",
        column="Pck1",
        table_layer="table_transcriptomics",
        region="segmentation_mask",
        output=os.path.join(tmp_path, "shapes_segmentation_mask_Pck1"),
    )

    # plot circles
    gdf = sdata_transcripts_no_backed["segmentation_mask_boundaries"]
    circles = gpd.GeoDataFrame({"geometry": gdf.geometry.centroid})
    circles["radius"] = 100
    sdata_transcripts_no_backed = add_shapes_layer(
        sdata_transcripts_no_backed,
        input=circles,
        output_layer="circles",
        overwrite=True,
    )

    plot_shapes(
        sdata_transcripts_no_backed,
        img_layer="raw_image",
        shapes_layer="circles",
        crd=[250, 1000, 1000, 2000],
        radius="radius",
        output=os.path.join(tmp_path, "circles"),
    )


def test_plot_shapes_umap_categories(sdata_transcripts_no_backed, tmp_path):
    table_layer = "table_transcriptomics_cluster"

    np.random.seed(42)
    sdata_transcripts_no_backed[table_layer].obs["new_category"] = np.random.randint(
        0, 15, size=len(sdata_transcripts_no_backed[table_layer].obs)
    )
    sdata_transcripts_no_backed[table_layer].obs["new_category"] = (
        sdata_transcripts_no_backed[table_layer].obs["new_category"].astype(int).astype("category")
    )
    sc.pl.umap(sdata_transcripts_no_backed.tables[table_layer], color=["new_category"], show=False)

    plot_shapes(
        sdata_transcripts_no_backed,
        shapes_layer="segmentation_mask_boundaries",
        img_layer="raw_image",
        alpha=1,
        table_layer=table_layer,
        column="new_category",
        linewidth=0,
        output=os.path.join(tmp_path, "shapes_segmentation_mask_categorical"),
    )


def test_plot_shapes_3D(sdata_transcripts_no_backed, tmp_path):
    arr_image = da.stack(
        [sdata_transcripts_no_backed["raw_image"].data, sdata_transcripts_no_backed["raw_image"].data], axis=1
    )

    sdata_transcripts_no_backed = add_image_layer(
        sdata_transcripts_no_backed,
        arr=arr_image,
        output_layer="raw_image_z",
        overwrite=True,
    )

    arr_labels = da.stack(
        [sdata_transcripts_no_backed["segmentation_mask"].data, sdata_transcripts_no_backed["segmentation_mask"].data],
        axis=0,
    )

    sdata_transcripts_no_backed = add_labels_layer(
        sdata_transcripts_no_backed,
        arr=arr_labels,
        output_layer="segmentation_mask_z",
        overwrite=True,
    )

    sdata_transcripts_no_backed = add_shapes_layer(
        sdata_transcripts_no_backed,
        input=sdata_transcripts_no_backed.labels["segmentation_mask_z"].data,
        output_layer="segmentation_mask_boundaries_z",
        overwrite=True,
    )

    plot_shapes(
        sdata_transcripts_no_backed,
        img_layer="raw_image_z",
        shapes_layer="segmentation_mask_boundaries_z",
        z_slice=0.5,
        crd=[400, 1500, 1500, 2500],
        output=os.path.join(tmp_path, "shapes_segmentation_mask_z_slice"),
    )


def test_sanity(sdata_transcripts_no_backed, tmp_path):
    sanity(
        sdata_transcripts_no_backed,
        img_layer="raw_image",
        points_layer="transcripts",
        shapes_layer="segmentation_mask_boundaries",
        crd=[500, 1500, 1500, 2500],
        plot_cell_number=True,
        n_sample=10000,
        output=os.path.join(tmp_path, "sanity_plot"),
    )
