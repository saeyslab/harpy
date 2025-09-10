import os

import anndata as ad
import pandas as pd
import pooch
from spatialdata import SpatialData, read_zarr

from harpy.datasets import get_registry
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _get_df(path):
    df = pd.read_csv(path, index_col=0).reset_index()
    return df


def _get_adata(path, label_prefix="label_whole_"):
    df = _get_df(path)
    df["fov"] = label_prefix + df["fov"]
    obs_columns: list[str] = ["index", "cell_size"] + df.columns[24:].to_list()
    obs = df.loc[:, df.columns.isin(obs_columns)]
    obs = obs.astype(
        {
            "index": "int64",
            "fov": "category",
            "cell_meta_cluster": "category",
            "label": "int64",
        }
    )
    X = df.loc[:, ~df.columns.isin(obs_columns)]
    adata = ad.AnnData(
        X=X,
        obs=obs,
    )
    return adata


def pixie_example_datasets(fovs: list | None = None, with_pixel_output=True, with_cells_output=True) -> SpatialData:
    """Example pixie dataset."""
    log.warning(
        "pixie_example_datasets() is deprecated: it relies on Hugging Face 'datasets' loading scripts, "
        "which were removed in datasets>=4.0.0. Use 'hp.datasets.pixie_example()' or pin 'datasets<4.0.0'."
    )
    import glob
    import os
    from pathlib import Path

    import dask.array as da
    import spatialdata as sd
    from dask_image import imread
    from datasets import load_dataset

    import harpy

    # ['segment_image_data', 'cluster_pixels', 'cluster_cells', 'post_clustering', 'fiber_segmentation', 'LDA_preprocessing', 'LDA_training_inference', 'neighborhood_analysis', 'pairwise_spatial_enrichment', 'ome_tiff', 'ez_seg_data']
    dataset = load_dataset("angelolab/ark_example", "cluster_cells", trust_remote_code=True)

    path_segment_data = Path(dataset["cluster_cells"]["deepcell_output"][0]) / "deepcell_output"

    path_image_data = Path(dataset["cluster_cells"]["image_data"][0]) / "image_data"

    # if with_pixel_output:
    #     path_pixel_data = Path(dataset["cluster_cells"]["example_pixel_output_dir"][0]) / "example_pixel_output_dir"

    if with_cells_output:
        # dataset = load_dataset("angelolab/ark_example", "post_clustering", trust_remote_code=True)
        # path_cell_data = Path(dataset["post_clustering"]["example_cell_output_dir"][0]) / "example_cell_output_dir"
        dataset = load_dataset("angelolab/ark_example", "pairwise_spatial_enrichment", trust_remote_code=True)
        path_post_data = Path(dataset["pairwise_spatial_enrichment"]["post_clustering"][0]) / "post_clustering"

    assert path_segment_data.exists(), path_segment_data

    assert path_image_data.exists(), path_image_data

    # if with_pixel_output:
    #     assert path_pixel_data.exists(), path_pixel_data
    #     print(path_pixel_data)

    if with_cells_output:
        # assert path_cell_data.exists(), path_cell_data
        # print(path_cell_data)
        assert path_post_data.exists(), path_post_data
        print(path_post_data)

    if fovs is None:
        fovs = range(11)
    if isinstance(fovs[0], int):
        fovs = [f"fov{i}" for i in fovs]

    sdata = SpatialData()
    for fov in fovs:
        path = os.path.join(path_image_data, fov)
        paths_images = glob.glob(os.path.join(path, "*.tiff"))
        results = []
        channels = []
        for path_image in paths_images:
            channel = os.path.splitext(os.path.basename(path_image))[0]
            channels.append(channel)
            arr = imread.imread(path_image)
            results.append(arr)
        da.stack(results, axis=0)

        sdata = harpy.im.add_image_layer(
            sdata,
            arr=da.stack(results, axis=0).squeeze(),
            output_layer=f"raw_image_{fov}",
            c_coords=channels,
            overwrite=True,
            transformations={fov: sd.transformations.Identity()},
        )

        sdata = harpy.im.add_labels_layer(
            sdata,
            arr=imread.imread(os.path.join(path_segment_data, f"{fov}_nuclear.tiff")).squeeze(),
            output_layer=f"label_nuclear_{fov}",
            overwrite=True,
            transformations={fov: sd.transformations.Identity()},
        )

        sdata = harpy.im.add_labels_layer(
            sdata,
            arr=imread.imread(os.path.join(path_segment_data, f"{fov}_whole_cell.tiff")).squeeze(),
            output_layer=f"label_whole_{fov}",
            overwrite=True,
            transformations={fov: sd.transformations.Identity()},
        )
    if with_cells_output:
        prefix = "label_whole_"
        adata = _get_adata(path_post_data / "updated_cell_table.csv", label_prefix=prefix)
        adata = adata[adata.obs["fov"].isin([prefix + fov for fov in fovs])]
        stable = sd.models.TableModel.parse(
            adata, region_key="fov", region=[prefix + fov for fov in fovs], instance_key="label"
        )
        sdata.tables["table"] = stable
    return sdata


def pixie_example() -> SpatialData:
    """Example pixie dataset, loaded from s3 bucket"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("proteomics/pixie/sdata_pixie.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


if __name__ == "__main__":
    pixie_example()
