import hydra
import pandas as pd
import squidpy as sq
from omegaconf import DictConfig

from napari_spongepy import functions as fc
from napari_spongepy import utils

log = utils.get_pylogger(__name__)


def clean(cfg: DictConfig, results: dict) -> DictConfig:
    # Perform BaSiCCorrection
    img, _, _ = fc.BasiCCorrection(path_image=cfg.dataset.image, device=cfg.device)

    # Preprocess Image
    img, _ = fc.preprocessImage(
        img=img,
        size_tophat=cfg.preprocess.size_tophat,
        contrast_clip=cfg.preprocess.contrast_clip,
    )
    results = {"preprocessimg": img}
    # cfg.result.preprocessimg = img

    return cfg, results


def segment(cfg: DictConfig, results: dict) -> DictConfig:
    import numpy as np
    from squidpy.im import ImageContainer

    from napari_spongepy import utils
    from napari_spongepy.widgets.segmentation_widget import _segmentation_worker

    subset = cfg.subset
    if subset:
        subset = utils.parse_subset(subset)
        log.info(f"Subset is {subset}")

    if cfg.segmentation.get("method"):
        method = cfg.segmentation.method
    else:
        method = hydra.utils.instantiate(cfg.segmentation)

    if cfg.dataset.dtype == "xarray":
        # TODO support preprocessing for zarr datasets
        ic = ImageContainer(cfg.dataset.data_dir)
        print(ic)

        worker = _segmentation_worker(
            ic,
            method=method,
            subset=subset,
            # TODO smarter selection of the z projection method
            reduce_z=3,
            reduce_c=3,
            # small chunks needed if subset is used
        )
    else:
        img = results["preprocessimg"]
        worker = _segmentation_worker(
            img,
            method=method,
            subset=subset,
            # small chunks needed if subset is used
        )

    log.info("Start segmentation")
    [masks, _] = worker.work()
    log.info(masks.shape)

    if cfg.paths.masks:
        log.info(f"Writing masks to {cfg.paths.masks}")
        np.save(cfg.paths.masks, masks)
    results["segmentationmasks"] = masks

    return cfg, results


def allocate(cfg: DictConfig, results: dict) -> DictConfig:
    masks = results["segmentationmasks"]
    img = results["preprocessimg"]
    adata = fc.create_adata_quick(cfg.dataset.coords, img, masks)
    adata, _ = fc.preprocessAdata(adata, masks)
    adata, _ = fc.filter_on_size(adata, min_size=500)
    fc.clustering(adata, 17, 35)

    results["adata"] = adata

    return cfg, results


def annotate(cfg: DictConfig, results: dict) -> DictConfig:
    adata = results["adata"]
    _, _ = fc.scoreGenesLiver(adata, cfg.dataset.markers)
    results["adata"] = adata

    return cfg, results


def visualize(cfg: DictConfig, results: dict) -> DictConfig:
    adata = results["adata"]

    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="maxScores")
    sq.pl.nhood_enrichment(adata, cluster_key="maxScores", method="ward")

    del adata.obsm["polygons"]["color"]
    adata.obsm["polygons"]["geometry"].to_file(cfg.paths.geojson, driver="GeoJSON")

    adata.obsm["polygons"] = pd.DataFrame(
        {
            "linewidth": adata.obsm["polygons"]["linewidth"],
            "X": adata.obsm["polygons"]["X"],
            "Y": adata.obsm["polygons"]["Y"],
        }
    )
    adata.write(cfg.paths.h5ad)

    return cfg