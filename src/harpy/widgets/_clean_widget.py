"""
Napari widget for cleaning spatial transcriptomics microscopy images.

The goal of cleaning is to improve the image quality so that subsequent image segmentation will be more accurate.
"""

import os
from collections.abc import Callable
from typing import Any

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from spatialdata import SpatialData, read_zarr
from xarray import DataTree

from harpy import utils as utils
from harpy.image._image import _get_translation
from harpy.pipeline import HarpyPipeline

log = utils.get_pylogger(__name__)


def cleanImage(
    sdata: SpatialData,
    pipeline: HarpyPipeline,
) -> SpatialData:
    """Function representing the cleaning step, this calls all the needed functions to improve the image quality."""
    sdata = pipeline.clean(sdata)

    return sdata


@thread_worker(progress=True)  # TODO: show string with description of current step in the napari progress bar
def _clean_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: dict[str, Any],
) -> SpatialData:
    """Clean image in a thread worker"""
    res = method(sdata, **fn_kwargs)

    return res


@magic_factory(call_button="Clean")
def clean_widget(
    viewer: napari.Viewer,
    image: napari.layers.Image,
    subset: napari.layers.Shapes | None = None,
    tiling_correction_step: bool = True,
    tile_size: int = 2144,
    min_max_filtering_step: bool = True,
    size_min_max_filter: list[int] = [85],  # noqa: B006 # magicgui does not accept None as default.
    contrast_enhancing_step: bool = True,
    contrast_clip: list[float] = [3.5],  # noqa: B006
):
    """Function represents the clean widget and is called by the wizard to create the widget."""
    contrast_clip = contrast_clip.copy()
    size_min_max_filter = size_min_max_filter.copy()

    if image is None:
        raise ValueError("Please select an image")

    if image.name != utils.LOAD:
        raise ValueError(
            f"Please run the cleaning step on the layer with name '{utils.LOAD}',"
            f"it seems layer with name '{image.name}' was selected."
        )

    pipeline = viewer.layers[utils.LOAD].metadata["pipeline"]

    # need to load it back from zarr store, because otherwise not able to overwrite it
    sdata = read_zarr(pipeline.cfg.paths.sdata)

    pipeline.cfg.clean.tilingCorrection = tiling_correction_step
    pipeline.cfg.clean.tile_size = tile_size

    pipeline.cfg.clean.minmaxFiltering = min_max_filtering_step
    pipeline.cfg.clean.size_min_max_filter = size_min_max_filter

    pipeline.cfg.clean.contrastEnhancing = contrast_enhancing_step
    pipeline.cfg.clean.contrast_clip = contrast_clip
    pipeline.cfg.clean.overwrite = True

    if subset:
        # Check if shapes layer only holds one shape and shape is rectangle
        if len(subset.shape_type) != 1 or subset.shape_type[0] != "rectangle":
            raise ValueError("Please select one rectangular subset")

        coordinates = np.array(subset.data[0])
        crd = [
            int(coordinates[:, 2].min()),
            int(coordinates[:, 2].max()),
            int(coordinates[:, 1].min()),
            int(coordinates[:, 1].max()),
        ]

        pipeline.cfg.clean.crop_param = crd

    else:
        pipeline.cfg.clean.crop_param = None

    pipeline.cfg.clean.small_size_vis = pipeline.cfg.clean.crop_param

    if pipeline.cfg.clean.crop_param is None:
        pipeline.cfg.clean.small_size_vis = [0, 20000, 0, 20000]

    fn_kwargs: dict[str, Any] = {
        "pipeline": pipeline,
    }

    worker = _clean_worker(sdata, method=cleanImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, pipeline: HarpyPipeline, layer_name: str):
        """Add the image to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        offset_x, offset_y = _get_translation(sdata[pipeline.cleaned_image_name])

        if isinstance(sdata[pipeline.cleaned_image_name], DataTree):
            raster = utils._get_raster_multiscale(sdata[pipeline.cleaned_image_name])
        else:
            raster = sdata[pipeline.cleaned_image_name]

        # Translate image to appear on selected region
        viewer.add_image(
            raster,
            rgb=False,
            translate=[
                offset_y,
                offset_x,
            ],
            name=layer_name,
        )

        viewer.layers[layer_name].metadata["pipeline"] = pipeline

        log.info(f"Added {utils.CLEAN} layer")

        utils._export_config(
            pipeline.cfg.clean,
            os.path.join(pipeline.cfg.paths.output_dir, "configs", "clean", "plugin.yaml"),
        )
        log.info("Cleaning finished")
        show_info("Cleaning finished")

    worker.returned.connect(lambda data: add_image(data, pipeline, utils.CLEAN))
    log.info("Cleaning started")
    show_info("Cleaning started")
    worker.start()

    # return worker, so status can be checked for unit testing
    return worker
