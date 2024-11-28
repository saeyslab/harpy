import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import napari
import napari.layers
import napari.types
import napari.utils
import numpy as np
from hydra import compose, initialize_config_dir
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from pkg_resources import resource_filename
from spatialdata import SpatialData
from xarray import DataTree

from harpy import utils
from harpy.image._image import _get_translation
from harpy.pipeline import HarpyPipeline

log = utils.get_pylogger(__name__)


def loadImage(
    pipeline: HarpyPipeline,
) -> SpatialData:
    """Function representing the loading step."""
    sdata = pipeline.load()

    return sdata


@thread_worker(progress=True)  # TODO: show string with description of current step in the napari progress bar
def _load_worker(
    method: Callable,
    fn_kwargs: dict[str, Any],
) -> list[np.ndarray]:
    """Load image in a thread worker"""
    res = method(**fn_kwargs)

    return res


@magic_factory(
    call_button="Load",
    path_zarr={"widget_type": "FileEdit", "mode": "d"},
    path_image={"widget_type": "FileEdit"},
    output_dir={"widget_type": "FileEdit", "mode": "d"},
)
def load_widget(
    viewer: napari.Viewer,
    path_zarr: Path = Path(""),
    path_image: Path = Path(""),
    image_layer: Optional[str] = utils.LOAD,
    output_dir: Path = Path(""),
    x_min: Optional[str] = "",
    x_max: Optional[str] = "",
    y_min: Optional[str] = "",
    y_max: Optional[str] = "",
):
    """Function represents the load widget and is called by the wizard to create the widget."""
    # get the default values for the configs
    abs_config_dir = resource_filename("harpy", "configs")

    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name="pipeline")

    cfg.paths.output_dir = str(output_dir)
    # set default scale factors to this value.
    cfg.dataset.scale_factors = [2, 2, 2, 2]

    if str(path_zarr).endswith(".zarr"):
        cfg.paths.sdata = str(path_zarr)
        cfg.dataset.image = cfg.paths.sdata
    elif path_image:
        cfg.paths.sdata = os.path.join(cfg.paths.output_dir, "sdata.zarr")
        cfg.dataset.image = path_image
    else:
        raise ValueError("Please provide either a path to a zarr store 'path_zarr', or an image 'path_image'")

    crd = [x_min, x_max, y_min, y_max]
    crd = [None if val == "" else int(val) for val in crd]

    cfg.dataset.crop_param = crd

    pipeline = HarpyPipeline(cfg, image_name=image_layer)

    fn_kwargs: dict[str, Any] = {"pipeline": pipeline}

    worker = _load_worker(method=loadImage, fn_kwargs=fn_kwargs)

    def add_image(sdata: SpatialData, pipeline: HarpyPipeline, layer_name: str):
        """Add the image to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing { layer_name }")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding { layer_name }")

        offset_x, offset_y = _get_translation(sdata[pipeline.loaded_image_name])

        if isinstance(sdata[pipeline.loaded_image_name], DataTree):
            raster = utils._get_raster_multiscale(sdata[pipeline.loaded_image_name])
        else:
            raster = sdata[pipeline.loaded_image_name]

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
        viewer.layers[layer_name].metadata["sdata"] = sdata

        log.info(f"Added '{ layer_name }' layer")

        show_info("Loading finished")

    worker.returned.connect(lambda data: add_image(data, pipeline, utils.LOAD))
    show_info("Loading started")
    worker.start()

    # return worker, so status can be checked for unit testing
    return worker
