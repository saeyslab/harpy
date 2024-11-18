"""Napari widget for cell segmentation of cleaned spatial transcriptomics microscopy images."""

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import napari
import napari.layers
import napari.types
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from spatialdata import SpatialData, read_zarr

import sparrow.utils as utils
from sparrow.pipeline import SparrowPipeline
from sparrow.utils.utils import _translate_polygons

log = utils.get_pylogger(__name__)


class ModelOption(Enum):
    nuclei = "nuclei"
    cyto = "cyto"


def segmentImage(
    sdata: SpatialData,
    pipeline: SparrowPipeline,
) -> SpatialData:
    """Function representing the segmentation step, this calls the segmentation function."""
    sdata = pipeline.segment(sdata)

    return sdata


@thread_worker(progress=True)
def _segmentation_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> SpatialData:
    """Segment image in a thread worker"""
    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Segment",
    cellprob_threshold={"widget_type": "SpinBox", "min": -50, "max": 100},
    channels={"layout": "vertical", "options": {"min": 0, "max": 3}},
)
def segment_widget(
    viewer: napari.Viewer,
    image: Optional[napari.layers.Image] = None,
    subset: Optional[napari.layers.Shapes] = None,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.8,
    diameter: int = 50,
    cellprob_threshold: int = -2,
    model_type: ModelOption = ModelOption.nuclei,
    channels: List[int] = [1, 0],  # noqa: B006 # magicgui does not accept None
    expand_radius: int = 0,
    chunks: int = 2048,
    depth: int = 100,
):
    """Function represents the segment widget and is called by the wizard to create the widget."""
    channels = channels.copy()
    if image is None:
        raise ValueError("Please select an image")

    fn_kwargs: Dict[str, Any] = {}

    pipeline = viewer.layers[utils.CLEAN].metadata["pipeline"]

    # need to load it back from zarr store, because otherwise not able to overwrite it
    sdata = read_zarr(pipeline.cfg.paths.sdata)

    if image.name == utils.CLEAN:
        log.info(
            f"Running segmentation on image layer '{utils.CLEAN}', "
            f"corresponding to image layer '{pipeline.cleaned_image_name}' in the SpatialData object used for backing."
        )
    elif image.name == utils.LOAD:
        log.info(
            f"Running segmentation on image layer '{utils.LOAD}', "
            f"corresponding to image layer '{pipeline.loaded_image_name}' in the SpatialData object used for backing."
        )
        # set cleaned image equal to loaded image in pipeline,
        # because we want to use image without cleaning applied for segmentation if we pick utils.LOAD
        pipeline.cleaned_image_name = pipeline.loaded_image_name

    else:
        raise ValueError(
            f"Please run the segmentation step on the layer with name '{utils.LOAD}' or '{utils.CLEAN}',"
            f"it seems layer with name '{image.name}' was selected."
        )

    # Subset shape
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

        pipeline.cfg.segmentation.crop_param = crd

    else:
        pipeline.cfg.segmentation.crop_param = None

    # update config
    pipeline.cfg.device = device
    pipeline.cfg.segmentation.small_size_vis = pipeline.cfg.segmentation.crop_param
    pipeline.cfg.segmentation.min_size = min_size
    pipeline.cfg.segmentation.flow_threshold = flow_threshold
    pipeline.cfg.segmentation.diameter = diameter
    pipeline.cfg.segmentation.cellprob_threshold = cellprob_threshold
    pipeline.cfg.segmentation.model_type = model_type.value
    pipeline.cfg.segmentation.channels = channels
    pipeline.cfg.segmentation.chunks = chunks
    pipeline.cfg.segmentation.depth = depth
    pipeline.cfg.segmentation.expand_radius = expand_radius

    # Bug in spatialdata. We can not pass overwrite==True if the labels layer is not yet available.
    if [*sdata.labels]:
        pipeline.cfg.segmentation.overwrite = True
    else:
        pipeline.cfg.segmentation.overwrite = False

    fn_kwargs["pipeline"] = pipeline

    worker = _segmentation_worker(sdata, segmentImage, fn_kwargs=fn_kwargs)

    def add_shape(sdata: SpatialData, pipeline: SparrowPipeline, layer_name: str):
        """Add the shapes to the napari viewer, overwrite if it already exists."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)

            log.info(f"Refreshing {layer_name}")
        except KeyError:
            # otherwise add it to the viewer
            log.info(f"Adding {layer_name}")

        if pipeline.cfg.segmentation.expand_radius:
            shapes_layer = f"expanded_cells_shapes_{pipeline.cfg.segmentation.expand_radius}"
        else:
            shapes_layer = pipeline.cfg.segmentation.output_shapes_layer

        polygons = _translate_polygons(sdata.shapes[shapes_layer].copy(), to_coordinate_system="global")

        polygons = utils._get_polygons_in_napari_format(df=polygons)

        show_info("Adding segmentation shapes, this can be slow on large images...")
        viewer.add_shapes(
            polygons,
            name=layer_name,
            shape_type="polygon",
            edge_color="coral",
            face_color="royalblue",
            edge_width=2,
            opacity=0.5,
        )

        # we need the original shapes, in order for next step (allocation) to be able to run allocation step multiple times
        shapes = {}
        for shapes_name, polygons in sdata.shapes.items():
            shapes[shapes_name] = polygons.copy()
        viewer.layers[layer_name].metadata["shapes"] = shapes
        viewer.layers[layer_name].metadata["pipeline"] = pipeline

        log.info(f"Added {utils.SEGMENT} layer")

        utils._export_config(
            pipeline.cfg.segmentation,
            os.path.join(pipeline.cfg.paths.output_dir, "configs", "segmentation", "plugin.yaml"),
        )

        show_info("Segmentation finished")

    worker.returned.connect(lambda data: add_shape(data, pipeline, utils.SEGMENT))  # type: ignore
    show_info("Segmentation started" + ", CPU selected: might take some time" if device == "cpu" else "")
    worker.start()

    return worker
