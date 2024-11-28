"""Annotation widget for scoring the genes, returns markergenes and adata objects."""

import os
import pathlib
from collections.abc import Callable
from typing import Any, List

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from spatialdata import SpatialData, read_zarr

import harpy.utils as utils
from harpy.pipeline import HarpyPipeline

log = utils.get_pylogger(__name__)


def annotateImage(
    sdata: SpatialData,
    pipeline: HarpyPipeline,
) -> SpatialData:
    """Function representing the annotation step, this calls all the needed functions to annotate the cells with the celltype."""
    sdata = pipeline.annotate(sdata)

    sdata = pipeline.visualize(
        sdata=sdata,
    )

    return sdata


@thread_worker(progress=True)
def _annotation_worker(sdata: SpatialData, method: Callable, fn_kwargs: dict[str, Any]) -> SpatialData:
    """Annotate data with marker genes in a thread worker"""
    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Annotate",
    markers_file={"widget_type": "FileEdit", "filter": "*.csv"},
)
def annotate_widget(
    viewer: napari.Viewer,
    markers_file: pathlib.Path = pathlib.Path(""),
    delimiter: str = ",",
    del_celltypes: List[str] = [],  # noqa: B006 # noqa: B006 # magicgui does not accept None
):
    """Function represents the annotation widget and is called by the wizard to create the widget."""
    # Check if a file was passed
    del_celltypes = del_celltypes.copy()

    if str(markers_file) in ["", "."]:
        raise ValueError("Please select marker file (.csv)")
    log.info(f"Marker file is {str(markers_file)}")

    # Load data from previous layers
    try:
        allocation_layer = viewer.layers[utils.ALLOCATION]
    except KeyError as err:
        raise RuntimeError(
            f"Layer with name '{utils.ALLOCATION}' is not available. Please run allocation step before running annotation step."
        ) from err

    try:
        pipeline = allocation_layer.metadata["pipeline"]
    except KeyError as err:
        raise RuntimeError("Please run allocation step before running annotation step.") from err

    # need to load it back from zarr store, because otherwise not able to overwrite it
    sdata = read_zarr(pipeline.cfg.paths.sdata)

    pipeline.cfg.dataset.markers = markers_file
    pipeline.cfg.annotate.del_celltypes = del_celltypes
    pipeline.cfg.annotate.delimiter = delimiter

    fn_kwargs = {
        "pipeline": pipeline,
    }

    worker = _annotation_worker(sdata, annotateImage, fn_kwargs)

    def add_metadata(
        sdata: SpatialData,
        pipeline: HarpyPipeline,
        layer_name: str,
    ):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""
        if layer_name not in viewer.layers:
            log.info(f"Layer '{layer_name}' does not exist. Please run allocation step before running annotation step.")
            raise KeyError(f"Layer '{layer_name}' does not exist")

        # Store data in previous layer

        viewer.layers[layer_name].metadata["pipeline"] = pipeline
        viewer.layers[layer_name].metadata["adata"] = sdata.tables[
            pipeline.cfg.allocate.table_layer_name
        ]  # spatialdata plugin uses this

        utils._export_config(
            pipeline.cfg.annotate,
            os.path.join(pipeline.cfg.paths.output_dir, "configs", "annotate", "plugin.yaml"),
        )

        log.info("Annotation metadata added")
        show_info("Annotation finished")

    worker.returned.connect(lambda data: add_metadata(data, pipeline, utils.ALLOCATION))
    show_info("Annotation started")
    worker.start()

    return worker
