"""Allocation widget for creating and preprocesing the adata object, filtering the cells and performing clustering."""

import os
import pathlib
from typing import Any, Callable, Dict, Optional

import napari
import napari.layers
import napari.types
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from spatialdata import SpatialData, read_zarr

import sparrow.utils as utils
from sparrow.pipeline import SparrowPipeline
from sparrow.plot._plot import _translate_polygons
from sparrow.shape._shape import add_shapes_layer

log = utils.get_pylogger(__name__)


def allocateImage(
    sdata: SpatialData,
    pipeline: SparrowPipeline,
) -> SpatialData:
    """Function representing the allocation step, this calls all the needed functions to allocate the transcripts to the cells."""
    sdata = pipeline.allocate(sdata)

    return sdata


@thread_worker(progress=True)
def _allocation_worker(
    sdata: SpatialData,
    method: Callable,
    fn_kwargs: Dict[str, Any],
) -> SpatialData:
    """Allocate transcripts in a thread worker"""
    return method(sdata, **fn_kwargs)


@magic_factory(
    call_button="Allocate",
    transcripts_file={"widget_type": "FileEdit"},
    transform_matrix={"widget_type": "FileEdit"},
)
def allocate_widget(
    viewer: napari.Viewer,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    delimiter: str = "\t",
    header: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_gene: int = 3,
    midcount: bool = False,
    column_midcount: Optional[int] = None,
    transform_matrix: Optional[pathlib.Path] = None,
    min_counts: int = 10,
    min_cells: int = 5,
    size_normalization: bool = True,
    n_comps: int = 50,
    min_size: int = 500,
    max_size: int = 100000,
    pcs: int = 17,
    neighbors: int = 35,
    cluster_resolution: float = 0.8,
):
    """Function represents the allocate widget and is called by the wizard to create the widget."""
    # Check if a file was passed
    if str(transcripts_file) in ["", "."]:
        raise ValueError("Please select transcripts file (.txt)")
    log.info(f"Transcripts file is {str(transcripts_file)}")

    # Load data from previous layers
    try:
        segment_layer = viewer.layers[utils.SEGMENT]
    except KeyError as err:
        raise RuntimeError(f"Layer with name '{utils.SEGMENT}' is not available") from err

    try:
        pipeline = segment_layer.metadata["pipeline"]
        shapes = segment_layer.metadata["shapes"]
    except KeyError as err:
        raise RuntimeError("Please run segmentation step before running allocation step.") from err

    # need to load it back from zarr store, because otherwise not able to overwrite it
    sdata = read_zarr(pipeline.cfg.paths.sdata)

    # need to add original unfiltered shapes to sdata object at the beginning of the allocation step.
    # otherwise polygons that were filtered out would not be available any more if you do a rerun of the allocation step.
    for shapes_name in [*shapes]:
        sdata = add_shapes_layer(sdata, input=shapes[shapes_name], output_layer=shapes_name, overwrite=True)

    # napari widget does not support the type Optional[int], therefore only choose whether there is a header or not,
    # and do same for midcount column
    if header:
        header = 0
    else:
        header = None

    if midcount:
        column_midcount = column_midcount
    else:
        column_midcount = None

    pipeline.cfg.dataset.coords = transcripts_file
    if transform_matrix:
        pipeline.cfg.dataset.transform_matrix = transform_matrix

    pipeline.cfg.allocate.size_norm = size_normalization
    pipeline.cfg.allocate.min_counts = min_counts
    pipeline.cfg.allocate.min_cells = min_cells
    pipeline.cfg.allocate.n_comps = n_comps
    pipeline.cfg.allocate.min_size = min_size
    pipeline.cfg.allocate.max_size = max_size
    pipeline.cfg.allocate.pcs = pcs
    pipeline.cfg.allocate.neighbors = neighbors
    pipeline.cfg.allocate.cluster_resolution = cluster_resolution
    pipeline.cfg.allocate.delimiter = delimiter
    pipeline.cfg.allocate.header = header
    pipeline.cfg.allocate.column_x = column_x
    pipeline.cfg.allocate.column_y = column_y
    pipeline.cfg.allocate.column_gene = column_gene
    pipeline.cfg.allocate.column_midcount = column_midcount
    pipeline.cfg.allocate.overwrite = True

    fn_kwargs = {
        "pipeline": pipeline,
    }

    worker = _allocation_worker(sdata, allocateImage, fn_kwargs=fn_kwargs)

    def add_metadata(sdata: SpatialData, pipeline: SparrowPipeline, layer_name: str):
        """Update the polygons, add anndata object to the metadata, so it can be viewed via napari spatialdata plugin, and it becomes visible in next steps."""
        try:
            # if the layer exists, update its data
            layer = viewer.layers[layer_name]
            viewer.layers.remove(layer)
            log.info(f"Refreshing {layer_name}")

        except KeyError:
            log.info(f"Layer '{layer_name}' does not exist.")

        polygons = _translate_polygons(sdata.shapes[pipeline.shapes_layer_name].copy(), to_coordinate_system="global")

        polygons = utils._get_polygons_in_napari_format(df=polygons)

        # we add the polygons again in this step, because some of them are filtered out in the allocate step
        # (i.e. due to size of polygons etc). If we do not update the polygons here, napari complains because
        # number of polygons does not match any more with e.g. number of polygons with a leiden cluster assigned.
        show_info("Adding updated segmentation shapes, this can be slow on large images...")
        viewer.add_shapes(
            polygons,
            name=layer_name,
            shape_type="polygon",
            edge_color="coral",
            face_color="royalblue",
            edge_width=2,
            opacity=0.5,
        )

        viewer.layers[layer_name].metadata["pipeline"] = pipeline
        viewer.layers[layer_name].metadata["adata"] = sdata.tables[
            pipeline.cfg.allocate.table_layer_name
        ]  # spatialdata plugin uses this

        log.info(f"Added '{utils.ALLOCATION}' layer")

        utils._export_config(
            pipeline.cfg.allocate,
            os.path.join(pipeline.cfg.paths.output_dir, "configs", "allocate", "plugin.yaml"),
        )

        show_info("Allocation finished")

        # Options for napari-spatialData plugin
        viewer.scale_bar.visible = True

    worker.returned.connect(lambda data: add_metadata(data, pipeline, f"{utils.ALLOCATION}"))
    show_info("Allocation started")
    worker.start()

    return worker
