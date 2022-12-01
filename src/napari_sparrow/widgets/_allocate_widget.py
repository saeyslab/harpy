"""
Allocation widget for creating and preprocesing the adata object, filtering the cells and performing clustering.
"""
import pathlib
from typing import Callable, Tuple

import napari
import napari.layers
import napari.types
import numpy as np
import scanpy as sc
import squidpy.im as sq
from anndata import AnnData
from magicgui import magic_factory, magicgui, widgets
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

import napari_sparrow.utils as utils
from napari_sparrow import functions as fc

from napari_sparrow.utils import UMAPPlotWidget

log = utils.get_pylogger(__name__)


def allocateImage(
    path: str,
    ic: sq.ImageContainer,
    masks: np.ndarray,
    pcs: int,
    neighbors: int,
    library_id: str = "melanoma",
    min_size: int = 100,
    max_size: int = 100000,
    cluster_resolution: float = 0.8,
    n_comps: int = 50,
) -> AnnData:
    """Function representing the allocation step, this calls all the needed functions to allocate the transcripts to the cells."""

    adata = fc.create_adata_quick(path, ic, masks, library_id)
    adata, _ = fc.preprocessAdata(adata, masks, n_comps=n_comps)
    adata, _ = fc.filter_on_size(adata, min_size, max_size)
    adata = fc.extract(ic, adata)
    adata = fc.clustering(adata, pcs, neighbors, cluster_resolution)
    return adata


@magicgui(
    call_button="Allocate",
    transcripts_file={"widget_type": "FileEdit", "filter": "*.txt"},
)
def allocation_parameters_widget(
    viewer: napari.Viewer,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    library_id: str = "melanoma",   # FIXME - pick more neutral default value
    min_size=500,
    max_size=100000,
    pcs: int = 17,
    neighbors: int = 35,
    cluster_resolution: float = 0.8,
    n_components: int = 50,
):
    """This function represents the allocate widget and is called by the wizard to create the widget."""

    # Check if a transcripts file was passed
    if str(transcripts_file) in ["", "."]:
        raise ValueError("Please select transcripts file (.txt)")
    log.info(f"Transcripts file is {str(transcripts_file)}")

    # Load data from previous layers
    try:
        ic = viewer.layers[utils.SEGMENT].metadata["ic"]
        masks = viewer.layers[utils.SEGMENT].data_raw
    except KeyError:
        raise RuntimeError("Please run previous steps first")

    @thread_worker(progress=True)
    def allocation_worker(*args, **kwargs):
        adata = allocateImage(*args, **kwargs)
        return adata

    worker = allocation_worker(str(transcripts_file), ic, masks, pcs, neighbors, library_id, min_size, max_size, cluster_resolution, n_components)
    return worker


class allocate_widget(widgets.Container):

    def __init__(self):
        super().__init__(labels=False)  # labels=False to suppress useless label with name of Container

        # Add allocation parameters magicgui widget.
        self.parameters_widget = allocation_parameters_widget
        self.parameters_widget.margins=(0, 0, 0, 0)
        self.append(self.parameters_widget)

        # Perform allocation of transcripts to cells in a worker thread
        # when the user clicked the Allocate button in the params_widget.
        self.parameters_widget.called.connect(self.start_allocation)

        # Add plot widget
        self.plot_widget = UMAPPlotWidget()
        self.plot_widget.hide() 
        self.native.layout().addWidget(self.plot_widget)

    def start_allocation(self, worker):
        log.info(f'Starting allocation in worker thread')
        worker.returned.connect(self.allocation_done)
        worker.start()

    def allocation_done(self, allocation_result: AnnData):
        """ Called by the allocation worker thread to return the result of the allocation. """
        adata = allocation_result
        viewer = self.parameters_widget.viewer.value
        library_id = self.parameters_widget.library_id.value

        self.add_metadata(viewer, adata, library_id) 

        # Begin hack: call sc.pl.umap() to trigger creation of
        # adata.uns['leiden_colors'] but ignore the returned plot.
        # FIXME - set colors some other way
        sc.pl.umap(adata, color=["leiden"], show=False)
        # End hack

        data = adata.obsm['X_umap']
        cluster_ids = adata.obs['leiden']
        cluster_colors = np.array(adata.uns['leiden_colors'])
        legend_columns = 4

        self.plot_widget.set_data(data, cluster_ids, cluster_colors)
        self.plot_widget.clear()
        self.plot_widget.draw(legend_columns)
        self.plot_widget.show()
        show_info("Allocation finished")

    def add_metadata(self,
                     viewer: napari.Viewer,
                     adata: AnnData,
                     library_id: str):
        """Add the metadata to the previous layer, this way it becomes available in the next steps."""
        try:
            # check if the previous layer exists
            layer = viewer.layers[utils.SEGMENT]
        except KeyError:
            log.info(f"Layer does not exist {utils.SEGMENT}")
            return

        # Store data in previous layer
        layer.metadata["adata"] = adata
        layer.metadata["library_id"] = library_id
        layer.metadata["labels_key"] = "cell_ID"
        layer.metadata["points"] = adata.uns["spatial"][library_id]["points"]
        layer.metadata["point_diameter"] = 10
