from typing import Dict, Tuple

import numpy as np
import spatialdata
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.shape._shape import _filter_shapes_layer
from sparrow.utils._keys import _CELLSIZE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ProcessTable:
    def __init__(
        self,
        sdata: SpatialData,
        labels_layer: str,
    ):
        """
        Base class for implementation of processing on tables.

        Parameters
        ----------
        - spatial_data: SpatialData
            The SpatialData object containing spatial data.
        - labels_layer: str
            The label layer to use.
        """
        if not hasattr(sdata, "table"):
            raise ValueError(
                "Provided SpatialData object 'sdata' does not have 'table' attribute. "
                "Please create table attribute via e.g. 'sp.tb.allocation' or 'sp.tb.allocation_intensity' functions."
            )
        if not hasattr(sdata, "labels"):
            raise ValueError(
                "Provided SpatialData object 'sdata' does not have 'labels' attribute. "
                "Please create labels attribute via e.g. 'sp.im.segment'."
            )
        self.sdata = sdata
        self.labels_layer = labels_layer
        self._validate_labels_layer()

    def _validate_labels_layer(self):
        """Validate if the specified labels layer exists in the SpatialData object."""
        if self.labels_layer not in [*self.sdata.labels]:
            raise ValueError("labels layer not in 'sdata.labels'")
        if self.labels_layer not in self.sdata.table.obs[_REGION_KEY].cat.categories:
            raise ValueError("labels layer not in 'sdata.table.obs[_REGION_KEY].cat.categories'")

    def _get_adata(self) -> AnnData:
        """Preprocess the data by filtering based on the labels layer and setting attributes."""
        adata = self.sdata.table[self.sdata.table.obs[_REGION_KEY] == self.labels_layer].copy()
        adata.uns["spatialdata_attrs"]["region"] = [self.labels_layer]
        return adata


def correct_marker_genes(
    sdata: SpatialData,
    celltype_correction_dict: Dict[str, Tuple[float, float]],
):
    """Returns the updated SpatialData object.

    Corrects celltypes that are higher expessed by dividing them by a value if they exceed a certain threshold.
    The celltype_correction_dict has as keys the celltypes that should be corrected and as values the threshold and the divider.
    """
    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in sdata.table.obs.columns:
            log.info(
                f"Cell type '{celltype}' not in obs of AnnData object. Skipping. Please first calculate gene expression for this cell type."
            )
            continue
        sdata.table.obs[celltype] = np.where(
            sdata.table.obs[celltype] < values[0],
            sdata.table.obs[celltype] / values[1],
            sdata.table.obs[celltype],
        )

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata


def filter_on_size(sdata: SpatialData, min_size: int = 100, max_size: int = 100000):
    """Returns the updated SpatialData object.

    All cells with a size outside of the min and max size range are removed.
    """
    start = sdata.table.shape[0]

    # Filter cells based on size and distance
    table = sdata.table[sdata.table.obs[_CELLSIZE_KEY] < max_size, :]
    table = table[table.obs[_CELLSIZE_KEY] > min_size, :]
    del sdata.table
    ## TODO: Look for a better way of doing this!
    sdata.table = spatialdata.models.TableModel.parse(table)

    indexes_to_keep = sdata.table.obs.index.values.astype(int)
    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_size",
    )

    filtered = start - table.shape[0]
    log.info(f"{filtered} cells were filtered out based on size.")

    return sdata


def _back_sdata_table_to_zarr(sdata: SpatialData):
    adata = sdata.table.copy()
    del sdata.table
    sdata.table = spatialdata.models.TableModel.parse(adata)
