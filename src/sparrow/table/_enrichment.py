import squidpy as sq
from spatialdata import SpatialData

from sparrow.table._table import _back_sdata_table_to_zarr
from sparrow.utils._keys import _ANNOTATION_KEY


def nhood_enrichment(sdata: SpatialData, celltype_column: str = _ANNOTATION_KEY, seed: int = 0) -> SpatialData:
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """
    # Adaptations for saving
    sdata.table.raw.var.index.names = ["genes"]
    sdata.table.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(sdata.table, coord_type="generic")
    sq.gr.nhood_enrichment(sdata.table, cluster_key=celltype_column, seed=seed)
    _back_sdata_table_to_zarr(sdata=sdata)
    return sdata
