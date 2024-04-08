import squidpy as sq
from spatialdata import SpatialData

from sparrow.table._table import ProcessTable, _add_table_layer
from sparrow.utils._keys import _ANNOTATION_KEY


def nhood_enrichment(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    celltype_column: str = _ANNOTATION_KEY,
    seed: int = 0,
    overwrite: bool = False,
) -> SpatialData:
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    # Adaptations for saving
    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=celltype_column, seed=seed)

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata
