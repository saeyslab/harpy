from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._keys import _ANNOTATION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import squidpy as sq
except ImportError:
    log.warning("'squidpy' not installed, to use 'harpy.tb.nhood_enrichment' please install this library.")


def nhood_enrichment(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    celltype_column: str = _ANNOTATION_KEY,
    seed: int = 0,
    overwrite: bool = False,
) -> SpatialData:
    """
    Calculate the nhood enrichment using squidpy via `sq.gr.spatial_neighbors` and `sq.gr.nhood_enrichment`.

    Parameters
    ----------
    sdata
        Input SpatialData object containing spatial data.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata`.
    output_layer
        The output table layer in `sdata`.
    celltype_column
        This will be passed to `cluster_key` of `squidpy.gr.nhood_enrichment`.
    seed
        seed
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=celltype_column, seed=seed)

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata
