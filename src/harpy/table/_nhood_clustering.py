from spatialdata import SpatialData
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from harpy.table._table import add_table_layer
from harpy.utils._keys import _REGION_KEY

def nhood_count(
    sdata: SpatialData,
    table_layer: str,
    output_layer: str,
    cell_type_column: str,
    connectivity_key: str = "spatial_connectivities",
    save_key="nhood_counts", 
    overwrite: bool = False,
):
    """
    Compute for each cell the number and fraction of neighbours that belong to every cell type. This uses an existing spatial graph calculated by Squidpy.

    The results are saved to `adata.uns[{cell_type_column}_{save_key}]` as (N × K) DataFrames.
    
    Args:
        sdata:
            The SpatialData object containing the input table.
        table_layer:
            The table layer in `sdata.tables` to use as input.
        output_layer:
            The output table layer in `sdata.tables` to which the updated table layer will be written.
        cell_type_column:
            The column in `sdata.tables[table_layer].obs` containing the cell type annotations.
        connectivity_key:
            The key in `sdata.tables[table_layer].obsp` that contains the connectivity matrix from `squidpy.gr.spatial_neighbors()`. Defaults to `spatial_connectivities`.
        save_key:
            '{save_key}' will be used to store the results in `sdata.tables[table_layer].uns`. Defaults to `nhood_counts`.
        overwrite
            If True, overwrites the `output_layer` if it already exists in `sdata`.
    
    Returns:
    -------
    The SpatialData object containing the updated AnnData object as an attribute (`sdata.tables[output_layer]`).
    """
    
    # Create copy of table layer
    adata = sdata.tables[table_layer].copy()
    
    # Get connectivities
    connectivities = adata.obsp[connectivity_key].tocsr() # N × N sparse

    # One‑hot encode cell types 
    cell_types = adata.obs[cell_type_column].astype("category")
    onehot = pd.get_dummies(cell_types, sparse=True) # N × K
    onehot_mat = onehot.sparse.to_coo().tocsr()

    # Counts neighbours of each cell type
    counts = connectivities.dot(onehot_mat) # N × K sparse

    # Convert to fractions
    neigh_totals = np.asarray(connectivities.sum(axis=1)).ravel() # length N
    with np.errstate(divide="ignore"):
        frac = counts.multiply(1 / neigh_totals[:, None])

    frac = frac.toarray()
    frac[np.isnan(frac)] = 0 # Set isolated cells to 0

    # Add results to adata.uns
    dict_uns = {
        "cell_type_column": cell_type_column,
        "rows": adata.obs_names.to_list(),
        "columns": cell_types.cat.categories.to_list(),
        "counts": counts.toarray(),
        "fractions": frac,
    }

    adata.uns[save_key] = dict_uns

    # Add table layer  
    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=overwrite,
    )
    
    return sdata


def nhood_kmeans(
    sdata: SpatialData,
    table_layer: str,
    output_layer: str,
    nhood_counts_key: str = "nhood_counts",
    output_column: str = "nhood_kmeans",
    n_clusters: int = 5,
    random_state: int = 100,
    nan_label: int | str | None = -1,
    overwrite: bool = False,
):
    """
    K‑means clustering on neighbour cell‑type composition. `harpy.table.nhood_count` must be run first to compute the neighbour cell‑type composition.

    Args:
        sdata:
            The SpatialData object containing the table to perform kmeans clustering.
        table_layer:
            The table layer in `sdata.tables` that will be used as input for kmeans clustering.
        output_layer:
            The output table layer in `sdata.tables` to which the updated table layer will be written.
        nhood_counts_key:
            Key in `sdata.tables[table_layer].uns` where counts/fractions dateframes are stored. Defaults to "nhood_counts".
        output_column:
            The key under which the clustering results are added to the SpatialData object (in `sdata.tables[table_layer].obs`). Defaults to "nhood_kmeans".
        n_clusters :
            k for k‑means. Defaults to 5.
        random_state :
            A random state for reproducibility of the clustering. Defaults to 100.
        nan_label :
            Label to assign to cells that had no neighbours (all zeros). Defaults to -1.
            Use `None` to leave them as `NaN`.
        overwrite
            If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns:
    -------
    The SpatialData object containing the updated AnnData object as an attribute (`sdata.tables[output_layer]`).
    """
    # Create copy of table layer
    adata = sdata.tables[table_layer].copy()

    # Check uns key
    if nhood_counts_key not in adata.uns:
        raise KeyError(
            f"`{nhood_counts_key}` not found in `adata.uns`. "
            "Run `nhood_count` first or check your keys."
        )

    # Get fractions matrix
    frac = adata.uns[nhood_counts_key]['fractions'] # dense (N × K)

    # Handle cells with zero neighbours (all‑zero rows)
    mask_valid = frac.sum(axis=1) > 0
    frac_valid = frac[mask_valid]

    # Run k‑means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit(frac_valid)

    labels_full = np.full(frac.shape[0], nan_label, dtype=object)
    labels_full[mask_valid] = kmeans.labels_
    adata.obs[output_column] = pd.Categorical(labels_full)
    
    # Add table layer  
    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=overwrite,
    )
    
    return sdata
