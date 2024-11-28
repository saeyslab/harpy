from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData

from harpy.utils._keys import ClusteringKey
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _get_mapping(
    adata: AnnData,
    keys: tuple[str, str] = (ClusteringKey._CLUSTERING_KEY.value, ClusteringKey._METACLUSTERING_KEY.value),
) -> pd.DataFrame:
    """Get mapping between flowsom clusters and metaclusters"""
    mapping = adata.obs[list(keys)]
    mapping = mapping.drop_duplicates().set_index(keys[0]).sort_index()[keys[1]]
    mapping.index.name = None
    mapping = mapping.astype(int)
    return mapping


def _export_to_ark_format(
    sdata: SpatialData, table_layer: str, output: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    clustering_key = ClusteringKey._CLUSTERING_KEY.value
    metaclustering_key = ClusteringKey._METACLUSTERING_KEY.value
    df_cell_som_cluster_count_avg = sdata.tables[table_layer].uns[clustering_key].copy()
    df_cell_som_cluster_count_avg.reset_index(inplace=True)
    log.warning(
        "Increasing cell cluster IDs (SOM cluster and meta cluster IDs) with +1 for visualization. The underlying dataframe in the SpatialData object remains unchanges."
    )
    df_cell_som_cluster_count_avg[clustering_key] += 1
    df_cell_som_cluster_count_avg[metaclustering_key] += 1
    df_cell_som_cluster_count_avg.rename(
        columns={clustering_key: "cell_som_cluster", metaclustering_key: "cell_meta_cluster"}, inplace=True
    )
    if output is not None:
        output_file = os.path.join(output, "cell_som_cluster_count_avg.csv")
        log.info(f"writing cell som cluster count average to {output_file}")
        df_cell_som_cluster_count_avg.to_csv(output_file, index=False)

    df_cell_som_cluster_channel_avg = None
    df_cell_meta_cluster_channel_avg = None
    for _key in [clustering_key, metaclustering_key]:
        if f"{_key}_channels" in sdata.tables[table_layer].uns:
            df = sdata.tables[table_layer].uns[f"{_key}_channels"].copy()
            df.reset_index(inplace=True)
            log.warning(
                "Increasing cell cluster IDs (SOM cluster and meta cluster IDs) with +1 for visualization. The underlying dataframe in the SpatialData object remains unchanges."
            )
            if _key == clustering_key:
                df[clustering_key] = df[clustering_key].astype(int)
                df[clustering_key] += 1
            df[metaclustering_key] = df[metaclustering_key].astype(int)
            df[metaclustering_key] += 1
            if _key == clustering_key:
                df.rename(columns={clustering_key: "cell_som_cluster"}, inplace=True)
            df.rename(columns={metaclustering_key: "cell_meta_cluster"}, inplace=True)
            df["cell_meta_cluster_rename"] = df[
                "cell_meta_cluster"
            ]  # need to add this dummy column otherwise not able to visualize in ark
            if output is not None:
                output_file = os.path.join(output, f"cell_{_key}_channel_avg.csv")
                log.info(f"writing {_key} channel average to {output_file}")
                df.to_csv(output_file, index=False)

            if _key == clustering_key:
                df_cell_som_cluster_channel_avg = df
            else:
                df_cell_meta_cluster_channel_avg = df

    return df_cell_som_cluster_count_avg, df_cell_som_cluster_channel_avg, df_cell_meta_cluster_channel_avg
