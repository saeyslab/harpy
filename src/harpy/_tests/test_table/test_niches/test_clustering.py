import numpy as np
import pandas as pd
import pytest
from scipy.sparse import diags
from spatialdata.models import TableModel

from harpy.table import nhood_kmeans
from harpy.utils._keys import _ANNOTATION_KEY


def _chain_graph(n_obs: int):
    values = np.ones(n_obs - 1, dtype=np.float32)
    return diags([values, values], offsets=[-1, 1], shape=(n_obs, n_obs), format="csr")


def test_nhood_kmeans(sdata_transcripts_no_backed):
    table_layer = "table_transcriptomics"
    output_layer = "table_transcriptomics_niches"
    adata = sdata_transcripts_no_backed.tables[table_layer]

    adata.obs[_ANNOTATION_KEY] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "even", "odd"))
    adata.obsp["radius_test_connectivities"] = _chain_graph(adata.n_obs)

    sdata_transcripts_no_backed = nhood_kmeans(
        sdata=sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer=table_layer,
        output_layer=output_layer,
        instance_type_key=_ANNOTATION_KEY,
        connectivity_key="radius_test",
        n_clusters=2,
        overwrite=True,
    )

    result = sdata_transcripts_no_backed.tables[output_layer]

    assert "nhood_kmeans" in result.obs.columns
    assert isinstance(result.obs["nhood_kmeans"].dtype, pd.CategoricalDtype)
    assert result.obs["nhood_kmeans"].cat.categories.size == 2

    assert result.obsm["nhood_composition"].shape == (adata.n_obs, 2)
    assert np.allclose(result.obsm["nhood_composition"][0], [0.0, 1.0])
    assert np.allclose(result.obsm["nhood_composition"][1], [1.0, 0.0])

    assert result.uns["nhood_composition"]["instance_type_key"] == _ANNOTATION_KEY
    assert result.uns["nhood_composition"]["connectivity_key"] == "radius_test_connectivities"
    assert result.uns["nhood_composition"]["instance_type_categories"].tolist() == ["even", "odd"]

    assert result.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["segmentation_mask"]


def test_nhood_kmeans_labels_isolated_cells(sdata_transcripts_no_backed):
    table_layer = "table_transcriptomics"
    output_layer = "table_transcriptomics_niches"
    adata = sdata_transcripts_no_backed.tables[table_layer]

    adata.obs[_ANNOTATION_KEY] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "even", "odd"))
    graph = _chain_graph(adata.n_obs).tolil()
    graph[0, 1] = 0
    graph[1, 0] = 0
    adata.obsp["manual_connectivities"] = graph.tocsr()

    sdata_transcripts_no_backed = nhood_kmeans(
        sdata=sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer=table_layer,
        output_layer=output_layer,
        instance_type_key=_ANNOTATION_KEY,
        connectivity_key="manual_connectivities",
        composition_key="manual_nhood_composition",
        key_added="manual_nhood_kmeans",
        n_clusters=2,
        nan_label="isolated",
        overwrite=True,
    )

    result = sdata_transcripts_no_backed.tables[output_layer]

    assert result.obs["manual_nhood_kmeans"].iloc[0] == "isolated"
    assert np.allclose(result.obsm["manual_nhood_composition"][0], [0.0, 0.0])
    assert result.obs["manual_nhood_kmeans"].iloc[1] != "isolated"


def test_nhood_kmeans_requires_existing_connectivity_key(sdata_transcripts_no_backed):
    adata = sdata_transcripts_no_backed.tables["table_transcriptomics"]
    adata.obs[_ANNOTATION_KEY] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "even", "odd"))

    with pytest.raises(KeyError, match="Connectivity key 'missing_graph' not found"):
        nhood_kmeans(
            sdata=sdata_transcripts_no_backed,
            labels_layer="segmentation_mask",
            table_layer="table_transcriptomics",
            output_layer="table_transcriptomics_niches",
            instance_type_key=_ANNOTATION_KEY,
            connectivity_key="missing_graph",
            overwrite=True,
        )
