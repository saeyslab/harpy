import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import diags
from spatialdata.models import TableModel

from harpy.table import nhood_kmeans
from harpy.utils._keys import _ANNOTATION_KEY


def _chain_graph(n_obs: int):
    """Return a simple line graph where each cell is connected to its immediate neighbors.

    The resulting sparse matrix has ones on the diagonals at offsets ``-1`` and
    ``+1``, so observation ``i`` is connected to ``i - 1`` and ``i + 1`` when
    those indices exist. For example, for ``n_obs = 5`` this encodes the chain
    ``0 - 1 - 2 - 3 - 4`` and the matrix

    ``[[0, 1, 0, 0, 0],``
    `` [1, 0, 1, 0, 0],``
    `` [0, 1, 0, 1, 0],``
    `` [0, 0, 1, 0, 1],``
    `` [0, 0, 0, 1, 0]]``.

    The tests use this deterministic graph together with alternating
    ``even``, ``odd``, ``even``, ``odd``, ... labels so the expected
    neighborhood compositions are easy to verify by hand. For example, cell 0
    is labeled ``even`` but its only neighbor is cell 1, labeled ``odd``, so
    its neighborhood composition is ``[0.0, 1.0]``. Cell 1 is labeled ``odd``
    and its neighbors are cells 0 and 2, both labeled ``even``, so its
    neighborhood composition is ``[1.0, 0.0]``.
    """
    values = np.ones(n_obs - 1, dtype=np.float32)
    return diags([values, values], offsets=[-1, 1], shape=(n_obs, n_obs), format="csr")


def test_nhood_kmeans(sdata_transcripts_no_backed):
    """Test neighborhood composition and clustering on a simple chain graph.

    The chain graph

    ``[[0, 1, 0, 0, 0],``
    `` [1, 0, 1, 0, 0],``
    `` [0, 1, 0, 1, 0],``
    `` [0, 0, 1, 0, 1],``
    `` [0, 0, 0, 1, 0]]``

    is multiplied by the one-hot encoded cell-type matrix

    ``[[1, 0],``
    `` [0, 1],``
    `` [1, 0],``
    `` [0, 1],``
    `` [1, 0]]``

    for alternating ``even``/``odd`` labels and the categories ``[even, odd]``.

    by that one-hot matrix gives neighbor counts

    ``[[0, 1],``
    `` [2, 0],``
    `` [0, 2],``
    `` [2, 0],``
    `` [0, 1]]``.

    Row-wise normalization divides each row by its row sum, here
    ``[1, 2, 2, 2, 1]``, to convert neighbor counts into fractions. This gives
    the neighborhood-composition matrix

    ``[[0.0, 1.0],``
    `` [1.0, 0.0],``
    `` [0.0, 1.0],``
    `` [1.0, 0.0],``
    `` [0.0, 1.0]]``.

    Cells 0 and 1 therefore have neighborhood compositions ``[0.0, 1.0]`` and
    ``[1.0, 0.0]``, respectively, which are asserted below.
    """
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
        cluster_key=_ANNOTATION_KEY,
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

    assert result.uns["nhood_composition"]["cluster_key"] == _ANNOTATION_KEY
    assert result.uns["nhood_composition"]["connectivity_key"] == "radius_test_connectivities"
    cluster_categories = result.uns["nhood_composition"]["cluster_categories"]
    if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.2.5"):
        assert json.loads(cluster_categories) == ["even", "odd"]
    else:
        assert cluster_categories.tolist() == ["even", "odd"]

    assert result.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["segmentation_mask"]


def test_nhood_kmeans_labels_isolated_cells(sdata_transcripts_no_backed):
    """Test that isolated cells keep an all-zero composition and receive ``nan_label``.

    Starting from the same chain graph as in ``test_nhood_kmeans``, this test
    removes the edge between cells 0 and 1 by setting entries ``(0, 1)`` and
    ``(1, 0)`` to zero. Cell 0 then has no neighbors at all, while cell 1
    remains connected to cell 2. Both entries are cleared to keep the
    connectivity matrix symmetric, since the graph is treated as an undirected
    neighbor graph. The updated graph is

    ``[[0, 0, 0, 0, 0],``
    `` [0, 0, 1, 0, 0],``
    `` [0, 1, 0, 1, 0],``
    `` [0, 0, 1, 0, 1],``
    `` [0, 0, 0, 1, 0]]``.

    Multiplying this graph by the one-hot encoded cell-type matrix

    ``[[1, 0],``
    `` [0, 1],``
    `` [1, 0],``
    `` [0, 1],``
    `` [1, 0]]``

    for alternating ``even``/``odd`` labels gives neighbor counts

    ``[[0, 0],``
    `` [1, 0],``
    `` [0, 2],``
    `` [2, 0],``
    `` [0, 1]]``.

    With alternating ``even``/``odd`` labels, the neighborhood counts for cell
    0 are therefore ``[0, 0]``. Row-wise normalization keeps this row at
    ``[0.0, 0.0]``, and the clustering code should assign the configured
    ``nan_label`` instead of a KMeans cluster. The assertions below check both
    behaviors: cell 0 is labeled ``"isolated"`` and its neighborhood
    composition remains all zeros.
    """
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
        cluster_key=_ANNOTATION_KEY,
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
            cluster_key=_ANNOTATION_KEY,
            connectivity_key="missing_graph",
            overwrite=True,
        )
