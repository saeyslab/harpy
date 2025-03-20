import dask.array as da
import numpy as np
from anndata import AnnData

from harpy.table.pixel_clustering._neighbors import spatial_pixel_neighbors
from harpy.utils._keys import _SPATIAL


def test_spatial_pixel_neighbors(sdata):
    # note: hp.tb.spatial_pixel_neighbors would typically be run on a labels layer obtained via `hp.im.flowsom`.
    adata = spatial_pixel_neighbors(
        sdata, labels_layer="blobs_labels", key_added="cluster_id", size=50, mode="center", subset=None
    )

    assert isinstance(adata, AnnData)
    # sanity check to see if we sampled all. Will evidentily fail if size is too large (e.g. size==100)
    assert np.array_equal(
        np.array(adata.obs["cluster_id"].cat.categories), da.unique(sdata["blobs_labels"].data).compute()
    )

    index = 2
    assert (
        adata.obs["cluster_id"][index]
        == sdata["blobs_labels"].data.compute()[adata.obsm["spatial"][index][0], adata.obsm[_SPATIAL][index][1]]
    )

    # divide in hexagon grid and take most frequent occuring cluster id in each bin.
    adata = spatial_pixel_neighbors(
        sdata,
        labels_layer="blobs_labels",
        key_added="cluster_id",
        size=50,
        mode="most_frequent",
        grid_type="hexagon",
        subset=None,
    )

    assert isinstance(adata, AnnData)
    assert adata.shape[0] == 143
    assert adata.obs.iloc[25].item() == 31
