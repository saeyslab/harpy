import dask.array as da
import numpy as np

from harpy.table.pixel_clustering._neighbors import spatial_pixel_neighbors
from harpy.utils._keys import _SPATIAL


def test_spatial_pixel_neighbors(sdata):
    # note: hp.tb.spatial_pixel_neighbors would typically be run on a labels layer obtained via `hp.im.flowsom`.
    adata = spatial_pixel_neighbors(sdata, labels_layer="blobs_labels", key_added="cluster_id", size=50, subset=None)

    # sanity check to see if we sampled all. Will evidentily fail if size is too large (e.g. size==100)
    assert np.array_equal(
        np.array(adata.obs["cluster_id"].cat.categories), da.unique(sdata["blobs_labels"].data).compute()
    )

    index = 2
    assert (
        adata.obs["cluster_id"][index]
        == sdata["blobs_labels"].data.compute()[adata.obsm["spatial"][index][0], adata.obsm[_SPATIAL][index][1]]
    )
