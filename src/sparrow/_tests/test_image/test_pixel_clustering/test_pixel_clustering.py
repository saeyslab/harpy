import importlib.util

import numpy as np
import pytest

from sparrow.utils._keys import _SPATIAL


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom(sdata_blobs):
    from sparrow.image.pixel_clustering._clustering import flowsom

    img_layer = "blobs_image"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    sdata_blobs, fsom, mapping = flowsom(
        sdata_blobs,
        img_layer=[img_layer],
        output_layer_clusters=[f"{img_layer}_clusters"],
        output_layer_metaclusters=[f"{img_layer}_metaclusters"],
        channels=channels,
        fraction=fraction,
        n_clusters=20,
        random_state=100,
        chunks=(1, 200, 200),
        overwrite=True,
    )

    assert f"{img_layer}_clusters" in sdata_blobs.labels
    assert f"{img_layer}_metaclusters" in sdata_blobs.labels
    assert fsom.model._is_fitted
    assert int(fraction * np.prod(sdata_blobs[img_layer].shape[1:])) == fsom.get_cell_data().shape[0]
    assert (fsom.get_cell_data().var.index == channels).all()

    # sanity check for consistency between flowsom object and sdata object.
    coord = fsom.get_cell_data().obsm[_SPATIAL][-2]
    assert (
        fsom.get_cell_data()[-2].to_df()["lineage_9"].values[0]
        == sdata_blobs[img_layer].sel(c=["lineage_9"]).data[0, coord[0], coord[1]].compute()
    )


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom_multi_c(sdata_multi_c):
    from sparrow.image.pixel_clustering._clustering import flowsom

    img_layer = "raw_image"
    fraction = 0.1

    sdata_multi_c, fsom, mapping = flowsom(
        sdata_multi_c,
        img_layer=[img_layer],
        output_layer_clusters=[f"{img_layer}_clusters"],
        output_layer_metaclusters=[f"{img_layer}_metaclusters"],
        fraction=fraction,
        n_clusters=20,
        random_state=100,
        chunks=(1, 200, 200),
        overwrite=True,
    )

    assert f"{img_layer}_clusters" in sdata_multi_c.labels
    assert f"{img_layer}_metaclusters" in sdata_multi_c.labels
    assert fsom.model._is_fitted
    assert int(fraction * np.prod(sdata_multi_c[img_layer].shape[1:])) == fsom.get_cell_data().shape[0]

    # sanity check for consistency between flowsom object and sdata object.
    coord = fsom.get_cell_data().obsm[_SPATIAL][-2]
    assert (
        fsom.get_cell_data()[-2].to_df()["0"].values[0]
        == sdata_multi_c[img_layer].sel(c=[0]).data[0, coord[0], coord[1]].compute()
    )


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom_algo(sdata_multi_c):
    from flowsom import FlowSOM

    adata = sdata_multi_c.tables["table_intensities"]
    fsom = FlowSOM(adata, cols_to_use=[0, 1], xdim=10, ydim=10, n_clusters=10, seed=10)

    assert len(fsom.get_cell_data()) == len(adata)
