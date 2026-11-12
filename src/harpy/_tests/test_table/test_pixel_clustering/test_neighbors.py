import importlib

import dask.array as da
import numpy as np
import pytest
from anndata import AnnData
from scipy import ndimage

import harpy.table.pixel_clustering._neighbors as neighbors_module
from harpy.image.segmentation._grid import add_grid_labels
from harpy.table.pixel_clustering._neighbors import _get_values_grid_most_frequent, spatial_pixel_neighbors
from harpy.utils._keys import _SPATIAL


@pytest.mark.skipif(
    not importlib.util.find_spec("squidpy"),
    reason="requires the squidpy library",
)
def test_spatial_pixel_neighbors(sdata):
    # note: hp.tb.spatial_pixel_neighbors would typically be run on a labels layer obtained via `hp.im.flowsom`.
    spatial_key = _SPATIAL
    adata = spatial_pixel_neighbors(
        sdata,
        labels_name="blobs_labels",
        key_added="cluster_id",
        size=50,
        mode="center",
        subset=None,
        spatial_key=spatial_key,
    )

    assert isinstance(adata, AnnData)
    # sanity check to see if we sampled all. Will evidentily fail if size is too large (e.g. size==100)
    assert np.array_equal(
        np.array(adata.obs["cluster_id"].cat.categories), da.unique(sdata["blobs_labels"].data).compute()
    )

    index = 2
    assert (
        adata.obs["cluster_id"][index]
        == sdata["blobs_labels"].data.compute()[adata.obsm["spatial"][index][0], adata.obsm[spatial_key][index][1]]
    )

    # divide in hexagon grid and take most frequent occuring cluster id in each bin.
    adata = spatial_pixel_neighbors(
        sdata,
        labels_name="blobs_labels",
        key_added="cluster_id",
        size=50,
        mode="most_frequent",
        grid_type="hexagon",
        subset=None,
        spatial_key=_SPATIAL,
    )

    assert isinstance(adata, AnnData)
    assert adata.shape[0] == 143
    assert adata.obs.iloc[25].item() == 31


def test_grid_most_frequent_coordinates_are_grid_centers(sdata, monkeypatch):
    # runs without squidpy: checks the centers of the grid cells used by mode="most_frequent"
    grids = []

    def record_grid(sdata, **kwargs):
        sdata = add_grid_labels(sdata, **kwargs)
        grids.append(sdata[kwargs["output_labels_name"]].data)
        return sdata

    # the function removes its grid before returning, so keep it when it is made
    monkeypatch.setattr(neighbors_module, "add_grid_labels", record_grid)
    values, coordinates = _get_values_grid_most_frequent(sdata, labels_name="blobs_labels", size=50)

    (grid,) = grids
    grid = grid.compute()
    index = np.unique(grid)
    index = index[index != 0]
    # coordinates and ndimage centers are both (y, x)
    expected = np.array(ndimage.center_of_mass(np.ones(grid.shape), grid, index))

    assert values.shape == (index.size, 1)
    np.testing.assert_allclose(coordinates, expected, rtol=1e-12)
