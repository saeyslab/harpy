import anndata as ad
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from spatialdata import SpatialData
from spatialdata.transformations import Identity

from harpy.image._image import add_labels
from harpy.plot._plot_density import plot_instance_density
from harpy.table._table import add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL


@pytest.fixture
def sdata_instances():
    sdata = SpatialData()
    labels = da.from_array(np.array([[1, 0], [0, 2]], dtype=np.uint16), chunks=(2, 2))
    sdata = add_labels(
        sdata,
        arr=labels,
        output_labels_name="labels_a",
        transformations={"global": Identity()},
    )
    sdata = add_labels(
        sdata,
        arr=labels,
        output_labels_name="labels_b",
        transformations={"global": Identity()},
    )

    adata = ad.AnnData(X=np.zeros((4, 0)))
    adata.obs[_INSTANCE_KEY] = [1, 2, 1, 2]
    adata.obs[_REGION_KEY] = pd.Categorical(["labels_a", "labels_a", "labels_b", "labels_b"])
    adata.obsm[_SPATIAL] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ]
    )
    adata.obsm["centroids"] = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [20.0, 20.0],
            [22.0, 22.0],
        ]
    )

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name="table_instances",
        region=["labels_a", "labels_b"],
    )
    return sdata


def test_plot_instance_density_returns_input_ax(sdata_instances, tmp_path):
    fig, ax = plt.subplots()
    try:
        result = plot_instance_density(
            sdata_instances,
            labels_name="labels_a",
            table_name="table_instances",
            spatial_key=_SPATIAL,
            bin_size=1,
            ax=ax,
        )

        assert result is ax
        assert isinstance(result, Axes)
        assert ax.images[0].get_array().sum() == 2
        fig.savefig(tmp_path / "plot_instance_density_returns_input_ax.png")
    finally:
        plt.close(fig)


def test_plot_instance_density_uses_all_observations_when_labels_name_is_none(sdata_instances):
    fig, ax = plt.subplots()
    try:
        result = plot_instance_density(
            sdata_instances,
            table_name="table_instances",
            spatial_key=_SPATIAL,
            bin_size=1,
            ax=ax,
        )

        assert result is ax
        assert isinstance(result, Axes)
        assert ax.images[0].get_array().sum() == 4
    finally:
        plt.close(fig)
