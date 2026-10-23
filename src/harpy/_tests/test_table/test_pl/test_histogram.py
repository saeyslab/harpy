import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel

from harpy.table.pl import histogram, histograms, scatter

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _table():
    """A table annotating the labels a and b: region a holds the metrics 0, 1, 2, 10 and a missing value, b holds 50."""
    obs = pd.DataFrame(
        {
            "metric": [0.0, 1.0, 2.0, 10.0, np.nan, 50.0],
            "region": pd.Categorical(["a"] * 5 + ["b"]),
            "instance": [1, 2, 3, 4, 5, 1],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame({"metric": [0.0, 1.0, 8.0]}, index=["g1", "g2", "g3"])
    table = TableModel.parse(
        AnnData(X=np.zeros((6, 3)), obs=obs, var=var),
        region=["a", "b"],
        region_key="region",
        instance_key="instance",
    )
    return SpatialData(
        tables={"table": table},
        labels={name: Labels2DModel.parse(np.array([[1, 2, 3, 4, 5]], dtype=np.uint32)) for name in ["a", "b"]},
    )


def _heights(ax):
    return np.array([patch.get_height() for patch in ax.patches])


def test_histograms(sdata_transcripts_no_backed, tmp_path):
    axes = histograms(
        sdata_transcripts_no_backed.tables["table_transcriptomics_preprocessed"],
        quantile_range=(0.1, 0.95),
    )

    axes = axes.ravel()

    assert axes.size == 6
    assert axes[0].get_xlabel() == "Total Counts"
    assert axes[1].get_xlabel() == "N Genes By Counts"
    axes[0].figure.savefig(tmp_path / "histograms.png")


def test_table_histograms_use_each_metric_population_and_preserve_axes():
    sdata = _table()
    _, axes = plt.subplots(1, 2)
    result = histograms(
        sdata.tables["table"],
        metrics=[("obs", "metric"), ("var", "metric")],
        ax=axes,
        range=(0, 1),
        histplot_kwargs={"element": "bars", "stat": "percent", "kde": False, "bins": [-0.5, 0.5, 1.5]},
    )
    assert result is axes
    np.testing.assert_allclose(_heights(axes[0]), [20, 20])  # Five non-null observations.
    np.testing.assert_allclose(_heights(axes[1]), [100 / 3, 100 / 3])
    assert axes[1].get_ylabel() == "Percentage of features (%)"


@pytest.mark.parametrize("region, expected_median", [(None, "2"), ("a", "1.50"), (["b"], "50")])
def test_table_plot_region_selects_the_rows_of_the_named_regions(region, expected_median):
    ax = histogram(_table().tables["table"], "metric", region=region, dataframe="obs", histplot_kwargs={"kde": False})
    assert f"Median: {expected_median}" in ax.texts[0].get_text()


def test_table_plots_reject_unknown_regions_var_subsets_and_region_without_annotation():
    adata = _table().tables["table"]
    with pytest.raises(ValueError, match=r"Unknown regions: \['c'\]; choose from \['a', 'b'\]"):
        histogram(adata, "metric", region="c", dataframe="obs")
    with pytest.raises(ValueError, match=r"A '\.var' column describes the whole table"):
        histogram(adata, "metric", region="a", dataframe="var")
    # Selecting every region keeps every row, so a '.var' column still describes the selection.
    histogram(adata, "metric", region=["a", "b"], dataframe="var")

    without_annotation = AnnData(obs=adata.obs[["metric", "instance"]].copy())
    with pytest.raises(ValueError, match="SpatialData annotation"):
        scatter(without_annotation, "metric", "instance", region="a")
    # region=None needs no annotation.
    scatter(without_annotation, "metric", "instance", show_regplot=False)
