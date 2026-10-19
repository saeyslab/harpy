import matplotlib
import matplotlib.pyplot as plt
import pytest

from harpy._tests.test_table.test_pl.test_histogram import _table
from harpy.table.pl import scatter

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_scatter(sdata_transcripts_no_backed, tmp_path):
    _, ax = plt.subplots()
    result = scatter(
        sdata_transcripts_no_backed.tables["table_transcriptomics_preprocessed"],
        column_x="shapeSize",
        column_y="total_counts",
        ax=ax,
    )

    assert result is ax
    assert ax.get_xlabel() == "Shapesize"
    assert ax.get_ylabel() == "Total Counts"
    ax.figure.savefig(tmp_path / "scatter.png")


def test_scatter_without_regplot(sdata_transcripts_no_backed, tmp_path):
    _, ax = plt.subplots()
    result = scatter(
        sdata_transcripts_no_backed.tables["table_transcriptomics_preprocessed"],
        column_x="shapeSize",
        column_y="total_counts",
        ax=ax,
        show_regplot=False,
    )

    assert result is ax
    assert len(ax.lines) == 0
    ax.figure.savefig(tmp_path / "scatter_without_regplot.png")


def test_table_scatter_plots_two_var_columns_but_not_columns_of_both_dataframes():
    adata = _table().tables["table"]
    adata.var["other"] = [1.0, 2.0, 3.0]
    ax = scatter(adata, "metric", "other", dataframe="var", show_regplot=False)
    assert ax.collections[0].get_array().sum() == 3  # One count per feature.
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Metric", "Other")
    with pytest.raises(ValueError, match="not both in 'adata.obs', nor both in 'adata.var'"):
        scatter(adata, "instance", "other", show_regplot=False)
    # metric is in both dataframes, but only .obs also holds instance.
    assert scatter(adata, "metric", "instance", show_regplot=False).get_ylabel() == "Instance"
    with pytest.raises(ValueError, match=r"A '\.var' column describes the whole table"):
        scatter(adata, "metric", "other", dataframe="var", region="a")
