from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel

import harpy.qc as qc
from harpy.qc import PointsSummary, PointsSummaryMetadata, spatial_bin_histogram, table_histogram, table_histograms
from harpy.qc._spatial_bin_summary import _summarize_spatial_bins

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _summary(counts):
    # Reference points retain every input bin, including Control zeros. The
    # terminal bin is empty across both classes and must never be plotted.
    values = np.array([[*counts, 0], [*np.ones(len(counts), dtype=int), 0]], dtype=np.uint64)[:, None, :]
    edges = tuple(float(value) for value in np.arange(len(counts) + 2) * 10)
    metadata = PointsSummaryMetadata(
        points_name="calls",
        sample_id="sample",
        feature_panel="panel",
        to_coordinate_system="sample_micron",
        crd=None,
        microns_per_unit=1.0,
        bin_size=10.0,
        x_edges=edges,
        y_edges=(0.0, 10.0),
    )
    grid = xr.DataArray(
        values,
        dims=("feature_class", "y", "x"),
        coords={"feature_class": ["Control", "Reference"], "y": [5.0], "x": np.asarray(edges[:-1]) + 5},
    )
    return PointsSummary(
        metadata=metadata,
        per_target=pd.DataFrame(),
        per_class=pd.DataFrame(),
        spatial_counts=grid,
        spatial_bins=_summarize_spatial_bins(grid, metadata=metadata),
    )


def _table():
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


@pytest.mark.parametrize("source", ["spatial", "table"])
def test_histogram_defaults_to_borderless_filled_step_with_kde(source):
    if source == "spatial":
        ax = spatial_bin_histogram(_summary([0, 1, 1, 2]), feature_class="Control", bins=3, show_median=False)
    else:
        ax = table_histogram(_table(), "table", column="metric", dataframe="obs", bins=3, show_median=False)
    assert not ax.patches
    assert len(ax.collections) == 1
    histogram = ax.collections[0]
    assert histogram.get_facecolor()[0, 3] == 0.5
    assert histogram.get_edgecolor().size == 0
    np.testing.assert_array_equal(histogram.get_linewidths(), [0])
    assert histogram.get_paths()[0].vertices[:, 1].max() > 0
    assert len(ax.lines) == 1  # KDE only; the median guide is disabled.
    assert ax.lines[0].get_linewidth() == 2


def test_histogram_preserves_explicit_style_and_nested_kde_options():
    options = {
        "element": "bars",
        "alpha": 0.8,
        "edgecolor": "red",
        "linewidth": 1.25,
        "line_kws": {"lw": 3, "linestyle": ":"},
        "kde_kws": {"bw_adjust": 0.8},
    }
    before = deepcopy(options)
    ax = spatial_bin_histogram(
        _summary([0, 1, 1, 2]), feature_class="Control", histplot_kwargs=options, show_median=False
    )
    assert ax.patches and not ax.collections
    assert ax.patches[0].get_facecolor()[3] == 0.8
    np.testing.assert_allclose(ax.patches[0].get_edgecolor()[:3], [1, 0, 0])
    assert ax.patches[0].get_linewidth() == 1.25
    assert ax.lines[0].get_linewidth() == 3
    assert ax.lines[0].get_linestyle() == ":"
    assert options == before


def test_filled_percentage_histogram_autoscales_to_percentage_not_raw_counts():
    ax = spatial_bin_histogram(
        _summary([0, 1, 1, 2] * 100),
        feature_class="Control",
        bins=3,
        histplot_kwargs={"stat": "percent", "kde": False},
        show_median=False,
    )
    # The peak represents 200 of 400 bins (50%), not the raw count of 200.
    assert ax.collections[0].get_paths()[0].vertices[:, 1].max() == 50
    assert 50 < ax.get_ylim()[1] < 100


@pytest.mark.parametrize("counts", [[0, 1, 3], [0, 0, 0], [4], []])
def test_bin_summary_sample_sd_includes_retained_zeros(counts):
    row = _summary(counts).spatial_bins.per_class.iloc[0]
    expected = np.std(counts, ddof=1) if len(counts) > 1 else np.nan
    assert row.std_points_per_bin == pytest.approx(expected, nan_ok=True)
    assert row.n_retained_bins == len(counts)
    assert row.n_excluded_bins == 1


def test_spatial_histogram_uses_prepared_population_annotations_and_style():
    summary = _summary([0, 1, 2, 10])
    summary.spatial_bins.per_class.loc[0, ["median_points_per_bin", "std_points_per_bin"]] = [3.0, 7.0]
    before = deepcopy(summary)
    options = {"element": "bars", "bins": [-0.5, 0.5, 1.5, 2.5, 10.5], "kde": False, "color": "red"}
    original_options = deepcopy(options)
    _, ax = plt.subplots()
    result = spatial_bin_histogram(summary, feature_class="Control", ax=ax, histplot_kwargs=options)
    assert result is ax
    np.testing.assert_array_equal(_heights(ax), [1, 1, 1, 1])
    assert "Median: 3" in ax.texts[0].get_text()
    assert "SD    : 7" in ax.texts[0].get_text()
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [3, 3])
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.get_xlabel() == "Control points per spatial bin"
    assert ax.get_ylabel() == "Number of spatial bins"
    assert ax.get_title() == ""
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert options == original_options
    pd.testing.assert_frame_equal(summary.spatial_bins.per_bin, before.spatial_bins.per_bin)
    pd.testing.assert_frame_equal(summary.spatial_bins.per_class, before.spatial_bins.per_class)
    xr.testing.assert_identical(summary.spatial_counts, before.spatial_counts)
    assert summary.metadata == before.metadata


@pytest.mark.parametrize("title, expected", [(None, "Existing title"), ("Custom title", "Custom title"), ("", "")])
def test_spatial_histogram_changes_existing_title_only_when_supplied(title, expected):
    _, ax = plt.subplots()
    ax.set_title("Existing title")
    spatial_bin_histogram(_summary([0, 1]), feature_class="Control", ax=ax, title=title, histplot_kwargs={"kde": False})
    assert ax.get_title() == expected


@pytest.mark.parametrize("stat, expected", [("count", [1, 1, 0, 0]), ("percent", [25, 25, 0, 0])])
@pytest.mark.parametrize("limits", [{"quantile_range": (0, 0.5)}, {"range": (0, 1), "quantile_range": (2, 3)}])
def test_display_filtering_preserves_full_population_denominator_and_annotations(stat, expected, limits):
    summary = _summary([0, 1, 2, 10])
    ax = spatial_bin_histogram(
        summary,
        feature_class="Control",
        **limits,
        histplot_kwargs={"element": "bars", "stat": stat, "kde": False, "bins": [-0.5, 0.5, 1.5, 2.5, 10.5]},
    )
    np.testing.assert_array_equal(_heights(ax), expected)
    assert "Median: 1.50" in ax.texts[0].get_text()
    assert ax.texts[1].get_text() == "Displayed: 2 / 4"
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1.5, 1.5])
    assert ax.get_ylabel() == ("Percentage of spatial bins (%)" if stat == "percent" else "Number of spatial bins")


@pytest.mark.parametrize(
    "element, fill", [("bars", True), ("bars", False), ("step", True), ("step", False), ("poly", True)]
)
def test_percentage_scales_histogram_and_kde_without_rescaling_existing_axes(element, fill):
    summary = _summary([0, 1, 1, 2, 3, 4, 5, 8])
    options = {"element": element, "fill": fill, "kde": True, "kde_kws": {"bw_adjust": 0.8}}
    before_options = deepcopy(options)
    _, (count_ax, percent_ax) = plt.subplots(1, 2)
    earlier = percent_ax.plot([0, 1], [4, 6])[0]
    spatial_bin_histogram(
        summary,
        feature_class="Control",
        ax=count_ax,
        range=(0, 4),
        bins=4,
        histplot_kwargs=options,
        show_median=False,
    )
    spatial_bin_histogram(
        summary,
        feature_class="Control",
        ax=percent_ax,
        range=(0, 4),
        bins=4,
        histplot_kwargs={**options, "stat": "percent"},
        show_median=False,
    )
    factor = 100 / 8
    np.testing.assert_allclose(_heights(percent_ax), _heights(count_ax) * factor)
    for count, percent in zip(count_ax.lines, percent_ax.lines[1:], strict=True):
        np.testing.assert_allclose(count.get_xdata(), percent.get_xdata())
        np.testing.assert_allclose(percent.get_ydata(), np.asarray(count.get_ydata()) * factor)
    for count, percent in zip(count_ax.collections, percent_ax.collections, strict=True):
        for original, scaled in zip(count.get_paths(), percent.get_paths(), strict=True):
            np.testing.assert_allclose(scaled.vertices[:, 0], original.vertices[:, 0])
            np.testing.assert_allclose(scaled.vertices[:, 1], original.vertices[:, 1] * factor)
    np.testing.assert_array_equal(earlier.get_ydata(), [4, 6])
    assert len(count_ax.lines) >= 1  # KDE exists, without a median guide.
    if not fill:
        for artist in [*count_ax.patches, *count_ax.lines]:
            assert artist.get_linewidth() > 0
    assert options == before_options


@pytest.mark.parametrize("counts", [[0, 0, 0], [2], [2, 2]])
def test_constant_and_single_bin_histograms_skip_kde_but_keep_counts(counts):
    ax = spatial_bin_histogram(_summary(counts), feature_class="Control", quantile_range=(0.1, 0.99))
    assert ax.collections[0].get_paths()[0].vertices[:, 1].max() == len(counts)
    assert len(ax.lines) == 1  # Only the median guide remains.
    assert ("N/A" in ax.texts[0].get_text()) == (len(counts) == 1)


def test_empty_population_and_invalid_spatial_histogram_requests():
    summary = _summary([0, 1])
    empty = spatial_bin_histogram(_summary([]), feature_class="Control")
    assert not empty.patches and not empty.lines
    assert "No retained spatial bins" in empty.texts[0].get_text()
    with pytest.raises(ValueError, match="bin_size"):
        spatial_bin_histogram(replace(summary, spatial_bins=None), feature_class="Control")
    with pytest.raises(ValueError, match="absent"):
        spatial_bin_histogram(summary, feature_class="Missing")
    with pytest.raises(ValueError, match="No values remaining"):
        spatial_bin_histogram(summary, feature_class="Control", range=(100, 200))
    with pytest.raises(ValueError, match="logarithmic"):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"log_scale": True})
    with pytest.raises(ValueError, match="weighting"):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"weights": [1, 2]})
    with pytest.raises(ValueError, match="stat="):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"stat": "density"})
    summary.spatial_bins.per_class.drop(columns="std_points_per_bin", inplace=True)
    with pytest.raises(ValueError, match="recompute"):
        spatial_bin_histogram(summary, feature_class="Control")


@pytest.mark.parametrize("stat, expected", [("count", [1, 1]), ("percent", [25, 25])])
def test_table_histogram_uses_selected_nonmissing_population_and_matches_spatial_bins(stat, expected):
    sdata = _table()
    original = sdata.tables["table"].copy()
    kwargs = {
        "range": (0, 1),
        "histplot_kwargs": {"element": "bars", "stat": stat, "kde": False, "bins": [-0.5, 0.5, 1.5]},
    }
    table_ax = table_histogram(sdata, "table", labels_name="a", column="metric", dataframe="obs", **kwargs)
    spatial_ax = spatial_bin_histogram(_summary([0, 1, 2, 10]), feature_class="Control", **kwargs)
    np.testing.assert_array_equal(_heights(table_ax), expected)
    np.testing.assert_array_equal(_heights(table_ax), _heights(spatial_ax))
    assert table_ax.get_ylabel() == ("Percentage of cells (%)" if stat == "percent" else "Number of cells")
    assert table_ax.texts[0].get_text() == spatial_ax.texts[0].get_text()
    pd.testing.assert_frame_equal(sdata.tables["table"].obs, original.obs)
    pd.testing.assert_frame_equal(sdata.tables["table"].var, original.var)


def test_table_histograms_use_each_metric_population_and_preserve_axes():
    sdata = _table()
    _, axes = plt.subplots(1, 2)
    result = table_histograms(
        sdata,
        "table",
        metrics=[("obs", "metric"), ("var", "metric")],
        ax=axes,
        range=(0, 1),
        histplot_kwargs={"element": "bars", "stat": "percent", "kde": False, "bins": [-0.5, 0.5, 1.5]},
    )
    assert result is axes
    np.testing.assert_allclose(_heights(axes[0]), [20, 20])  # Five non-null observations.
    np.testing.assert_allclose(_heights(axes[1]), [100 / 3, 100 / 3])
    assert axes[1].get_ylabel() == "Percentage of genes (%)"


@pytest.mark.parametrize(
    "alias, canonical", [("metric_histogram", "table_histogram"), ("metrics_histogram", "table_histograms")]
)
def test_deprecated_histogram_aliases_warn_once_for_public_access_and_import(monkeypatch, alias, canonical):
    from harpy.qc import _histogram as module

    messages = []
    monkeypatch.setattr(module, "log", SimpleNamespace(warning=messages.append))
    monkeypatch.setattr(module, "_WARNED_DEPRECATED_ATTRIBUTES", set())
    monkeypatch.delitem(vars(qc), alias, raising=False)
    function = getattr(qc, canonical)
    assert messages == []
    assert getattr(qc, alias) is function
    imported = getattr(__import__("harpy.qc", fromlist=[alias]), alias)
    assert imported is function
    assert messages == [f"`harpy.qc.{alias}` is deprecated. Import and use `harpy.qc.{canonical}` instead."]
