from copy import deepcopy
from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import harpy.qc as qc
from harpy._feature_panels import _make_feature_panel
from harpy._tests.test_table.test_pl.test_histogram import _table
from harpy.qc import (
    FeaturePointsSummary,
    PointsSummary,
    PointsSummaryMetadata,
    spatial_bin_histogram,
    spatial_bin_histogram_by_feature,
)
from harpy.qc.points._spatial_bin_summary import _summarize_feature_bins, _summarize_spatial_bins
from harpy.table.pl import histogram

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
        per_feature=pd.DataFrame(),
        per_class=pd.DataFrame(),
        spatial_counts=grid,
        spatial_bins=_summarize_spatial_bins(grid, metadata=metadata, n_features={"Control": 5, "Reference": 2}),
    )


def _feature_summary(counts):
    original = _summary(counts)
    grid = original.spatial_counts.rename({"feature_class": "feature"}).assign_coords(feature=["EPCAM", "VIM"])
    panel = _make_feature_panel(
        feature_key="gene",
        feature_class_key="code_class",
        features_by_class={"Endogenous": ("EPCAM", "VIM")},
    )
    return FeaturePointsSummary(
        metadata=original.metadata,
        per_feature=pd.DataFrame(),
        spatial_counts=grid,
        spatial_bins=_summarize_feature_bins(
            grid,
            metadata=original.metadata,
            panel=panel,
            feature_classes=("Endogenous",),
            retained_bin_mask=original.retained_bin_mask,
        ),
        retained_bin_mask=original.retained_bin_mask,
    )


def _heights(ax):
    return np.array([patch.get_height() for patch in ax.patches])


@pytest.mark.parametrize("source", ["spatial", "feature", "table"])
def test_histogram_defaults_to_borderless_filled_step_with_kde(source):
    if source == "spatial":
        ax = spatial_bin_histogram(_summary([0, 1, 1, 2]), feature_class="Control", bins=3, show_median=False)
    elif source == "feature":
        ax = spatial_bin_histogram_by_feature(
            _feature_summary([0, 1, 1, 2]), feature="EPCAM", bins=3, show_median=False
        )
    else:
        ax = histogram(_table().tables["table"], "metric", dataframe="obs", bins=3, show_median=False)
    assert not ax.patches
    assert len(ax.collections) == 1
    bars = ax.collections[0]
    assert bars.get_facecolor()[0, 3] == 0.5
    assert bars.get_edgecolor().size == 0
    np.testing.assert_array_equal(bars.get_linewidths(), [0])
    assert bars.get_paths()[0].vertices[:, 1].max() > 0
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


@pytest.mark.parametrize("earlier, top", [("percentage_histogram", 50), ("scatter", 80)])
def test_a_percentage_histogram_keeps_the_limits_of_earlier_axes_contents(earlier, top):
    options = {"stat": "percent", "kde": False}
    _, ax = plt.subplots()
    if earlier == "percentage_histogram":
        # Its filled area cached its bounds in counts, a peak of 200, before it was scaled to 50%.
        spatial_bin_histogram(
            _summary([0, 1, 1, 2] * 100),
            feature_class="Control",
            bins=3,
            ax=ax,
            histplot_kwargs=options,
            show_median=False,
        )
    else:
        # Scatter points are placed by offsets, so their bounds come from their data limits.
        ax.scatter([1], [top])
    spatial_bin_histogram(
        _summary([0, 1, 1, 2] * 50), feature_class="Control", bins=3, ax=ax, histplot_kwargs=options, show_median=False
    )
    # Its peak is 50% as well, so the upper limit follows the earlier contents, with a margin.
    assert top < ax.get_ylim()[1] < 1.2 * top


@pytest.mark.parametrize("counts", [[0, 1, 3], [0, 0, 0], [4], []])
def test_bin_summary_sample_sd_includes_retained_zeros(counts):
    row = _summary(counts).spatial_bins.per_class.iloc[0]
    expected = np.std(counts, ddof=1) if len(counts) > 1 else np.nan
    assert row.std_points_per_bin == pytest.approx(expected, nan_ok=True)
    assert row.n_retained_bins == len(counts)
    assert row.n_excluded_bins == 1


def test_spatial_histogram_annotations_from_plotted_values_and_style():
    """The median and SD of [0, 1, 2, 10] are 1.5 and 4.57, the stored statistics of the class."""
    summary = _summary([0, 1, 2, 10])
    statistics = summary.spatial_bins.per_class.iloc[0]
    assert statistics.median_points_per_bin == 1.5
    assert statistics.std_points_per_bin == pytest.approx(np.std([0, 1, 2, 10], ddof=1))
    before = deepcopy(summary)
    options = {"element": "bars", "bins": [-0.5, 0.5, 1.5, 2.5, 10.5], "kde": False, "color": "red"}
    original_options = deepcopy(options)
    _, ax = plt.subplots()
    result = spatial_bin_histogram(summary, feature_class="Control", ax=ax, histplot_kwargs=options)
    assert result is ax
    np.testing.assert_array_equal(_heights(ax), [1, 1, 1, 1])
    assert "Median: 1.50" in ax.texts[0].get_text()
    assert "SD    : 4.57" in ax.texts[0].get_text()
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1.5, 1.5])
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


def _with_panel_sizes(summary, panel_sizes):
    """The summary with a per_class holding each class's panel feature count, as summarize_points records it."""
    classes = list(panel_sizes)
    per_class = pd.DataFrame(
        {"feature_class": pd.Categorical(classes, categories=classes), "n_features": list(panel_sizes.values())}
    )
    return replace(summary, per_class=per_class)


def test_per_panel_feature_divides_counts_median_and_sd_by_the_panel_size():
    summary = _with_panel_sizes(_summary([0, 1, 2, 10]), {"Control": 5, "Reference": 2})
    before = deepcopy(summary)
    statistics = summary.spatial_bins.per_class.iloc[0]
    median, std = statistics.median_points_per_bin / 5, statistics.std_points_per_bin / 5
    # One bar per divided count: 0, 0.2, 0.4 and 2.0.
    options = {"element": "bars", "bins": [-0.1, 0.1, 0.3, 0.5, 2.1], "kde": False}
    _, ax = plt.subplots()
    spatial_bin_histogram(
        summary, feature_class="Control", normalization="per_panel_feature", ax=ax, histplot_kwargs=options
    )
    np.testing.assert_array_equal(_heights(ax), [1, 1, 1, 1])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [median, median])
    assert "Median: 0.30" in ax.texts[0].get_text()
    assert f"SD    : {std:.2f}" in ax.texts[0].get_text()
    assert ax.get_xlabel() == "Control points per panel feature per spatial bin"
    pd.testing.assert_frame_equal(summary.spatial_bins.per_bin, before.spatial_bins.per_bin)
    pd.testing.assert_frame_equal(summary.spatial_bins.per_class, before.spatial_bins.per_class)


def test_invalid_normalization_and_panel_sizes_raise():
    summary = _with_panel_sizes(_summary([0, 1, 2, 10]), {"Control": 0, "Reference": 2})
    with pytest.raises(ValueError, match="Unknown normalization"):
        spatial_bin_histogram(summary, feature_class="Control", normalization="per_area")
    with pytest.raises(ValueError, match="positive panel size"):
        spatial_bin_histogram(summary, feature_class="Control", normalization="per_panel_feature")


def test_pct_points_plots_the_class_share_of_each_bin_points():
    """Control has [0, 1, 2, 10] points and Reference 1 per bin: shares of 0, 50, 66.7 and 90.9%."""
    summary = _with_panel_sizes(_summary([0, 1, 2, 10]), {"Control": 5, "Reference": 2})
    shares = 100 * np.array([0, 1, 2, 10]) / np.array([1, 2, 3, 11])
    # One bar per share.
    options = {"element": "bars", "bins": [-1, 25, 60, 80, 101], "kde": False}
    _, ax = plt.subplots()
    spatial_bin_histogram(summary, feature_class="Control", measure="pct_points", ax=ax, histplot_kwargs=options)
    np.testing.assert_array_equal(_heights(ax), [1, 1, 1, 1])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [np.median(shares)] * 2)
    assert f"SD    : {np.std(shares, ddof=1):.2f}" in ax.texts[0].get_text()
    assert ax.get_xlabel() == "Control share of the spatial bin's points (%)"


def test_invalid_spatial_bin_measures_raise():
    summary = _with_panel_sizes(_summary([0, 1, 2, 10]), {"Control": 5, "Reference": 2})
    with pytest.raises(ValueError, match=r"not computed per spatial bin; choose from \['points', 'pct_points'\]"):
        spatial_bin_histogram(summary, feature_class="Control", measure="detected_features")
    with pytest.raises(ValueError, match="Unknown measure 'area'"):
        spatial_bin_histogram(summary, feature_class="Control", measure="area")
    with pytest.raises(ValueError, match="applies to measure 'points' only, found 'pct_points'"):
        spatial_bin_histogram(summary, feature_class="Control", measure="pct_points", normalization="per_panel_feature")

    # A summary of the Control class alone has no share column.
    grid = summary.spatial_counts.isel(feature_class=[0])
    bins = _summarize_spatial_bins(grid, metadata=summary.metadata, n_features={"Control": 5})
    one_class = _with_panel_sizes(replace(summary, spatial_counts=grid, spatial_bins=bins), {"Control": 5})
    with pytest.raises(ValueError, match=r"two classes or more; this summary has only \['Control'\]"):
        spatial_bin_histogram(one_class, feature_class="Control", measure="pct_points")


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
@pytest.mark.parametrize("feature_result", [False, True])
def test_constant_and_single_bin_histograms_skip_kde_but_keep_counts(counts, feature_result):
    summary = _feature_summary(counts) if feature_result else _summary(counts)
    selector = {"feature": "EPCAM"} if feature_result else {"feature_class": "Control"}
    plotter = spatial_bin_histogram_by_feature if feature_result else spatial_bin_histogram
    ax = plotter(summary, **selector, quantile_range=(0.1, 0.99))
    assert ax.collections[0].get_paths()[0].vertices[:, 1].max() == len(counts)
    assert len(ax.lines) == 1  # Only the median guide remains.
    assert ("N/A" in ax.texts[0].get_text()) == (len(counts) == 1)


@pytest.mark.parametrize("stat, expected", [("count", [1, 1, 1, 0]), ("percent", [25, 25, 25, 0])])
def test_feature_histogram_annotations_from_plotted_values_and_shared_bins_without_source_reads(
    monkeypatch, stat, expected
):
    """The median and SD of [0, 1, 2, 10] are 1.5 and 4.57, the stored statistics of the feature."""
    summary = _feature_summary([0, 1, 2, 10])
    statistics = summary.spatial_bins.per_feature.iloc[0]
    assert statistics.median_points_per_bin == 1.5
    assert statistics.std_points_per_bin == pytest.approx(np.std([0, 1, 2, 10], ddof=1))
    before = deepcopy(summary)

    def forbidden(*args, **kwargs):
        pytest.fail("Plotting must consume prepared summaries without recomputing them.")

    monkeypatch.setattr(qc, "summarize_points_by_feature", forbidden)
    monkeypatch.setattr(qc, "summarize_points", forbidden)
    options = {"element": "bars", "bins": [-0.5, 0.5, 1.5, 2.5, 10.5], "stat": stat, "kde": True}
    original_options = deepcopy(options)
    _, ax = plt.subplots()
    ax.set_title("Keep this title")
    result = spatial_bin_histogram_by_feature(summary, feature="EPCAM", range=(0, 2), ax=ax, histplot_kwargs=options)
    assert result is ax
    np.testing.assert_allclose(_heights(ax), expected)
    assert "Median: 1.50" in ax.texts[0].get_text()
    assert "SD    : 4.57" in ax.texts[0].get_text()
    assert ax.texts[1].get_text() == "Displayed: 3 / 4"
    assert ax.get_xlabel() == "EPCAM points per spatial bin"
    assert ax.get_ylabel() == ("Number of spatial bins" if stat == "count" else "Percentage of spatial bins (%)")
    assert ax.get_title() == "Keep this title"
    assert len(ax.lines) == 2  # KDE and median; zeros remain in the population.
    assert options == original_options
    pd.testing.assert_frame_equal(summary.spatial_bins.per_bin, before.spatial_bins.per_bin)
    pd.testing.assert_frame_equal(summary.spatial_bins.per_feature, before.spatial_bins.per_feature)
    xr.testing.assert_identical(summary.spatial_counts, before.spatial_counts)
    assert summary.metadata == before.metadata


@pytest.mark.parametrize(
    "plotter, summary_factory, selector, required_type",
    [
        (spatial_bin_histogram, _feature_summary, {"feature_class": "Control"}, "PointsSummary"),
        (spatial_bin_histogram_by_feature, _summary, {"feature": "EPCAM"}, "FeaturePointsSummary"),
    ],
)
def test_spatial_histogram_entry_points_reject_wrong_summary_types(plotter, summary_factory, selector, required_type):
    with pytest.raises(TypeError, match=f"summary must be a {required_type}"):
        plotter(summary_factory([0, 1]), **selector)


def test_empty_population_and_invalid_spatial_histogram_requests():
    summary = _summary([0, 1])
    empty = spatial_bin_histogram(_summary([]), feature_class="Control")
    assert not empty.patches and not empty.lines
    assert "No retained spatial bins" in empty.texts[0].get_text()
    with pytest.raises(ValueError, match="bin_size"):
        spatial_bin_histogram(replace(summary, spatial_bins=None), feature_class="Control")
    with pytest.raises(ValueError, match=r"summarize_points\(\) with bin_size"):
        spatial_bin_histogram_by_feature(replace(_feature_summary([0, 1]), spatial_bins=None), feature="EPCAM")
    with pytest.raises(ValueError, match=r"Unknown feature_class 'Missing'; choose from \['Control', 'Reference'\]"):
        spatial_bin_histogram(summary, feature_class="Missing")
    with pytest.raises(ValueError, match=r"Unknown feature 'Missing'; choose from \['EPCAM', 'VIM'\]"):
        spatial_bin_histogram_by_feature(_feature_summary([0, 1]), feature="Missing")
    with pytest.raises(ValueError, match="No values remaining"):
        spatial_bin_histogram(summary, feature_class="Control", range=(100, 200))
    with pytest.raises(ValueError, match="logarithmic"):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"log_scale": True})
    with pytest.raises(ValueError, match="weighting"):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"weights": [1, 2]})
    with pytest.raises(ValueError, match="stat="):
        spatial_bin_histogram(summary, feature_class="Control", histplot_kwargs={"stat": "density"})


@pytest.mark.parametrize("stat, expected", [("count", [1, 1]), ("percent", [25, 25])])
def test_table_histogram_uses_selected_nonmissing_population_and_matches_spatial_bins(stat, expected):
    sdata = _table()
    original = sdata.tables["table"].copy()
    kwargs = {
        "range": (0, 1),
        "histplot_kwargs": {"element": "bars", "stat": stat, "kde": False, "bins": [-0.5, 0.5, 1.5]},
    }
    table_ax = histogram(sdata.tables["table"], "metric", region="a", dataframe="obs", **kwargs)
    spatial_ax = spatial_bin_histogram(_summary([0, 1, 2, 10]), feature_class="Control", **kwargs)
    np.testing.assert_array_equal(_heights(table_ax), expected)
    np.testing.assert_array_equal(_heights(table_ax), _heights(spatial_ax))
    assert table_ax.get_ylabel() == ("Percentage of instances (%)" if stat == "percent" else "Number of instances")
    assert table_ax.texts[0].get_text() == spatial_ax.texts[0].get_text()
    pd.testing.assert_frame_equal(sdata.tables["table"].obs, original.obs)
    pd.testing.assert_frame_equal(sdata.tables["table"].var, original.var)
