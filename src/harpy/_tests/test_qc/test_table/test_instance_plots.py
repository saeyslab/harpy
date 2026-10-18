"""hp.qc.instance_histogram and hp.qc.instance_scatter on a table summary."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import harpy as hp
import harpy.qc._histogram as histogram_module
import harpy.qc._scatter as scatter_module
from harpy._tests.test_qc.test_table.test_summarize_table import _OBSM_KEYS, _sdata
from harpy.qc import summarize_table


@pytest.fixture
def summary():
    """The summary of x, negative and system_control: panel sizes 2, 4 and 5."""
    return summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _capture(monkeypatch, module, name):
    """Record the calls of a renderer, which still draws, and return them."""
    calls = []
    original = getattr(module, name)

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, capture)
    return calls


@pytest.mark.parametrize(
    ("feature_class", "measure", "column", "label"),
    [
        ("x", "points", "n_x_points", "x points per instance"),
        ("negative", "detected_features", "n_negative_detected_features", "negative detected features per instance"),
        (
            "system_control",
            "pct_points",
            "pct_system_control_points",
            "system_control share of the instance's points (%)",
        ),
    ],
)
def test_instance_histogram_plots_one_class_measure(monkeypatch, summary, feature_class, measure, column, label):
    calls = _capture(monkeypatch, histogram_module, "_plot_histogram")
    per_instance = summary.per_instance.copy()
    ax = hp.qc.instance_histogram(summary, feature_class=feature_class, measure=measure)

    (values,), options = calls[0]
    pd.testing.assert_series_equal(values, summary.per_instance[column])
    # The median and SD of the plotted values, with the conventions of the summaries.
    assert options["median"] == pytest.approx(np.median(values))
    assert options["std"] == pytest.approx(values.astype(float).std(ddof=1))
    assert (ax.get_xlabel(), ax.get_ylabel()) == (label, "Number of instances")
    # The summary is not modified.
    pd.testing.assert_frame_equal(summary.per_instance, per_instance)


def test_the_annotations_of_points_equal_the_class_statistics(monkeypatch, summary):
    calls = _capture(monkeypatch, histogram_module, "_plot_histogram")
    hp.qc.instance_histogram(summary, feature_class="negative")
    _, options = calls[0]
    row = summary.per_class.set_index("feature_class").loc["negative"]
    assert options["median"] == pytest.approx(row["median_points_per_instance"])
    assert options["std"] == pytest.approx(row["std_points_per_instance"])


def test_per_panel_feature_normalization_divides_the_points_and_their_annotations(monkeypatch, summary):
    calls = _capture(monkeypatch, histogram_module, "_plot_histogram")
    ax = hp.qc.instance_histogram(summary, feature_class="negative", normalization="per_panel_feature")

    (values,), options = calls[0]
    n_features = summary.panel_feature_counts["negative"]
    np.testing.assert_allclose(values, summary.per_instance["n_negative_points"] / n_features)
    row = summary.per_class.set_index("feature_class").loc["negative"]
    assert options["median"] == pytest.approx(row["median_points_per_instance"] / n_features)
    assert options["std"] == pytest.approx(row["std_points_per_instance"] / n_features)
    assert ax.get_xlabel() == "negative points per panel feature per instance"


def test_instance_histogram_counts_or_percentages_of_instances(summary):
    ax = hp.qc.instance_histogram(summary, feature_class="x", histplot_kwargs={"stat": "percent"})
    assert ax.get_ylabel() == "Percentage of instances (%)"


def test_instance_histogram_of_no_retained_instances_shows_an_empty_state(summary):
    empty = replace(summary, per_instance=summary.per_instance.iloc[0:0])
    ax = hp.qc.instance_histogram(empty, feature_class="x")
    assert [text.get_text() for text in ax.texts] == ["No retained instances"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"feature_class": "blank"},
            r"Unknown feature_class 'blank'; choose from \['x', 'negative', 'system_control'\]",
        ),
        ({"feature_class": "x", "measure": "counts"}, "Unknown measure 'counts'; choose from"),
        ({"feature_class": "x", "normalization": "per_area"}, "Unknown normalization"),
        (
            {"feature_class": "x", "measure": "detected_features", "normalization": "per_panel_feature"},
            "applies to measure 'points' only",
        ),
        ({"feature_class": "x", "histplot_kwargs": {"stat": "density"}}, "stat='count' or stat='percent'"),
    ],
)
def test_instance_histogram_rejects_invalid_selections(summary, kwargs, message):
    with pytest.raises(ValueError, match=message):
        hp.qc.instance_histogram(summary, **kwargs)


def test_the_share_of_points_needs_two_classes():
    one_class = summarize_table(_sdata(), "counts")
    with pytest.raises(ValueError, match="needs a summary of two classes or more"):
        hp.qc.instance_histogram(one_class, feature_class="x", measure="pct_points")


def test_instance_plots_take_a_table_summary(summary):
    with pytest.raises(TypeError, match="summary must be a TableSummary"):
        hp.qc.instance_histogram(summary.per_instance, feature_class="x")
    with pytest.raises(TypeError, match="summary must be a TableSummary"):
        hp.qc.instance_scatter(summary.per_instance, x=("x", "points"), y=("x", "points"))


def test_instance_scatter_plots_two_class_measures(monkeypatch, summary):
    calls = _capture(monkeypatch, scatter_module, "_plot_scatter")
    ax = hp.qc.instance_scatter(summary, x=("negative", "points"), y=("x", "detected_features"))

    (x_values, y_values), _ = calls[0]
    pd.testing.assert_series_equal(x_values, summary.per_instance["n_negative_points"])
    pd.testing.assert_series_equal(y_values, summary.per_instance["n_x_detected_features"])
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("negative points per instance", "x detected features per instance")


def test_instance_scatter_normalizes_each_axis_by_its_own_class(monkeypatch, summary):
    calls = _capture(monkeypatch, scatter_module, "_plot_scatter")
    hp.qc.instance_scatter(
        summary, x=("negative", "points"), y=("system_control", "points"), normalization="per_panel_feature"
    )

    (x_values, y_values), options = calls[0]
    np.testing.assert_allclose(x_values, summary.per_instance["n_negative_points"] / 4)
    np.testing.assert_allclose(y_values, summary.per_instance["n_system_control_points"] / 5)
    assert options["x_label"] == "negative points per panel feature per instance"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": ("negative",), "y": ("x", "points")}, r"x must be a pair \(feature_class, measure\)"),
        ({"x": ("negative", "points"), "y": ["x", "points"]}, r"y must be a pair \(feature_class, measure\)"),
        ({"x": ("blank", "points"), "y": ("x", "points")}, "Unknown feature_class 'blank'"),
        (
            {"x": ("negative", "points"), "y": ("x", "pct_points"), "normalization": "per_panel_feature"},
            "applies to measure 'points' only",
        ),
    ],
)
def test_instance_scatter_rejects_invalid_axes(summary, kwargs, message):
    with pytest.raises(ValueError, match=message):
        hp.qc.instance_scatter(summary, **kwargs)


def test_instance_scatter_of_no_retained_instances_raises(summary):
    empty = replace(summary, per_instance=summary.per_instance.iloc[0:0])
    with pytest.raises(ValueError, match="retains no instances"):
        hp.qc.instance_scatter(empty, x=("x", "points"), y=("negative", "points"))
