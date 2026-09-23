from copy import deepcopy
from dataclasses import replace

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.colors import ListedColormap
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations import Affine

from harpy.plot import plot_points_density
from harpy.plot._plot_sdata import plot_sdata
from harpy.points._feature_panel import add_feature_panel
from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._summarize_points import PointsSummary, summarize_points
from harpy.qc.points._summarize_points_by_feature import FeaturePointsSummary


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def summary():
    """Three class planes, with a narrower final row/column and three retained bins.

    Endogenous: 4 0 0     Negative: 0 2 0     Empty: all zeros
                0 6 0               0 0 0

    The Negative-only bin remains visible as zero in an Endogenous map.
    Physical areas are [[1, 1, .5], [.5, .5, .25]] µm².
    """
    metadata = PointsSummaryMetadata(
        points_name="calls",
        sample_id="sample_a",
        feature_panel="panel_a",
        to_coordinate_system="sample_pixels",
        crd=None,
        microns_per_unit=0.5,
        bin_size=2,
        x_edges=(10, 12, 14, 15),
        y_edges=(20, 22, 23),
    )
    grid = xr.DataArray(
        np.array([[[4, 0, 0], [0, 6, 0]], [[0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.uint64),
        dims=("feature_class", "y", "x"),
        coords={"feature_class": ["Endogenous", "Negative", "Empty"], "y": [21, 22.5], "x": [11, 13, 14.5]},
    )
    return PointsSummary(
        metadata=metadata,
        per_feature=pd.DataFrame(
            {
                "feature": ["A", "B", "Zero", "Neg", "NoCalls"],
                "feature_class": ["Endogenous", "Endogenous", "Endogenous", "Negative", "Empty"],
                "n_points": [4, 6, 0, 2, 0],
            }
        ),
        per_class=pd.DataFrame(
            {
                "feature_class": ["Endogenous", "Negative", "Empty"],
                "n_features": [3, 1, 1],
                "n_points": [10, 2, 0],
            }
        ),
        spatial_counts=grid,
        spatial_bins=None,
    )


@pytest.fixture
def feature_summary(summary):
    grid = xr.DataArray(
        np.array([[[4, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 6, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.uint64),
        dims=("feature", "y", "x"),
        coords={"feature": ["A", "B", "Zero"], "y": [21, 22.5], "x": [11, 13, 14.5]},
    )
    return FeaturePointsSummary(
        metadata=summary.metadata,
        per_feature=summary.per_feature.iloc[:3].copy(),
        spatial_counts=grid,
        spatial_bins=None,
        retained_bin_mask=summary.retained_bin_mask,
    )


@pytest.mark.parametrize(
    ("normalization", "divisor", "units"),
    [
        (None, 1, "Points per bin"),
        ("per_area", np.array([[1, 1, 0.5], [0.5, 0.5, 0.25]]), "Points per µm²"),
        ("per_panel_feature", 3, "Points per panel feature per bin"),
        ("per_panel_feature_per_area", np.array([[3, 3, 1.5], [1.5, 1.5, 0.75]]), "Points per panel feature per µm²"),
    ],
)
def test_class_normalization_uses_actual_areas_and_complete_selected_panel(summary, normalization, divisor, units):
    ax = plot_points_density(summary, feature_class="Endogenous", normalization=normalization)
    values = ax.collections[0].get_array()
    expected = summary.spatial_counts.sel(feature_class="Endogenous").values / divisor
    np.testing.assert_allclose(values.compressed(), expected[summary.retained_bin_mask.values])
    np.testing.assert_array_equal(values.mask, ~summary.retained_bin_mask.values)
    assert ax.figure.axes[1].get_ylabel() == f"Endogenous — {units}"


def test_pooled_classes_divide_summed_counts_by_combined_panel_size(summary):
    ax = plot_points_density(summary, normalization="per_panel_feature")
    np.testing.assert_allclose(ax.collections[0].get_array().compressed(), [4 / 5, 2 / 5, 6 / 5])
    assert "Combined: Endogenous + Negative + Empty" in ax.figure.axes[1].get_ylabel()


@pytest.mark.parametrize("features", ["A", ["A", "B"], "Zero"])
def test_feature_selection_preserves_inherited_population(feature_summary, features):
    """Keep the parent summary's retained bins regardless of feature selection.

    The fixture retains three bins containing A, a Negative feature, and B,
    respectively. Plotting A, A+B, or the undetected feature Zero must preserve
    all three bins, including zeros for the displayed features. An all-zero
    feature must not be treated as an empty population.

    This guards against rebuilding the mask from the displayed features.
    Also check area normalization (B's 6 points / 0.5 µm² = 12 points/µm²)
    and the colorbar's selection label and units.
    """
    ax = plot_points_density(feature_summary, features=features, normalization="per_area")
    values = ax.collections[0].get_array()
    np.testing.assert_array_equal(values.mask, ~feature_summary.retained_bin_mask.values)
    expected = [4, 0, 12] if isinstance(features, list) else [4, 0, 0] if features == "A" else [0, 0, 0]
    np.testing.assert_allclose(values.compressed(), expected)
    assert not ax.texts  # An all-zero feature in a nonempty population is still a map.
    label = "Combined: A + B" if isinstance(features, list) else features
    assert ax.figure.axes[1].get_ylabel() == f"{label} — Points per µm²"


def test_only_feature_is_inferred(feature_summary):
    one = replace(feature_summary, spatial_counts=feature_summary.spatial_counts.sel(feature=["A"]))
    ax = plot_points_density(one, colorbar=False)
    np.testing.assert_array_equal(ax.collections[0].get_array().compressed(), [4, 0, 0])
    assert len(ax.figure.axes) == 1


@pytest.mark.parametrize("smoothing_sigma", [None, 2])
def test_all_zero_class_and_empty_population_are_distinct(summary, smoothing_sigma):
    ax = plot_points_density(summary, feature_class="Empty", colorbar=False, smoothing_sigma=smoothing_sigma)
    assert len(ax.collections) == 1
    assert not ax.collections[0].get_array().compressed().any()
    assert ax.collections[0].get_clim() == (0, 1)
    empty = replace(summary, spatial_counts=xr.zeros_like(summary.spatial_counts))
    empty_ax = plot_points_density(empty, smoothing_sigma=smoothing_sigma)
    assert not empty_ax.collections
    assert [text.get_text() for text in empty_ax.texts] == ["No retained spatial bins"]


@pytest.mark.parametrize("smoothing_sigma", [None, 2])
def test_geometry_uses_bin_edges_and_figure_sizing_does_not_change_values(summary, smoothing_sigma):
    with plt.rc_context({"figure.dpi": 83}):
        ax = plot_points_density(summary, colorbar=False, smoothing_sigma=smoothing_sigma)
    np.testing.assert_allclose(ax.figure.get_size_inches(), [8, 8])
    assert ax.figure.dpi == 83
    corners = ax.collections[0].get_coordinates()
    np.testing.assert_array_equal(corners[0, :, 0], [10, 12, 14, 15])
    np.testing.assert_array_equal(corners[:, 0, 1], [20, 22, 23])
    assert ax.get_xlim() == (10, 15)
    assert ax.get_ylim() == (23, 20)
    assert ax.get_aspect() == 1
    assert ax.get_title() == ""
    assert "sample_pixels" in ax.get_xlabel()

    fig, reused = plt.subplots(figsize=(4, 2), dpi=120)
    assert plot_points_density(summary, ax=reused, figsize=(99, 99), smoothing_sigma=smoothing_sigma) is reused
    np.testing.assert_array_equal(reused.collections[0].get_coordinates(), corners)
    np.testing.assert_array_equal(reused.collections[0].get_array(), ax.collections[0].get_array())
    np.testing.assert_allclose(fig.get_size_inches(), [4, 2])
    assert fig.dpi == 120
    custom = plot_points_density(summary, figsize=(3, 5), colorbar=False, smoothing_sigma=smoothing_sigma)
    np.testing.assert_allclose(custom.figure.get_size_inches(), [3, 5])


@pytest.mark.parametrize("smoothing_sigma", [None, 1])
def test_large_grid_keeps_bounded_canvas(summary, smoothing_sigma):
    grid = xr.DataArray(
        np.ones((1, 400, 1000), dtype=np.uint64),
        dims=("feature_class", "y", "x"),
        coords={"feature_class": ["Endogenous"], "y": np.arange(400) + 0.5, "x": np.arange(1000) + 0.5},
    )
    large = replace(
        summary,
        spatial_counts=grid,
        metadata=replace(
            summary.metadata,
            x_edges=tuple(range(1001)),
            y_edges=tuple(range(401)),
        ),
    )
    ax = plot_points_density(large, figsize=(3, 2), colorbar=False, smoothing_sigma=smoothing_sigma)
    np.testing.assert_allclose(ax.figure.get_size_inches(), [3, 2])
    assert ax.collections[0].get_array().shape == (400, 1000)
    np.testing.assert_allclose(ax.collections[0].get_array(), 1)


@pytest.mark.parametrize("inverted_y", [False, True])
@pytest.mark.parametrize("smoothing_sigma", [None, 2])
def test_overlay_preserves_view_and_zero_bins_are_not_transparent(summary, inverted_y, smoothing_sigma):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=90)
    image = ax.imshow(np.ones((2, 3)), extent=(5, 20, 15, 30), origin="lower", zorder=3)
    ax.set(xlim=(9, 14), ylim=(24, 19) if inverted_y else (19, 24), aspect=2, title="sample_a / calls / panel_a")
    before = (ax.get_xlim(), ax.get_ylim(), ax.get_aspect(), ax.get_title())
    cmap = ListedColormap(["blue", "yellow"])
    cmap.set_bad("red")  # The renderer must still make excluded bins transparent.
    returned = plot_points_density(
        summary,
        feature_class="Endogenous",
        ax=ax,
        alpha=0.5,
        cmap=cmap,
        vmin=0,
        vmax=8,
        smoothing_sigma=smoothing_sigma,
    )
    fig.canvas.draw()
    assert returned is ax
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_aspect(), ax.get_title()) == before
    mesh = ax.collections[-1]
    assert mesh.get_zorder() > image.get_zorder()
    assert mesh.get_clim() == (0, 8)
    alpha = mesh.get_facecolors()[:, 3].reshape(2, 3)
    np.testing.assert_allclose(alpha[summary.retained_bin_mask.values], 0.5)
    np.testing.assert_allclose(alpha[~summary.retained_bin_mask.values], 0)
    np.testing.assert_array_equal(cmap.get_bad(), [1, 0, 0, 1])


@pytest.mark.parametrize(
    ("kind", "options", "match"),
    [
        ("class", {"features": "A"}, "Use feature_class"),
        ("class", {"feature_class": "Absent"}, "absent"),
        ("class", {"normalization": "counts"}, "Unknown normalization"),
        ("feature", {"feature_class": "Endogenous"}, "Use features"),
        ("feature", {}, "Select features explicitly"),
        ("feature", {"features": "Absent"}, "absent"),
        ("feature", {"features": []}, "distinct names"),
        ("feature", {"features": ["A", "A"]}, "distinct names"),
        ("feature", {"features": "A", "normalization": "per_panel_feature"}, "only for PointsSummary"),
        ("feature", {"features": "A", "normalization": "per_panel_feature_per_area"}, "only for PointsSummary"),
    ],
)
def test_invalid_display_requests(summary, feature_summary, kind, options, match):
    with pytest.raises(ValueError, match=match):
        plot_points_density(summary if kind == "class" else feature_summary, **options)


@pytest.mark.parametrize("kind", ["class", "feature"])
def test_unbinned_summary_and_missing_calibration_give_actionable_errors(summary, feature_summary, kind):
    result = summary if kind == "class" else feature_summary
    changes = {
        "spatial_counts": None,
        "metadata": replace(result.metadata, bin_size=None, x_edges=None, y_edges=None),
    }
    if kind == "feature":
        changes["retained_bin_mask"] = None
    unbinned = replace(result, **changes)
    with pytest.raises(ValueError, match="summarize_points.*bin_size"):
        plot_points_density(unbinned)
    result = replace(result, metadata=replace(result.metadata, microns_per_unit=None))
    options = {} if kind == "class" else {"features": "A"}
    plot_points_density(result, **options)  # Raw display does not need calibration.
    with pytest.raises(ValueError, match="microns_per_unit"):
        plot_points_density(result, normalization="per_area", **options)


def test_source_data_and_bare_arrays_are_not_accepted(summary):
    for invalid in (summary.spatial_counts, SpatialData()):
        with pytest.raises(TypeError, match="PointsSummary"):
            plot_points_density(invalid)


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf, True, np.bool_(True), "2", [2]])
def test_invalid_smoothing_scale(summary, sigma):
    with pytest.raises(ValueError, match="smoothing_sigma.*positive, finite.*coordinate-system units"):
        plot_points_density(summary, smoothing_sigma=sigma)


@pytest.mark.parametrize(
    ("normalization", "panel_size", "area_normalized", "units"),
    [
        (None, 1, False, "points per bin"),
        ("per_area", 1, True, "points per µm²"),
        ("per_panel_feature", 5, False, "points per panel feature per bin"),
        ("per_panel_feature_per_area", 5, True, "points per panel feature per µm²"),
    ],
)
def test_smoothed_normalization_matches_direct_center_weights(
    summary, normalization, panel_size, area_normalized, units
):
    """Compare pooled classes with a direct 2D calculation on unequal-width bins.

    Retained centers are (11, 21), (13, 21), (13, 22.5), with counts 4, 2, 6
    and areas 1, 1, 0.5 µm². Pairwise distances, not array-index distances,
    define the Gaussian. The panel denominator includes undetected features.
    """
    centers = np.array([[11, 21], [13, 21], [13, 22.5]])
    distances = centers[:, None, :] - centers[None, :, :]
    weights = np.exp(-0.5 * (distances**2).sum(axis=2) / 2**2)
    support = [1, 1, 0.5] if area_normalized else [1, 1, 1]
    expected = (weights @ [4, 2, 6]) / (weights @ support) / panel_size
    ax = plot_points_density(summary, smoothing_sigma=2, normalization=normalization)
    values = ax.collections[0].get_array()
    np.testing.assert_allclose(values.compressed(), expected)
    np.testing.assert_array_equal(values.mask, ~summary.retained_bin_mask.values)
    assert ax.figure.axes[1].get_ylabel() == (
        f"Combined: Endogenous + Negative + Empty — Smoothed {units}\nσ = 2 (sample_pixels units)"
    )


@pytest.mark.parametrize("features", ["A", ["A", "B"], "Zero"])
def test_smoothed_features_include_inherited_zero_bins(feature_summary, features):
    """The Negative-only bin remains support for feature maps, including undetected features."""
    counts = [4, 0, 6] if isinstance(features, list) else [4, 0, 0] if features == "A" else [0, 0, 0]
    centers = np.array([[11, 21], [13, 21], [13, 22.5]])
    distances = centers[:, None, :] - centers[None, :, :]
    weights = np.exp(-0.5 * (distances**2).sum(axis=2) / 2**2)
    expected = (weights @ counts) / (weights @ [1, 1, 0.5])
    ax = plot_points_density(feature_summary, features=features, normalization="per_area", smoothing_sigma=2)
    values = ax.collections[0].get_array()
    np.testing.assert_allclose(values.compressed(), expected)
    np.testing.assert_array_equal(values.mask, ~feature_summary.retained_bin_mask.values)


def test_smoothing_preserves_constant_density_across_cropped_bins(summary):
    grid = xr.zeros_like(summary.spatial_counts)
    # Every bin has 8 points/µm², despite unequal areas in both dimensions.
    grid.loc[{"feature_class": "Endogenous"}] = [[8, 8, 4], [4, 4, 2]]
    result = replace(summary, spatial_counts=grid)
    ax = plot_points_density(result, normalization="per_area", smoothing_sigma=2, colorbar=False)
    np.testing.assert_allclose(ax.collections[0].get_array(), 8)


@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_smoothing_excludes_holes_and_outside_grid_but_keeps_retained_zeros(feature_summary, sigma):
    """At the left boundary use real neighbors only, never reflected copies.

    Counts:   10  0  0  2
    Retained:  Y  Y  N  Y

    With sigma=1, the leftmost result is (10 + 2*exp(-4.5)) /
    (1 + exp(-0.5) + exp(-4.5)). The retained zero contributes weight;
    the hole and outside-grid positions do not. With sigma=0.1, the
    four-sigma cutoff excludes every neighbor and leaves counts unchanged.
    """
    grid = xr.DataArray(
        np.array([[[10, 0, 0, 2]]], dtype=np.uint64),
        dims=("feature", "y", "x"),
        coords={"feature": ["A"], "y": [0.5], "x": [0.5, 1.5, 2.5, 3.5]},
    )
    mask = xr.DataArray([[True, True, False, True]], dims=("y", "x"), coords={"y": grid.y, "x": grid.x})
    result = replace(
        feature_summary,
        spatial_counts=grid,
        retained_bin_mask=mask,
        metadata=replace(feature_summary.metadata, bin_size=1, x_edges=(0, 1, 2, 3, 4), y_edges=(0, 1)),
    )
    ax = plot_points_density(result, smoothing_sigma=sigma, colorbar=False)
    values = ax.collections[0].get_array()
    np.testing.assert_array_equal(values.mask, ~mask.values)
    if sigma == 1:
        expected_left = (10 + 2 * np.exp(-4.5)) / (1 + np.exp(-0.5) + np.exp(-4.5))
        assert values[0, 0] == pytest.approx(expected_left)
        assert values[0, 1] > 0  # A measured zero may have a positive local estimate.
    else:
        np.testing.assert_allclose(values.compressed(), [10, 0, 2])


def test_smoothing_scale_uses_coordinate_units_not_calibration(summary):
    original = plot_points_density(summary, smoothing_sigma=2, colorbar=False).collections[0].get_array()
    uncalibrated = replace(summary, metadata=replace(summary.metadata, microns_per_unit=None))
    ax = plot_points_density(uncalibrated, smoothing_sigma=2, colorbar=False)
    np.testing.assert_allclose(ax.collections[0].get_array(), original)
    # Express the same geometry in a coordinate system with units ten times smaller.
    scaled = replace(
        uncalibrated,
        spatial_counts=summary.spatial_counts.assign_coords(
            x=summary.spatial_counts.x * 10, y=summary.spatial_counts.y * 10
        ),
        metadata=replace(
            uncalibrated.metadata,
            bin_size=20,
            x_edges=tuple(np.asarray(summary.metadata.x_edges) * 10),
            y_edges=tuple(np.asarray(summary.metadata.y_edges) * 10),
        ),
    )
    ax = plot_points_density(scaled, smoothing_sigma=20, colorbar=False)
    np.testing.assert_allclose(ax.collections[0].get_array(), original)


@pytest.mark.parametrize("smoothing_sigma", [None, 2])
def test_repeated_rendering_never_computes_points_or_mutates_summaries(
    summary, feature_summary, monkeypatch, smoothing_sigma
):
    """Detached results suffice: display choices must not invoke computation or panel resolution."""
    snapshots = [deepcopy(result) for result in (summary, feature_summary)]

    def forbidden(*args, **kwargs):
        pytest.fail("Rendering attempted to compute or resolve source data")

    monkeypatch.setattr(dask, "compute", forbidden)
    monkeypatch.setattr(dd.DataFrame, "compute", forbidden)
    monkeypatch.setattr("harpy.qc.points._summarize_points.summarize_points", forbidden)
    monkeypatch.setattr("harpy.qc.points._summarize_points._reduce_points_by_class", forbidden)
    monkeypatch.setattr("harpy.qc.points._summarize_points_by_feature.summarize_points_by_feature", forbidden)
    monkeypatch.setattr("harpy.qc.points._summarize_points_by_feature._resolve_points_feature_panel", forbidden)
    monkeypatch.setattr("harpy.qc.points._points_reduction._resolve_points_feature_panel", forbidden)
    plot_points_density(
        summary, feature_class="Negative", normalization="per_panel_feature_per_area", smoothing_sigma=smoothing_sigma
    )
    plot_points_density(summary, feature_class="Endogenous", smoothing_sigma=smoothing_sigma)
    plot_points_density(feature_summary, features="A", normalization="per_area", smoothing_sigma=smoothing_sigma)
    plot_points_density(feature_summary, features=["A", "B"], smoothing_sigma=smoothing_sigma)
    for result, before in zip((summary, feature_summary), snapshots, strict=True):
        assert result.metadata == before.metadata
        xr.testing.assert_identical(result.spatial_counts, before.spatial_counts)
        xr.testing.assert_identical(result.retained_bin_mask, before.retained_bin_mask)
        pd.testing.assert_frame_equal(result.per_feature, before.per_feature)
    pd.testing.assert_frame_equal(summary.per_class, snapshots[0].per_class)


def test_image_overlay_aligns_in_shared_system_without_transforming_grid_again(monkeypatch):
    """Align an image and density grid in their shared coordinate system.

    In their own intrinsic coordinate systems:

    - The image is 8 × 8 pixels, with outer boundaries x=0–8, y=0–8.
    - The points have coordinates (0.5, 0.5) and (1.5, 1.5).

    Each element has its own transformation to the shared "sample" system:

    - Image: (x, y) -> (0.5*x + 10, 0.5*y + 20).
      Its outer boundaries become x=10–14, y=20–24.
    - Points: (x, y) -> (2*x + 10, 2*y + 20).
      Their positions become (11, 21) and (13, 23).

    These transformed positions are directly comparable because they now
    use the same coordinate system. The original intrinsic coordinates
    belong to separate elements and cannot be compared directly.

    summarize_points() applies the points transformation before binning.
    With bin size 2, the resulting grid has counts [[1, 0], [0, 1]].
    plot_sdata() independently renders the image in the same system.

    The overlay must use the grid's already-transformed bin boundaries,
    without applying another transformation, and preserve the existing
    axis limits and aspect ratio. This guards against double transformation
    or plotting bin indices instead of spatial coordinates.
    """
    point_transform = Affine(
        np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]]), input_axes=("x", "y"), output_axes=("x", "y")
    )
    image_transform = Affine(
        np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1]]), input_axes=("x", "y"), output_axes=("x", "y")
    )
    image = Image2DModel.parse(
        np.ones((1, 8, 8), dtype=np.float32),
        dims=("c", "y", "x"),
        c_coords=["DAPI"],
        transformations={"sample": image_transform},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [0.5, 1.5], "y": [0.5, 1.5], "gene": ["A", "A"]}),
        transformations={"sample": point_transform},
    )
    sdata = SpatialData(images={"DAPI": image}, points={"calls": points})
    add_feature_panel(
        sdata, "calls", feature_key="gene", feature_class_key="code_class", features_by_class={"Expression": ["A"]}
    )
    summary = summarize_points(sdata, "calls", bin_size=2, crd=(10, 14, 20, 24), to_coordinate_system="sample")
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = plot_sdata(
        sdata, image_name="DAPI", channel="DAPI", to_coordinate_system="sample", show_kwargs={"colorbar": False}
    )
    image_artist = ax.images[0]
    # Map the image's local extent into the plotting coordinate system.
    left, right, bottom, top = image_artist.get_extent()
    to_data = image_artist.get_transform() - ax.transData
    image_corners = to_data.transform([[left, bottom], [right, top]])
    np.testing.assert_allclose(np.sort(image_corners, axis=0), [[10, 20], [14, 24]])
    before = ax.get_xlim(), ax.get_ylim(), ax.get_aspect()
    plot_points_density(summary, ax=ax, alpha=0.5, colorbar=False)
    mesh = ax.collections[-1]
    np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], [10, 12, 14])
    np.testing.assert_allclose(mesh.get_coordinates()[:, 0, 1], [20, 22, 24])
    np.testing.assert_array_equal(mesh.get_array().filled(0), [[1, 0], [0, 1]])
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_aspect()) == before
