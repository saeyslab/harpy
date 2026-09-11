from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata import SpatialData, read_zarr
from spatialdata.models import PointsModel
from spatialdata.transformations import Affine, Identity, Scale, set_transformation

import harpy as hp
from harpy.qc._points_binning import _count_point_bins, _point_bin_edges

PANEL = {
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "classes": ["EmptyClass", "Endogenous", "Negative", "SystemControl"],
    "features_by_class": {
        "Endogenous": ["GeneA", "GeneB", "ZeroGene"],
        "Negative": ["NegA", "NegB", "NegC"],
        "SystemControl": ["SysA", "SysB"],
        "EmptyClass": ["NoCalls"],
    },
}


def _frame():
    return pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "y": np.tile([0.0, 1.0], 4),
            "gene": ["GeneA", "NegA", "NegA", "NegB", "SysA", "GeneB", "NegA", "GeneA"],
            "code_class": pd.Categorical(
                [
                    "Endogenous",
                    "Negative",
                    "Negative",
                    "Negative",
                    "SystemControl",
                    "Endogenous",
                    "Negative",
                    "Endogenous",
                ],
                categories=PANEL["classes"],
            ),
            "unused": "do not project this column",
        }
    )


def _sdata(frame=None, *, panel=None, npartitions=3):
    if frame is None:
        frame = _frame()
    points = frame if isinstance(frame, dd.DataFrame) else dd.from_pandas(frame, npartitions=npartitions)
    return SpatialData(
        points={"calls": PointsModel.parse(points, transformations={"global": Identity()})},
        attrs={
            "harpy": {
                "metadata_version": 1,
                "points": {"calls": {"feature_panel": "panel_a", "sample_id": "sample_a"}},
                "feature_panels": {"panel_a": deepcopy(PANEL if panel is None else panel)},
            }
        },
    )


def test_summary_returns_complete_panel_counts_and_class_statistics():
    sdata = _sdata()
    result = hp.qc.summarize_points(sdata, "calls", top_n=1)
    assert isinstance(result, hp.qc.PointsSummary)
    assert result.spatial_counts is None
    assert result.spatial_bins is None
    assert result.retained_bin_mask is None
    targets = result.per_target.set_index("feature")
    assert targets.index.tolist() == ["NoCalls", "GeneA", "GeneB", "ZeroGene", "NegA", "NegB", "NegC", "SysA", "SysB"]
    assert targets.n_points.tolist() == [0, 2, 1, 0, 3, 1, 0, 1, 0]
    assert targets.n_points.dtype == np.dtype("uint64")
    assert targets.loc["NegA", "within_class_fraction"] == 0.75
    assert targets.loc["NegC", "within_class_fraction"] == 0
    assert np.isnan(targets.loc["NoCalls", "within_class_fraction"])
    classes = result.per_class.set_index("feature_class")
    negative = classes.loc["Negative"]
    assert negative.n_features == 3
    assert negative.n_zero_features == 1
    assert negative.pct_zero_features == pytest.approx(100 / 3)
    assert negative.n_points == 4
    assert negative.mean_points_per_feature == pytest.approx(4 / 3)
    assert negative.median_points_per_feature == 1
    assert negative.p95_points_per_feature == pytest.approx(2.8)
    assert negative.pct_points_top_n_features == 75
    assert negative.top_n == 1
    assert negative.n_top_features == 1
    assert classes.loc["EmptyClass", "pct_zero_features"] == 100
    assert classes.loc["EmptyClass", "mean_points_per_feature"] == 0
    assert np.isnan(classes.loc["EmptyClass", "pct_points_top_n_features"])
    assert result.metadata.points_name == "calls"
    assert result.metadata.sample_id == "sample_a"
    assert result.metadata.feature_panel == "panel_a"
    assert result.metadata.crd is None
    assert result.metadata.extent is None
    assert result.per_target.n_points.sum() == result.per_class.n_points.sum() == 8


@pytest.mark.parametrize("bin_size", [None, 2])
def test_summary_metadata_matches_request(bin_size):
    """Return the requested context and panel sizes without duplicating metadata."""
    result = hp.qc.summarize_points(
        _sdata(), "calls", feature_classes="Negative", bin_size=bin_size, microns_per_unit=0.5, crd=(0, 8, 0, 2)
    )
    metadata = result.metadata
    assert isinstance(metadata, hp.qc.PointsSummaryMetadata)
    assert metadata.points_name == "calls"
    assert metadata.feature_panel == "panel_a"
    assert metadata.sample_id == "sample_a"
    assert metadata.to_coordinate_system == "global"
    assert metadata.crd == hp.SpatialBounds(x=(0, 8), y=(0, 2))
    assert metadata.microns_per_unit == 0.5
    assert metadata.bin_size == bin_size
    assert result.panel_feature_counts == {"Negative": 3}  # includes the undetected NegC
    frames = [result.per_target, result.per_class]
    if bin_size is None:
        assert metadata.x_edges is metadata.y_edges is metadata.extent is None
    else:
        assert metadata.x_edges == (0, 2, 4, 6, 8)
        assert metadata.y_edges == (0, 2)
        assert metadata.extent == (0, 8, 0, 2)
        assert not result.spatial_counts.attrs
        frames.extend([result.spatial_bins.per_bin, result.spatial_bins.per_class])
    for frame in frames:
        assert not frame.attrs
        assert not {"points_name", "sample_id", "feature_panel", "to_coordinate_system"} & set(frame.columns)


def test_retained_bin_mask_uses_any_selected_class_and_preserves_coordinates():
    """Any selected class can retain a bin; all-empty bins are excluded without modifying the grid."""
    result = hp.qc.summarize_points(
        _sdata(), "calls", feature_classes=["Endogenous", "Negative"], bin_size=2, crd=(0, 10, 0, 2)
    )
    grid_before = result.spatial_counts.copy(deep=True)
    mask = result.retained_bin_mask
    assert mask.dims == ("y", "x")
    assert mask.dtype == np.dtype(bool)
    # Bin 1 has only Negative points, bin 2 only Endogenous; bin 4 is empty.
    np.testing.assert_array_equal(mask, [[True, True, True, True, False]])
    xr.testing.assert_identical(mask.x, result.spatial_counts.x)
    xr.testing.assert_identical(mask.y, result.spatial_counts.y)
    xr.testing.assert_identical(result.spatial_counts, grid_before)


def test_summary_metadata_is_frozen_and_normalizes_sequences_to_tuples():
    original = hp.qc.summarize_points(_sdata(), "calls", bin_size=2).metadata
    edges = [0, 2, 3]
    crop = hp.SpatialBounds(x=(0, 3), y=(0, 2))
    metadata = replace(original, x_edges=edges, crd=crop)
    edges[0] = -999
    assert metadata.x_edges == (0, 2, 3)
    assert metadata.crd is crop
    assert metadata.extent == (0, 3, 0, 2)
    assert original.extent == (0, 8, 0, 2)
    with pytest.raises(FrozenInstanceError):
        metadata.points_name = "changed"


@pytest.mark.parametrize(
    "selection, expected",
    [
        (None, PANEL["classes"]),
        ("Negative", ["Negative"]),
        (["SystemControl", "Negative"], ["Negative", "SystemControl"]),
    ],
)
def test_class_selection_keeps_panel_order_and_shared_bins(selection, expected):
    result = hp.qc.summarize_points(_sdata(), "calls", feature_classes=selection, bin_size=2, crd=(0, 8, 0, 2))
    assert result.per_class.feature_class.tolist() == expected
    grid = result.spatial_counts
    assert grid.dims == ("feature_class", "y", "x")
    assert grid.feature_class.values.tolist() == expected
    assert result.metadata.x_edges == (0, 2, 4, 6, 8)
    assert result.metadata.y_edges == (0, 2)
    assert result.metadata.extent == (0, 8, 0, 2)
    assert result.metadata.to_coordinate_system == "global"
    np.testing.assert_array_equal(grid.x, [1, 3, 5, 7])
    for name in expected:
        total = result.per_class.set_index("feature_class").loc[name, "n_points"]
        assert grid.sel(feature_class=name).sum().item() == total
    if "Negative" in expected:
        np.testing.assert_array_equal(grid.sel(feature_class="Negative"), [[1, 2, 0, 1]])


@pytest.mark.parametrize("bin_size", [None, 2])
@pytest.mark.parametrize(
    "empty_source, kwargs",
    [
        (True, {}),
        (False, {"feature_classes": "EmptyClass"}),
        (False, {"feature_classes": "EmptyClass", "crd": (0, 8, 0, 2)}),
        (False, {"crd": (20, 30, 0, 2)}),
        (False, {"crd": hp.SpatialBounds(x=(0, 8), y=(0, 2), z=(3, 4))}),
    ],
    ids=["empty_source", "empty_class", "empty_class_with_crop", "empty_xy_crop", "empty_z_crop"],
)
def test_empty_selection_always_raises(bin_size, empty_source, kwargs):
    """Reject an entirely empty selection, with or without a grid or explicit crop."""
    frame = _frame()
    frame["z"] = 1.0
    if empty_source:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError, match="No points remain"):
        hp.qc.summarize_points(_sdata(frame), "calls", bin_size=bin_size, **kwargs)


def test_spatial_bin_population_uses_only_selected_classes():
    """Negative alone uses five positive bins; a joint call retains 95 class-specific zeros."""
    frame = pd.DataFrame(
        {
            "x": np.concatenate([np.arange(100) + 0.25, np.repeat(np.arange(5) + 0.25, 2)]),
            "y": 0.25,
            "gene": ["GeneA"] * 100 + ["NegA"] * 10,
            "code_class": pd.Categorical(["Endogenous"] * 100 + ["Negative"] * 10, categories=PANEL["classes"]),
        }
    )
    sdata = _sdata(frame, npartitions=7)
    all_classes = hp.qc.summarize_points(sdata, "calls", bin_size=1, crd=(0, 101, 0, 1))
    negative = hp.qc.summarize_points(sdata, "calls", feature_classes="Negative", bin_size=1, crd=(0, 101, 0, 1))
    np.testing.assert_array_equal(
        all_classes.spatial_counts.sel(feature_class="Negative"), negative.spatial_counts.sel(feature_class="Negative")
    )
    for result, retained, mean, median, n_zero in (
        (all_classes, 100, 0.1, 0, 95),
        (negative, 5, 2, 2, 0),
    ):
        mask = result.retained_bin_mask
        assert mask.dims == ("y", "x")
        np.testing.assert_array_equal(mask, [[True] * retained + [False] * (101 - retained)])
        row = result.spatial_bins.per_class.set_index("feature_class").loc["Negative"]
        assert row.n_points == 10
        assert row.mean_points_per_bin == mean
        assert row.median_points_per_bin == median
        assert row.n_retained_bins_without_class == n_zero
        assert row.pct_retained_bins_without_class == 100 * n_zero / retained
        for key, value in {
            "n_total_bins": 101,
            "n_retained_bins": retained,
            "n_excluded_bins": 101 - retained,
            "pct_excluded_bins": 100 * (101 - retained) / 101,
        }.items():
            np.testing.assert_allclose(result.spatial_bins.per_class[key], value)
        bins = result.spatial_bins.per_bin
        counts = bins.loc[bins.feature_class == "Negative", "n_points"].to_numpy()
        np.testing.assert_array_equal(counts, [2] * 5 + [0] * n_zero)
    assert negative.spatial_counts.feature_class.values.tolist() == ["Negative"]
    assert len(negative.spatial_bins.per_bin) == 5


def test_pixel_and_micron_bins_have_equivalent_counts_and_physical_areas():
    sdata = _sdata()
    set_transformation(sdata.points["calls"], Scale([0.25, 0.25], axes=("x", "y")), to_coordinate_system="micron")
    pixels = hp.qc.summarize_points(sdata, "calls", bin_size=2, crd=(0, 8, 0, 2), microns_per_unit=0.25)
    microns = hp.qc.summarize_points(
        sdata, "calls", bin_size=0.5, crd=(0, 2, 0, 0.5), to_coordinate_system="micron", microns_per_unit=1
    )
    raw = hp.qc.summarize_points(sdata, "calls", bin_size=2, crd=(0, 8, 0, 2))
    np.testing.assert_array_equal(pixels.spatial_counts.values, microns.spatial_counts.values)
    np.testing.assert_array_equal(pixels.spatial_counts.values, raw.spatial_counts.values)
    for name in ("n_points", "bin_area_um2"):
        np.testing.assert_allclose(pixels.spatial_bins.per_bin[name], microns.spatial_bins.per_bin[name])
    pd.testing.assert_frame_equal(pixels.spatial_bins.per_class, microns.spatial_bins.per_class)
    pd.testing.assert_frame_equal(pixels.spatial_bins.per_class, raw.spatial_bins.per_class)
    pd.testing.assert_frame_equal(pixels.per_target, raw.per_target)
    pd.testing.assert_frame_equal(pixels.per_class, raw.per_class)
    assert "bin_area_um2" not in raw.spatial_bins.per_bin
    assert raw.metadata.microns_per_unit is None


@pytest.mark.parametrize("calibration", [0, -1, np.nan, np.inf, True, "1", 1j])
def test_invalid_physical_calibration_fails_before_reduction(calibration, monkeypatch):
    from harpy.qc import _summarize_points as module

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid calibration should fail before reducing points.")

    monkeypatch.setattr(module, "_summarize_point_partition", forbidden)
    with pytest.raises(ValueError, match="microns_per_unit"):
        hp.qc.summarize_points(_sdata(), "calls", bin_size=2, microns_per_unit=calibration)


def test_calibration_does_not_enable_binning_or_add_feature_statistics():
    result = hp.qc.summarize_points(_sdata(), "calls", microns_per_unit=1)
    assert result.spatial_counts is None
    assert result.spatial_bins is None
    assert not any("um2" in column for column in result.per_target)
    assert not any("um2" in column for column in result.per_class)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("selection", [None, "Negative"])
def test_automatic_extent_uses_selected_transformed_coordinates(ndim, selection):
    """Infer bounds from selected classes after an affine, including source z for XYZ."""
    frame = _frame()
    if ndim == 2:
        axes = ("x", "y")
        matrix = np.array([[-1, 2, 10], [1, 1, -5], [0, 0, 1]], dtype=float)
    else:
        frame["z"] = [1, 1, 0, 1, 1, 0, 1, 1]
        axes = ("x", "y", "z")
        matrix = np.array([[-1, 2, 3, 10], [1, 1, -2, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    sdata = _sdata(frame)
    set_transformation(
        sdata.points["calls"],
        Affine(matrix, input_axes=axes, output_axes=axes),
        to_coordinate_system="world",
    )
    result = hp.qc.summarize_points(sdata, "calls", bin_size=2, feature_classes=selection, to_coordinate_system="world")
    if selection is not None:
        frame = frame.loc[frame.code_class == selection]
    xy = (frame[list(axes)].to_numpy() @ matrix[:-1, :-1].T + matrix[:-1, -1])[:, :2]
    grid = result.spatial_counts
    for column, axis in enumerate(("x", "y")):
        minimum, maximum = xy[:, column].min(), xy[:, column].max()
        count = int(np.floor((maximum - minimum) / 2)) + 1
        np.testing.assert_array_equal(getattr(result.metadata, f"{axis}_edges"), minimum + 2 * np.arange(count + 1))
    for name in grid.feature_class.values:
        selected_xy = xy[frame.code_class == name]
        expected, _, _ = np.histogram2d(
            selected_xy[:, 1], selected_xy[:, 0], bins=[result.metadata.y_edges, result.metadata.x_edges]
        )
        np.testing.assert_array_equal(grid.sel(feature_class=name), expected)
    assert grid.sum().item() == len(frame)


def test_top_n_does_not_change_raw_counts_or_other_statistics():
    sdata = _sdata()
    small = hp.qc.summarize_points(sdata, "calls", top_n=1)
    large = hp.qc.summarize_points(sdata, "calls", top_n=20)
    pd.testing.assert_frame_equal(small.per_target, large.per_target)
    stable = [
        "feature_class",
        "n_features",
        "n_zero_features",
        "n_points",
        "mean_points_per_feature",
        "median_points_per_feature",
        "p95_points_per_feature",
    ]
    pd.testing.assert_frame_equal(small.per_class[stable], large.per_class[stable])
    assert large.per_class.set_index("feature_class").loc["Negative", "pct_points_top_n_features"] == 100
    assert large.per_class.set_index("feature_class").loc["Negative", "n_top_features"] == 3


@pytest.mark.parametrize("bin_size", [None, 2.0])
@pytest.mark.parametrize("crd", [(7, 10, -5, 5), hp.SpatialBounds(x=(7, 10), y=(-5, 5))], ids=["tuple", "named_axes"])
def test_affine_xy_crop_of_xyz_points_applies_to_every_output(bin_size, crd):
    frame = _frame()
    frame["z"] = [1, 1, 0, 1, 1, 0, 1, 1]
    sdata = _sdata(frame)
    # Output x depends on source z; crop only after full XYZ transformation.
    matrix = np.array([[0, -2, 10, 0], [3, 0, 0, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    set_transformation(
        sdata.points["calls"],
        Affine(matrix, input_axes=("x", "y", "z"), output_axes=("x", "y", "z")),
        to_coordinate_system="world",
    )
    result = hp.qc.summarize_points(sdata, "calls", bin_size=bin_size, crd=crd, to_coordinate_system="world")
    # Source rows 1 and 3 remain: transformed XY=(8,-2) and (8,4).
    counts = result.per_target.set_index("feature")
    assert counts.loc["NegA", "n_points"] == 1
    assert counts.loc["NegB", "n_points"] == 1
    assert result.per_class.n_points.sum() == 2
    if bin_size is not None:
        grid = result.spatial_counts
        assert result.metadata.x_edges == (7, 9, 10)
        expected = np.zeros((5, 2), dtype=np.uint64)
        expected[1, 0] = expected[4, 0] = 1
        np.testing.assert_array_equal(grid.sel(feature_class="Negative"), expected)
    assert result.metadata.crd == hp.SpatialBounds(x=(7, 10), y=(-5, 5))


@pytest.mark.parametrize("bin_size", [None, 2.0])
def test_named_and_tuple_crops_produce_identical_summaries_and_metadata(bin_size):
    sdata = _sdata()
    before_attrs = deepcopy(sdata.attrs)
    crop = hp.SpatialBounds(x=(1, 6), y=(0, 1))
    named = hp.qc.summarize_points(sdata, "calls", bin_size=bin_size, crd=crop)
    positional = hp.qc.summarize_points(sdata, "calls", bin_size=bin_size, crd=(1, 6, 0, 1))
    pd.testing.assert_frame_equal(named.per_target, positional.per_target)
    pd.testing.assert_frame_equal(named.per_class, positional.per_class)
    assert named.metadata == positional.metadata
    assert named.metadata.crd is crop
    assert positional.metadata.crd.x == (1.0, 6.0)
    assert positional.metadata.crd.y == (0.0, 1.0)
    assert positional.metadata.crd.z is None
    if bin_size is not None:
        xr.testing.assert_identical(named.spatial_counts, positional.spatial_counts)
    assert sdata.attrs == before_attrs
    assert crop == hp.SpatialBounds(x=(1, 6), y=(0, 1))


@pytest.mark.parametrize("bin_size", [None, 2.0])
@pytest.mark.parametrize("use_tuple", [False, True])
def test_xyz_bounds_filter_transformed_z_before_xy_projection(bin_size, use_tuple):
    frame = _frame()
    frame["z"] = 1.0
    sdata = _sdata(frame)
    # All source z values are 1, but output z = 2*x + 3*z + 10.
    # The target interval [15, 21) selects source rows 1, 2, 3, including
    # the lower edge and excluding row 4 exactly on the upper edge.
    matrix = np.array([[1, 0, 0, 10], [0, 1, 0, -5], [2, 0, 3, 10], [0, 0, 0, 1]], dtype=float)
    set_transformation(
        sdata.points["calls"],
        Affine(matrix, input_axes=("x", "y", "z"), output_axes=("x", "y", "z")),
        to_coordinate_system="world",
    )
    bounds = hp.SpatialBounds(x=(10, 18), y=(-5, -3), z=(15, 21))
    before = sdata.points["calls"].compute()
    result = hp.qc.summarize_points(
        sdata,
        "calls",
        bin_size=bin_size,
        crd=bounds.as_tuple() if use_tuple else bounds,
        to_coordinate_system="world",
    )
    targets = result.per_target.set_index("feature")
    assert targets.loc["NegA", "n_points"] == 2
    assert targets.loc["NegB", "n_points"] == 1
    assert result.per_class.n_points.sum() == 3
    assert result.metadata.crd == bounds
    assert result.metadata.crd.z == (15.0, 21.0)
    if bin_size is not None:
        assert result.spatial_counts.dims == ("feature_class", "y", "x")
        np.testing.assert_array_equal(result.spatial_counts.sel(feature_class="Negative"), [[1, 2, 0, 0]])
        assert result.spatial_counts.sum().item() == 3
    unrestricted = hp.qc.summarize_points(
        sdata, "calls", crd=hp.SpatialBounds(x=bounds.x, y=bounds.y), to_coordinate_system="world"
    )
    assert unrestricted.per_target.n_points.sum() == 8
    pd.testing.assert_frame_equal(sdata.points["calls"].compute(), before)


def test_z_bounds_do_not_hide_invalid_source_panel_assignments():
    frame = _frame()
    frame["z"] = 0.0
    frame.loc[0, "gene"] = "Unknown"
    with pytest.raises(ValueError, match="absent from the panel"):
        hp.qc.summarize_points(_sdata(frame), "calls", crd=hp.SpatialBounds(x=(0, 8), y=(0, 2), z=(3, 4)))


def test_summarize_points_no_longer_accepts_source_z_plane():
    with pytest.raises(TypeError, match="z_plane"):
        hp.qc.summarize_points(_sdata(), "calls", z_plane=1)


def test_generic_coordinate_bins_have_half_open_edges_without_panel_or_raster():
    edges = _point_bin_edges((0, 450, 0, 200), 200, explicit_extent=True)
    xy = np.array([[0, 0], [199.9, 0], [200, 0], [250, 80], [400, 199]])
    result = _count_point_bins(xy, np.array(["group"] * 5), edges=edges)
    assert result.to_dict() == {("group", 0, 0): 2, ("group", 0, 1): 2, ("group", 0, 2): 1}


def test_fractional_bins_at_nonzero_origins_preserve_exact_edge_membership():
    exact_edges = _point_bin_edges((0, 3 * 0.1, 0, 0.1), 0.1, explicit_extent=True)
    assert len(exact_edges[0]) == 4  # no zero-width terminal bin from ceil rounding
    edges = _point_bin_edges((10, 10.6, -2, -1), 0.2, explicit_extent=True)
    edge = edges[0][1]
    xy = np.array([[np.nextafter(edge, -np.inf), -2], [edge, -2], [np.nextafter(edge, np.inf), -2]])
    counts = _count_point_bins(xy, np.array(["group"] * 3), edges=edges)
    assert counts.to_dict() == {("group", 0, 0): 1, ("group", 0, 1): 2}
    auto_edges = _point_bin_edges((10, 10.2, -2, -2), 0.2, explicit_extent=False)
    assert auto_edges[0][-1] > 10.2
    assert auto_edges[1][-1] > -2
    counts = _count_point_bins(np.array([[10.2, -2]]), np.array(["group"]), edges=auto_edges)
    assert counts.to_dict() == {("group", 0, 1): 1}


@pytest.mark.parametrize(
    "bounds, bin_size, explicit_extent, shape_xy",
    [
        ((0, 3 * 0.1, 0, 0.1), 0.1, True, (3, 1)),
        ((10, 10.2, -2, -2), 0.2, False, (2, 1)),
    ],
)
def test_grid_budget_matches_exact_bin_shape_before_edge_allocation(
    bounds, bin_size, explicit_extent, shape_xy, monkeypatch
):
    """Budget the actual bins, including clipping and floating-point edge corrections."""
    required = 3 * shape_xy[0] * shape_xy[1] * np.dtype(np.uint64).itemsize
    x_edges, y_edges = _point_bin_edges(
        bounds, bin_size, explicit_extent=explicit_extent, class_count=3, max_grid_bytes=required
    )
    assert (len(x_edges) - 1, len(y_edges) - 1) == shape_xy

    def forbidden(*args, **kwargs):
        pytest.fail("An over-budget request must fail before allocating bin edges.")

    monkeypatch.setattr(np, "arange", forbidden)
    with pytest.raises(ValueError, match=f"requires {required:,} bytes"):
        _point_bin_edges(bounds, bin_size, explicit_extent=explicit_extent, class_count=3, max_grid_bytes=required - 1)


def test_grid_budget_rejects_extreme_extent_without_overflow_or_edge_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An oversized extent must fail before allocating bin edges.")

    monkeypatch.setattr(np, "arange", forbidden)
    required = 4 * 10**12 * 10**12 * 8  # larger than a uint64 integer can represent
    with pytest.raises(ValueError, match=f"requires {required:,} bytes"):
        _point_bin_edges((0, 1e12, 0, 1e12), 1, explicit_extent=True, class_count=4, max_grid_bytes=1024)


@pytest.mark.parametrize("crd", [None, (0, 8, 0, 2)])
def test_grid_budget_fails_before_count_reduction(crd, monkeypatch):
    """Only automatic extent discovery may read points before rejecting the grid."""
    from harpy.qc import _summarize_points as module

    reads = []

    def read_frame():
        reads.append(1)
        return _frame()

    points = dd.from_delayed([dask.delayed(read_frame)()], meta=_frame().iloc[:0])
    sdata = _sdata(points)
    reads.clear()  # Exclude any index inspection by PointsModel.parse.

    def forbidden(*args, **kwargs):
        pytest.fail("An over-budget request must fail before reducing point counts.")

    monkeypatch.setattr(module, "_summarize_point_partition", forbidden)
    with dask.config.set(scheduler="synchronous"), pytest.raises(ValueError, match="requires 128 bytes"):
        hp.qc.summarize_points(sdata, "calls", bin_size=2, crd=crd, max_grid_bytes=127)
    assert len(reads) == (1 if crd is None else 0)


def test_default_grid_budget_can_be_disabled_without_allocating_large_grid(monkeypatch):
    """The same 3.2 GB grid fails by default but reaches construction with None.

    Intercept only final dense-grid construction to avoid a large allocation;
    real edge construction and point reductions run with the limit disabled.
    """
    from harpy.qc import _summarize_points as module

    class GridAllocationReached(Exception):
        pass

    def stop_before_allocation(*args, **kwargs):
        raise GridAllocationReached

    monkeypatch.setattr(module, "_spatial_count_array", stop_before_allocation)
    sdata = _sdata()
    with pytest.raises(ValueError, match="max_grid_bytes=1,073,741,824"):
        hp.qc.summarize_points(sdata, "calls", bin_size=1, crd=(0, 10000, 0, 10000))
    with pytest.raises(GridAllocationReached):
        hp.qc.summarize_points(sdata, "calls", bin_size=1, crd=(0, 10000, 0, 10000), max_grid_bytes=None)


def test_grid_budget_allows_exact_limit():
    result = hp.qc.summarize_points(_sdata(), "calls", bin_size=2, max_grid_bytes=128)
    assert result.spatial_counts.nbytes == 128
    assert result.spatial_counts.sum().item() == result.per_target.n_points.sum() == 8


def test_grid_budget_uses_selected_classes_and_is_not_enforced_without_binning():
    sdata = _sdata()
    selected = hp.qc.summarize_points(
        sdata, "calls", feature_classes="Negative", bin_size=2, crd=(0, 8, 0, 2), max_grid_bytes=32
    )
    assert selected.spatial_counts.nbytes == 32
    with pytest.raises(ValueError, match="requires 128 bytes"):
        hp.qc.summarize_points(sdata, "calls", bin_size=2, max_grid_bytes=32)
    result = hp.qc.summarize_points(sdata, "calls", max_grid_bytes=1)
    assert result.spatial_counts is None
    assert result.per_target.n_points.sum() == 8


@pytest.mark.parametrize("selection", [None, "Negative", ["Negative", "SystemControl"]])
def test_random_partitioned_counts_match_direct_numpy_histogram(selection):
    rng = np.random.default_rng(42)
    frame = pd.concat([_frame()] * 50, ignore_index=True)
    frame["x"] = rng.uniform(-5, 30, len(frame))
    frame["y"] = rng.uniform(-10, 12, len(frame))
    result = hp.qc.summarize_points(
        _sdata(frame, npartitions=19), "calls", feature_classes=selection, bin_size=2.7, crd=(-2, 25, -7, 8)
    )
    frame = frame.loc[frame.code_class.isin(result.spatial_counts.feature_class.values)]
    selected = frame.loc[(frame.x >= -2) & (frame.x < 25) & (frame.y >= -7) & (frame.y < 8)]
    edges = [result.metadata.y_edges, result.metadata.x_edges]
    total, _, _ = np.histogram2d(selected.y, selected.x, bins=edges)
    included = total > 0
    for name in result.spatial_counts.feature_class.values:
        rows = selected.loc[selected.code_class == name]
        expected, _, _ = np.histogram2d(rows.y, rows.x, bins=edges)
        np.testing.assert_array_equal(result.spatial_counts.sel(feature_class=name), expected)
        assert result.per_class.set_index("feature_class").loc[name, "n_points"] == len(rows)
        bins = result.spatial_bins.per_bin.loc[result.spatial_bins.per_bin.feature_class == name]
        np.testing.assert_array_equal(bins.n_points, expected[included])
        y_bin, x_bin = np.nonzero(included)
        np.testing.assert_array_equal(bins.y_bin, y_bin)
        np.testing.assert_array_equal(bins.x_bin, x_bin)
        overview = result.spatial_bins.per_class.set_index("feature_class").loc[name]
        assert overview.n_points == len(rows)
        assert overview.mean_points_per_bin == expected[included].mean()


@pytest.mark.parametrize("crd", [None, (0, 8, 0, 2)])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_spatial_summary_rejects_nonfinite_coordinates(crd, invalid):
    frame = _frame()
    frame.loc[0, "x"] = invalid
    with pytest.raises(ValueError, match="finite coordinates"):
        hp.qc.summarize_points(_sdata(frame), "calls", bin_size=2, crd=crd)


def test_categorical_feature_values_do_not_introduce_unused_source_targets():
    frame = _frame()
    frame["gene"] = pd.Categorical(frame.gene, categories=[*frame.gene.unique(), "NotInPanel"])
    result = hp.qc.summarize_points(_sdata(frame), "calls")
    assert "NotInPanel" not in result.per_target.feature.to_list()
    assert "ZeroGene" in result.per_target.feature.to_list()
    assert result.per_target.n_points.sum() == 8


def test_custom_source_keys_and_unknown_dask_categories():
    panel = deepcopy(PANEL)
    panel.update(feature_key="marker", feature_class_key="kind")
    frame = _frame().rename(columns={"gene": "marker", "code_class": "kind"})
    points = dd.from_pandas(frame, npartitions=3)
    points["kind"] = points["kind"].cat.as_unknown()
    sdata = _sdata(points, panel=panel)
    del sdata.attrs["harpy"]["points"]["calls"]["sample_id"]
    result = hp.qc.summarize_points(sdata, "calls")
    assert {"feature", "feature_class", "n_points"} <= set(result.per_target.columns)
    assert not {"marker", "kind", "sample_id"} & set(result.per_target.columns)
    assert result.metadata.sample_id is None
    assert result.per_target.n_points.sum() == 8


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("no_panel", "mapping"),
        ("wrong_version", "version"),
        ("missing_key", "panel column"),
        ("noncategorical", "categorical"),
        ("unknown_feature", "absent from the panel"),
        ("wrong_class", "expected"),
        ("null_class", "must not be null"),
        ("null_feature", "must not be null"),
        ("unknown_class", "panel classes"),
    ],
)
def test_panel_errors_are_not_hidden_by_class_or_spatial_filters(mutation, match):
    frame = _frame()
    if mutation == "unknown_feature":
        frame.loc[0, "gene"] = "Unknown"
    elif mutation == "wrong_class":
        frame.loc[0, "code_class"] = "SystemControl"
    elif mutation == "null_class":
        frame.loc[0, "code_class"] = None
    elif mutation == "null_feature":
        frame.loc[0, "gene"] = None
    elif mutation == "unknown_class":
        frame["code_class"] = frame.code_class.cat.add_categories("Unknown")
        frame.loc[0, "code_class"] = "Unknown"
    elif mutation == "noncategorical":
        frame["code_class"] = frame.code_class.astype(str)
    sdata = _sdata(frame)
    if mutation == "no_panel":
        sdata.attrs["harpy"].pop("feature_panels")
    elif mutation == "wrong_version":
        sdata.attrs["harpy"]["metadata_version"] = 2
    elif mutation == "missing_key":
        sdata.attrs["harpy"]["feature_panels"]["panel_a"]["feature_key"] = "missing"
    with pytest.raises(ValueError, match=match):
        hp.qc.summarize_points(sdata, "calls", feature_classes="Negative", crd=(1, 3, -1, 2))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"feature_classes": []}, "one or more"),
        ({"feature_classes": ["Negative", "Negative"]}, "duplicate"),
        ({"feature_classes": "Unknown"}, "Unknown feature classes"),
        ({"bin_size": 0}, "bin_size"),
        ({"bin_size": np.nan}, "bin_size"),
        ({"max_grid_bytes": 0}, "max_grid_bytes"),
        ({"max_grid_bytes": 1.5}, "max_grid_bytes"),
        ({"max_grid_bytes": True}, "max_grid_bytes"),
        ({"top_n": 0}, "top_n"),
        ({"top_n": 1.5}, "top_n"),
        ({"crd": (1, 0, 0, 2)}, "crd"),
        ({"crd": (0, 1, 2)}, "crd"),
        ({"crd": hp.SpatialBounds(x=(0, 8), y=(0, 2), z=(0, 1))}, "z bounds require 3D"),
        ({"crd": (0, 8, 0, 2, 0, 1)}, "z bounds require 3D"),
        ({"bin_size": 1, "to_coordinate_system": "absent"}, "coordinate system"),
    ],
)
def test_invalid_requests(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hp.qc.summarize_points(_sdata(), "calls", **kwargs)


def test_constant_xy_coordinates_include_all_source_z_values():
    frame = _frame()
    frame["z"] = [0, 0, 0, 0, 1, 0, 0, 0]
    frame["x"] = 4.0
    frame["y"] = 0.0
    sdata = _sdata(frame)
    result = hp.qc.summarize_points(sdata, "calls", bin_size=2)
    assert result.spatial_counts.shape == (4, 1, 1)
    assert result.metadata.extent == (4, 6, 0, 2)
    assert result.spatial_counts.sum().item() == len(frame)


@pytest.mark.parametrize("crd", [None, (0, 8, 0, 2)])
def test_joint_reductions_share_source_reads(monkeypatch, crd):
    """Protect shared input work across validation, target counts, bins, and classes.

    Seventeen delayed source partitions exercise multiple merge-tree levels.
    Automatic bounds add one class-filtered coordinate pass, shared by min and max;
    explicit bounds avoid it. The subsequent count reductions share one pass.
    Bin inclusion and the derived spatial-bin dataframes add no scans.
    Count source evaluations, not Dask task names or graph layout.
    """
    reads = Counter()

    def read_partition(ordinal):
        reads[ordinal] += 1
        frame = _frame()
        frame.index = pd.RangeIndex(ordinal * len(frame), (ordinal + 1) * len(frame))
        return frame

    points = dd.from_delayed([dask.delayed(read_partition)(ordinal) for ordinal in range(17)], meta=_frame().iloc[:0])
    sdata = _sdata(points)
    # PointsModel.parse may inspect source indices while preparing the fixture;
    # only source reads performed by summarize_points belong to this contract.
    reads.clear()
    from harpy.qc import _summarize_points as module

    original = module._summarize_point_partition

    def check_projection(partition, **kwargs):
        assert set(partition.columns) == {"x", "y", "gene", "code_class"}
        return original(partition, **kwargs)

    monkeypatch.setattr(module, "_summarize_point_partition", check_projection)
    original_xy = module._transformed_point_xy

    def check_coordinate_projection(partition, **kwargs):
        assert set(partition.columns) == {"x", "y"}
        return original_xy(partition, **kwargs)

    monkeypatch.setattr(module, "_transformed_point_xy", check_coordinate_projection)
    assert not reads
    with dask.config.set(scheduler="synchronous"):
        result = hp.qc.summarize_points(sdata, "calls", feature_classes="Negative", bin_size=2, crd=crd)
    assert reads == Counter(dict.fromkeys(range(17), 2 if crd is None else 1))
    assert result.per_target.n_points.sum() == result.spatial_counts.sum().item() == 17 * 4
    assert result.spatial_bins.per_bin.n_points.sum() == result.spatial_bins.per_class.n_points.sum() == 17 * 4


def test_nonspatial_summary_does_not_project_coordinates_or_require_registration(monkeypatch):
    sdata = _sdata()
    from harpy.qc import _summarize_points as module

    original = module._summarize_point_partition

    def check_projection(partition, **kwargs):
        assert set(partition.columns) == {"gene", "code_class"}
        return original(partition, **kwargs)

    monkeypatch.setattr(module, "_summarize_point_partition", check_projection)
    result = hp.qc.summarize_points(sdata, "calls", to_coordinate_system="unused")
    assert result.per_target.n_points.sum() == 8


def test_summary_does_not_mutate_points_metadata_or_backing_store(tmp_path, monkeypatch):
    store = tmp_path / "source.zarr"
    _sdata().write(store)
    sdata = read_zarr(store)
    before_attrs = deepcopy(sdata.attrs)
    before_points = sdata.points["calls"].compute()
    before_files = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Summary must not aggregate labels, write elements, or plot.")

    monkeypatch.setattr(hp.tb, "aggregate_points", forbidden)
    monkeypatch.setattr(SpatialData, "write_element", forbidden)
    monkeypatch.setattr(SpatialData, "write_attrs", forbidden)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "subplots", forbidden)
    result = hp.qc.summarize_points(sdata, "calls", bin_size=2)
    result.per_target.loc[0, "n_points"] = 999
    grid_before = result.spatial_counts.copy(deep=True)
    result.spatial_bins.per_bin.loc[0, "n_points"] = 999
    xr.testing.assert_identical(result.spatial_counts, grid_before)
    result.spatial_counts.values[:] = 999
    assert sdata.attrs == before_attrs
    pd.testing.assert_frame_equal(sdata.points["calls"].compute(), before_points)
    assert {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()} == before_files
    assert not sdata.images and not sdata.labels and not sdata.tables


def test_separate_results_do_not_pool_sample_identity_or_coordinate_frames():
    first, second = _sdata(), _sdata()
    second.attrs["harpy"]["points"]["calls"]["sample_id"] = "sample_b"
    set_transformation(
        second.points["calls"],
        Affine([[1, 0, 100], [0, 1, -30], [0, 0, 1]], input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="other",
    )
    a = hp.qc.summarize_points(first, "calls", bin_size=2)
    b = hp.qc.summarize_points(second, "calls", bin_size=2, to_coordinate_system="other")
    assert a.metadata.sample_id == "sample_a"
    assert b.metadata.sample_id == "sample_b"
    assert a.metadata.to_coordinate_system == "global"
    assert b.metadata.to_coordinate_system == "other"
    assert b.metadata.extent == (100, 108, -30, -28)
    np.testing.assert_array_equal(a.spatial_counts.values, b.spatial_counts.values)
