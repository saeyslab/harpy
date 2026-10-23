import numpy as np
import pandas as pd
import pytest
import xarray as xr

from harpy import SpatialBounds
from harpy.qc import PointsSummaryMetadata, SpatialBinSummary
from harpy.qc.points._spatial_bin_summary import _summarize_spatial_bins

# Each class's number of panel features, as the parent summary's per_class holds it.
_N_FEATURES = {"Endogenous": 3, "Negative": 2}


def _grid_and_metadata(*, microns_per_unit=None):
    counts = np.array([[[2, 0, 2]], [[0, 0, 0]]], dtype=np.uint64)
    grid = xr.DataArray(
        counts,
        dims=("feature_class", "y", "x"),
        coords={
            "feature_class": ["Endogenous", "Negative"],
            "y": [1.0],
            "x": [1.0, 3.0, 4.5],
        },
    )
    metadata = PointsSummaryMetadata(
        x_edges=(0, 2, 4, 5),
        y_edges=(0, 2),
        bin_size=2,
        crd=SpatialBounds(x=(0, 5), y=(0, 2)),
        to_coordinate_system="pixels",
        microns_per_unit=microns_per_unit,
        points_name="calls",
        feature_panel="panel_a",
        sample_id="sample_a",
    )
    return grid, metadata


@pytest.mark.parametrize("microns_per_unit", [None, 0.5])
def test_bin_rows_use_actual_areas_and_class_statistics_use_the_same_population(microns_per_unit):
    """A narrower terminal bin changes its area, not inclusion or count statistics."""
    grid, metadata = _grid_and_metadata(microns_per_unit=microns_per_unit)
    result = _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES)
    assert isinstance(result, SpatialBinSummary)
    bins = result.per_bin
    expected_columns = ["feature_class", "y_bin", "x_bin", "x", "y", "bin_area", "n_points", "pct_points"]
    if microns_per_unit is not None:
        expected_columns.append("bin_area_um2")
    assert bins.columns.tolist() == expected_columns
    assert bins.feature_class.tolist() == ["Endogenous", "Endogenous", "Negative", "Negative"]
    assert isinstance(bins.feature_class.dtype, pd.CategoricalDtype)
    assert bins.n_points.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(bins.y_bin, [0, 0, 0, 0])
    np.testing.assert_array_equal(bins.x_bin, [0, 2, 0, 2])
    np.testing.assert_array_equal(bins.x, [1, 4.5, 1, 4.5])
    np.testing.assert_array_equal(bins.y, [1, 1, 1, 1])
    np.testing.assert_array_equal(bins.bin_area, [4, 2, 4, 2])
    np.testing.assert_array_equal(bins.n_points, [2, 2, 0, 0])

    overview = result.per_class.set_index("feature_class")
    for name in grid.feature_class.values:
        values = bins.loc[bins.feature_class == name, "n_points"].to_numpy()
        row = overview.loc[name]
        assert row.n_points == values.sum()
        assert row.mean_points_per_bin == values.mean()
        assert row.median_points_per_bin == np.median(values)
        assert row.p95_points_per_bin == np.percentile(values, 95)
    assert overview.loc["Negative", "n_retained_bins_without_class"] == 2
    assert overview.loc["Negative", "pct_retained_bins_without_class"] == 100
    population = {"n_total_bins": 3, "n_retained_bins": 2, "n_excluded_bins": 1, "pct_excluded_bins": 100 / 3}
    for key, value in population.items():
        np.testing.assert_allclose(overview[key], value)
    assert not {"n_zero_bins", "pct_zero_bins"} & set(overview.columns)
    for frame in (bins, result.per_class):
        assert not frame.attrs
        assert not any("per_100_um2" in name for name in frame.columns)

    if microns_per_unit is None:
        assert "bin_area_um2" not in bins
    else:
        np.testing.assert_array_equal(bins.bin_area_um2, [1, 0.5, 1, 0.5])


def test_empty_population_retains_classes_but_has_no_distribution():
    grid, metadata = _grid_and_metadata(microns_per_unit=1)
    grid.values[:] = 0
    result = _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES)
    assert result.per_bin.empty
    assert result.per_bin.n_points.dtype == np.dtype("uint64")
    assert result.per_bin.pct_points.dtype == np.dtype("float64")
    assert result.per_bin.feature_class.cat.categories.tolist() == ["Endogenous", "Negative"]
    assert result.per_class.feature_class.tolist() == ["Endogenous", "Negative"]
    assert not result.per_class.n_points.any()
    assert not result.per_class.n_retained_bins_without_class.any()
    distribution_columns = [
        name for name in result.per_class if "points_per_" in name or name == "pct_retained_bins_without_class"
    ]
    assert result.per_class[distribution_columns].isna().all().all()
    for key, value in {"n_total_bins": 3, "n_retained_bins": 0, "n_excluded_bins": 3, "pct_excluded_bins": 100}.items():
        assert (result.per_class[key] == value).all()


def test_bin_summary_is_an_independent_snapshot_and_does_not_mutate_grid():
    grid, metadata = _grid_and_metadata(microns_per_unit=1)
    original = grid.copy(deep=True)
    result = _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES)
    xr.testing.assert_identical(grid, original)
    result.per_bin.loc[0, "n_points"] = 999
    xr.testing.assert_identical(grid, original)
    assert result.per_class.loc[0, "n_points"] == 4
    assert metadata.x_edges == (0, 2, 4, 5)


def test_unrepresentable_physical_bin_area_raises():
    grid, metadata = _grid_and_metadata(microns_per_unit=1e308)
    with pytest.raises(ValueError, match="Calibrated spatial bin areas"):
        _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES)


def test_share_of_a_bins_points_is_per_class_and_sums_to_100_per_bin():
    """Endogenous [2, 0, 3] and Negative [2, 1, 0]: every bin is retained."""
    grid, metadata = _grid_and_metadata()
    grid.values[:] = np.array([[[2, 0, 3]], [[2, 1, 0]]], dtype=np.uint64)
    bins = _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES).per_bin
    np.testing.assert_array_equal(bins.pct_points, [50, 0, 100, 50, 100, 0])
    np.testing.assert_array_equal(bins.groupby(["y_bin", "x_bin"]).pct_points.sum(), [100, 100, 100])


def test_share_of_a_bins_points_is_absent_with_one_class():
    grid, metadata = _grid_and_metadata()
    result = _summarize_spatial_bins(grid.isel(feature_class=[0]), metadata=metadata, n_features=_N_FEATURES)
    assert "pct_points" not in result.per_bin


def test_per_class_ends_with_each_class_panel_size():
    grid, metadata = _grid_and_metadata()
    per_class = _summarize_spatial_bins(grid, metadata=metadata, n_features=_N_FEATURES).per_class
    assert per_class.columns[-1] == "n_features"
    assert per_class.n_features.tolist() == [3, 2]
