from collections import Counter
from copy import deepcopy

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Affine, Scale, set_transformation

from harpy import SpatialBounds
from harpy._tests.test_qc.test_summarize_points import PANEL, _frame, _sdata
from harpy.qc import (
    FeaturePointsSummary,
    FeatureSpatialBinSummary,
    summarize_points,
    summarize_points_by_feature,
)


def test_feature_planes_and_bin_statistics_share_one_population():
    """Use the same retained bins for every selected feature and its statistics.

    The default points are documented in ``_sdata()``. With bin_size=2 and
    crop (0, 10, 0, 2), the grid has five x bins and one y bin. The GeneB
    point at (5, 1) falls in bin 2: x in [4, 6), y in [0, 2).

    The class-level summary includes Endogenous and Negative, retaining
    bins 0–3. GeneB belongs to Endogenous, so it keeps bin 2 in that population.
    The subsequent feature request excludes GeneB; its requested features
    therefore all have zero counts in bin 2, as shown below::

        Feature     Bin 0  Bin 1  Bin 2  Bin 3  Bin 4
        GeneA           1      0      0      1      0
        ZeroGene        0      0      0      0      0
        NegA            1      1      0      1      0

    Every requested feature must still use bins 0–3 from the supplied summary,
    including its zero in bin 2. GeneA's counts are [1, 0, 0, 1], giving a
    mean of 2/4. Recomputing bin inclusion from only the requested features
    would wrongly drop bin 2 and change that mean to 2/3. This test protects
    against feature selection changing the population used for statistics.
    """
    sdata = _sdata()
    summary = summarize_points(
        sdata, "calls", feature_classes=["Endogenous", "Negative"], bin_size=2, crd=(0, 10, 0, 2)
    )
    result = summarize_points_by_feature(sdata, summary=summary, features=["NegA", "ZeroGene", "GeneA"])
    grid = result.spatial_counts
    assert grid.dims == ("feature", "y", "x")
    assert grid.feature.values.tolist() == ["GeneA", "ZeroGene", "NegA"]
    np.testing.assert_array_equal(grid.values[:, 0, :], [[1, 0, 0, 1, 0], [0] * 5, [1, 1, 0, 1, 0]])
    np.testing.assert_array_equal(result.retained_bin_mask, [[True, True, True, True, False]])

    bins = result.spatial_bins.per_bin
    expected_counts = {"GeneA": [1, 0, 0, 1], "ZeroGene": [0, 0, 0, 0], "NegA": [1, 1, 0, 1]}
    assert bins.feature.unique().tolist() == list(expected_counts)
    for feature, counts in expected_counts.items():
        rows = bins.loc[bins.feature == feature]
        assert rows.x_bin.tolist() == [0, 1, 2, 3]
        assert rows.n_points.tolist() == counts

    overview = result.spatial_bins.per_feature.set_index("feature")
    assert overview.index.tolist() == list(expected_counts)
    np.testing.assert_allclose(overview.mean_points_per_bin, [0.5, 0, 0.75])
    assert overview.n_retained_bins_without_feature.tolist() == [2, 4, 1]
    assert overview.loc["ZeroGene", "pct_retained_bins_without_feature"] == 100
    for column, value in {
        "n_total_bins": 5,
        "n_retained_bins": 4,
        "n_excluded_bins": 1,
        "pct_excluded_bins": 20,
    }.items():
        assert (overview[column] == value).all()


def test_feature_summary_returns_typed_outputs_and_shared_metadata():
    """Keep source and geometry context on the parent, not on nested outputs."""
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=2, crd=(0, 8, 0, 2), microns_per_unit=0.5)
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "NegA"])
    assert isinstance(result, FeaturePointsSummary)
    assert isinstance(result.spatial_bins, FeatureSpatialBinSummary)
    grid = result.spatial_counts
    bins = result.spatial_bins.per_bin
    assert grid.dtype == np.dtype("uint64")
    assert isinstance(bins.feature.dtype, pd.CategoricalDtype)
    assert bins.n_points.dtype == np.dtype("uint64")
    assert "feature_class" not in bins
    mask = result.retained_bin_mask
    assert mask.dims == ("y", "x")
    xr.testing.assert_identical(mask.x, grid.x)
    xr.testing.assert_identical(mask.y, grid.y)
    assert result.metadata == summary.metadata
    for frame in (result.per_feature, bins, result.spatial_bins.per_feature):
        assert not frame.attrs
        assert not {"points_name", "to_coordinate_system", "feature_panel", "sample_id"} & set(frame)
    assert not grid.attrs


@pytest.mark.parametrize("crd", [None, (0, 4, 0, 2)])
def test_unbinned_reference_reuses_totals_without_source_or_coordinate_work(monkeypatch, crd):
    from harpy.qc import _points_reduction as reductions
    from harpy.qc import _summarize_points_by_feature as module

    sdata = _sdata()
    summary = summarize_points(sdata, "calls", crd=crd)

    def forbidden(*args, **kwargs):
        pytest.fail("An unbinned reference already contains the cropped totals; no source compute is needed.")

    monkeypatch.setattr(reductions, "get_transformation", forbidden)
    monkeypatch.setattr(reductions, "_transformed_point_xy", forbidden)
    monkeypatch.setattr(reductions, "_point_bin_edges", forbidden)
    monkeypatch.setattr(module, "_spatial_count_array", forbidden)
    monkeypatch.setattr(module, "_summarize_feature_bins", forbidden)
    monkeypatch.setattr(dask, "compute", forbidden)
    result = summarize_points_by_feature(
        sdata, summary=summary, features=["NegA", "ZeroGene", "GeneA"], max_grid_bytes=1
    )
    pd.testing.assert_frame_equal(
        result.per_feature,
        pd.DataFrame(
            {
                "feature": ["GeneA", "ZeroGene", "NegA"],
                "feature_class": ["Endogenous", "Endogenous", "Negative"],
                "n_points": np.array([2, 0, 3] if crd is None else [1, 0, 2], dtype=np.uint64),
            }
        ),
    )
    assert result.spatial_counts is None
    assert result.spatial_bins is None
    assert result.retained_bin_mask is None
    assert result.metadata.bin_size is None
    assert result.metadata.x_edges is None and result.metadata.y_edges is None
    assert result.metadata.extent is None


@pytest.mark.parametrize("bin_size", [None, 1, 2])
def test_feature_totals_are_independent_of_binning(bin_size):
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=bin_size, crd=(0, 4, 0, 2))
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "ZeroGene", "NegA"])
    assert result.per_feature.n_points.tolist() == [1, 0, 2]
    if bin_size is not None:
        np.testing.assert_array_equal(result.spatial_counts.sum(dim=("y", "x")), result.per_feature.n_points)
    else:
        assert result.spatial_counts is None and result.spatial_bins is None
        assert result.metadata.crd == SpatialBounds(x=(0, 4), y=(0, 2))


@pytest.mark.parametrize("crd", [None, (0, 10, 0, 2)])
def test_feature_selection_changes_neither_reference_grid_nor_statistics(monkeypatch, crd):
    """A feature alone or alongside others uses the same edges and population.

    For an inferred reference extent, GeneB alone would formerly shift the
    grid origin to (5, 1). It must instead keep the full reference grid.
    """
    from harpy.qc import _points_reduction as module

    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=2, crd=crd)

    def forbidden(*args, **kwargs):
        pytest.fail("The reference supplies exact edges; do not infer extent or regenerate edges.")

    monkeypatch.setattr(module, "_transformed_point_xy", forbidden)
    monkeypatch.setattr(module, "_point_bin_edges", forbidden)
    solo = summarize_points_by_feature(sdata, summary=summary, features="GeneB")
    joint = summarize_points_by_feature(sdata, summary=summary, features=["GeneB", "GeneA", "NegA"])
    assert solo.metadata == joint.metadata == summary.metadata
    xr.testing.assert_identical(solo.retained_bin_mask, summary.retained_bin_mask)
    xr.testing.assert_identical(joint.retained_bin_mask, summary.retained_bin_mask)
    xr.testing.assert_identical(solo.spatial_counts.sel(feature="GeneB"), joint.spatial_counts.sel(feature="GeneB"))
    pd.testing.assert_series_equal(
        solo.spatial_bins.per_feature.iloc[0],
        joint.spatial_bins.per_feature.query("feature == 'GeneB'").iloc[0],
        check_names=False,
    )
    assert solo.spatial_bins.per_bin.n_points.tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize("bin_size", [None, 2])
@pytest.mark.parametrize("feature", ["ZeroGene", "GeneB"])
def test_entirely_undetected_features_keep_the_reference_population(bin_size, feature):
    # GeneB has detections elsewhere, but none inside the inherited crop.
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=bin_size, crd=(0, 4, 0, 2))
    result = summarize_points_by_feature(sdata, summary=summary, features=feature)
    assert result.per_feature.n_points.tolist() == [0]
    if bin_size is not None:
        assert not result.spatial_counts.values.any()
        assert result.spatial_bins.per_bin.n_points.tolist() == [0, 0]
        row = result.spatial_bins.per_feature.iloc[0]
        assert row.mean_points_per_bin == row.std_points_per_bin == 0
        assert row.n_retained_bins == row.n_retained_bins_without_feature == 2
        assert row.pct_retained_bins_without_feature == 100


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"features": None}, "one or more"),
        ({"features": []}, "one or more"),
        ({"features": ["GeneA", "GeneA"]}, "duplicate"),
        ({"features": ["Unknown"]}, "Unknown features"),
        ({"features": "NegA"}, "represented classes"),
        ({"max_grid_bytes": 0}, "max_grid_bytes"),
    ],
)
def test_invalid_feature_requests(kwargs, match):
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", feature_classes="Endogenous", bin_size=2)
    request = {"features": "GeneA", **kwargs}
    with pytest.raises(ValueError, match=match):
        summarize_points_by_feature(sdata, summary=summary, **request)


@pytest.mark.parametrize("mutation", ["missing_panel", "unknown_feature", "wrong_class", "null_feature"])
def test_panel_validation_runs_before_feature_and_crop_filtering(mutation):
    summary = summarize_points(_sdata(), "calls", bin_size=2, crd=(1, 4, 0, 2))
    frame = _frame()
    if mutation == "unknown_feature":
        frame.loc[0, "gene"] = "Unknown"
    elif mutation == "wrong_class":
        frame.loc[0, "code_class"] = "SystemControl"
    elif mutation == "null_feature":
        frame.loc[0, "gene"] = None
    sdata = _sdata(frame)
    if mutation == "missing_panel":
        del sdata.attrs["harpy"]["points"]["calls"]["feature_panel"]
    with pytest.raises(ValueError):
        summarize_points_by_feature(sdata, summary=summary, features="NegA")


def test_feature_bins_reuse_full_3d_affine_crop_and_actual_area_calibration():
    frame = _frame()
    frame["z"] = np.arange(len(frame), dtype=float)
    sdata = _sdata(frame)
    # Mix source z into output x and source x into output z before selecting.
    matrix = np.array([[1, 0.5, 0.25, 10], [0, 2, 0, -5], [0.5, 0, 1, 0], [0, 0, 0, 1]])
    set_transformation(
        sdata.points["calls"],
        Affine(matrix, input_axes=("x", "y", "z"), output_axes=("x", "y", "z")),
        to_coordinate_system="shifted",
    )
    summary = summarize_points(
        sdata,
        "calls",
        bin_size=3,
        to_coordinate_system="shifted",
        microns_per_unit=0.5,
        crd=SpatialBounds(x=(10, 20), y=(-5, 0), z=(1, 11)),
    )
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "NegA"])
    xyz = frame[["x", "y", "z"]].to_numpy() @ matrix[:-1, :-1].T + matrix[:-1, -1]
    keep = (
        (xyz[:, 0] >= 10) & (xyz[:, 0] < 20) & (xyz[:, 1] >= -5) & (xyz[:, 1] < 0) & (xyz[:, 2] >= 1) & (xyz[:, 2] < 11)
    )
    for feature in result.spatial_counts.feature.values:
        selected = keep & (frame.gene == feature).to_numpy()
        expected, _, _ = np.histogram2d(
            xyz[selected, 1], xyz[selected, 0], bins=(result.metadata.y_edges, result.metadata.x_edges)
        )
        np.testing.assert_array_equal(result.spatial_counts.sel(feature=feature), expected)
    bins = result.spatial_bins.per_bin
    np.testing.assert_allclose(bins.bin_area_um2, bins.bin_area * 0.25)
    assert result.metadata.x_edges[-2:] == (19, 20)
    assert result.metadata.y_edges[-2:] == (-2, 0)
    assert result.metadata.to_coordinate_system == "shifted"
    unbinned = summarize_points(
        sdata,
        "calls",
        to_coordinate_system="shifted",
        crd=result.metadata.crd,
    )
    totals = summarize_points_by_feature(sdata, summary=unbinned, features=["GeneA", "NegA"])
    pd.testing.assert_frame_equal(totals.per_feature, result.per_feature)


def test_feature_pixel_and_micron_grids_have_equivalent_counts_and_areas():
    sdata = _sdata()
    set_transformation(sdata.points["calls"], Scale([0.5, 0.5], axes=("x", "y")), to_coordinate_system="micron")
    pixel_summary = summarize_points(sdata, "calls", bin_size=2, microns_per_unit=0.5)
    micron_summary = summarize_points(sdata, "calls", bin_size=1, microns_per_unit=1, to_coordinate_system="micron")
    pixels = summarize_points_by_feature(sdata, summary=pixel_summary, features=["GeneA", "NegA"])
    microns = summarize_points_by_feature(sdata, summary=micron_summary, features=["GeneA", "NegA"])
    np.testing.assert_array_equal(pixels.spatial_counts.values, microns.spatial_counts.values)
    np.testing.assert_allclose(pixels.spatial_bins.per_bin.bin_area_um2, microns.spatial_bins.per_bin.bin_area_um2)
    pd.testing.assert_frame_equal(pixels.spatial_bins.per_feature, microns.spatial_bins.per_feature)


def test_grid_budget_counts_requested_features_not_classes(monkeypatch):
    from harpy.qc import _points_reduction as module

    sdata = _sdata()
    summary = summarize_points(sdata, "calls", feature_classes="Endogenous", bin_size=2, crd=(0, 8, 0, 2))
    # Two features from one class require two planes (4 bins × 8 bytes each).
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "ZeroGene"], max_grid_bytes=64)
    assert result.spatial_counts.nbytes == 64

    def forbidden(*args, **kwargs):
        pytest.fail("The grid budget must fail before reducing counts.")

    monkeypatch.setattr(module, "_summarize_point_partition", forbidden)
    with pytest.raises(ValueError, match="requires 64 bytes"):
        summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "ZeroGene"], max_grid_bytes=63)


@pytest.mark.parametrize("bin_size, crd", [(None, None), (2, None), (2, (0, 8, 0, 2))])
def test_multiple_features_share_source_reads_and_return_compact_counts(monkeypatch, bin_size, crd):
    """One count pass serves all requested features, validation and bin statistics.

    The feature summary reuses the bin boundaries saved in the supplied
    summary, even when no crop was specified. It must not perform another
    coordinate scan to determine the grid extent.

    When the supplied summary has ``bin_size=None``, feature totals are taken
    directly from ``summary.per_feature``. No source points need to be read again.

    With binning enabled, each of the 17 source partitions produces partial
    counts. The merge tree combines at most eight results per task:
    17 partition results -> 3 intermediate results -> 1 final result.
    The test checks that each source partition is read exactly once despite
    these multiple merge stages.

    The compute guard checks that the returned reduction contains count Series,
    not point dataframes. It does not measure peak memory or inspect intermediate
    task results.
    """
    reads = Counter()

    def read_partition(ordinal):
        reads[ordinal] += 1
        frame = _frame()
        frame.index = pd.RangeIndex(ordinal * len(frame), (ordinal + 1) * len(frame))
        return frame

    points = dd.from_delayed([dask.delayed(read_partition)(index) for index in range(17)], meta=_frame().iloc[:0])
    sdata = _sdata(points)
    with dask.config.set(scheduler="synchronous"):
        summary = summarize_points(sdata, "calls", bin_size=bin_size, crd=crd)
    reads.clear()
    original_compute = dask.compute

    def compact_compute(*args, **kwargs):
        computed = original_compute(*args, **kwargs)
        assert len(computed) == 1
        feature_counts, bin_counts = computed[0]
        assert isinstance(feature_counts, pd.Series)
        assert isinstance(bin_counts, pd.Series)
        return computed

    monkeypatch.setattr(dask, "compute", compact_compute)
    with dask.config.set(scheduler="synchronous"):
        result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "NegA"])
    assert reads == (Counter(dict.fromkeys(range(17), 1)) if bin_size is not None else Counter())
    assert result.per_feature.n_points.sum() == 17 * 5
    if bin_size is not None:
        assert result.spatial_counts.sum().item() == 17 * 5
        assert result.spatial_bins.per_bin.n_points.sum() == 17 * 5
        assert result.spatial_bins.per_feature.n_points.sum() == 17 * 5


def test_custom_source_keys_and_unused_categorical_features():
    panel = deepcopy(PANEL)
    panel.update(feature_key="marker", feature_class_key="kind")
    frame = _frame().rename(columns={"gene": "marker", "code_class": "kind"})
    frame["marker"] = pd.Categorical(frame.marker, categories=[*frame.marker.unique(), "UnusedUnknown"])
    points = dd.from_pandas(frame, npartitions=3)
    points["kind"] = points["kind"].cat.as_unknown()
    sdata = _sdata(points, panel=panel)
    summary = summarize_points(sdata, "calls", bin_size=2)
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "NegA"])
    assert result.spatial_counts.feature.values.tolist() == ["GeneA", "NegA"]
    assert result.spatial_counts.sum().item() == 5
    assert result.spatial_bins.per_feature.feature_class.tolist() == ["Endogenous", "Negative"]


@pytest.mark.parametrize("bin_size", [None, 2])
def test_feature_summary_does_not_mutate_backed_source_or_invoke_aggregation(tmp_path, monkeypatch, bin_size):
    store = tmp_path / "source.zarr"
    _sdata().write(store)
    sdata = read_zarr(store)
    summary = summarize_points(sdata, "calls", bin_size=bin_size)
    original_summary = deepcopy(summary)
    points = sdata.points["calls"].compute()
    attrs = deepcopy(sdata.attrs)
    files = {path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Feature summaries must not write, assign points to labels, or plot.")

    import matplotlib.pyplot as plt

    import harpy as hp

    monkeypatch.setattr(hp.tb, "aggregate_points", forbidden)
    monkeypatch.setattr(SpatialData, "write_element", forbidden)
    monkeypatch.setattr(SpatialData, "write_attrs", forbidden)
    monkeypatch.setattr(plt, "subplots", forbidden)
    result = summarize_points_by_feature(sdata, summary=summary, features=["GeneA", "NegA"])
    result.per_feature.loc[0, "n_points"] = 999
    if bin_size is not None:
        result.spatial_bins.per_bin.loc[0, "n_points"] = 999
        assert result.spatial_counts.sum().item() == 5
        result.spatial_counts.values[:] = 999
        result.retained_bin_mask.values[:] = False
        xr.testing.assert_identical(summary.spatial_counts, original_summary.spatial_counts)
    pd.testing.assert_frame_equal(summary.per_feature, original_summary.per_feature)
    pd.testing.assert_frame_equal(sdata.points["calls"].compute(), points)
    assert sdata.attrs == attrs
    assert {path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()} == files


@pytest.mark.parametrize("change", ["panel_key", "panel_features", "sample_id"])
def test_reference_source_metadata_must_still_match(change):
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=2)
    root = sdata.attrs["harpy"]
    if change == "panel_key":
        root["feature_panels"]["other"] = root["feature_panels"]["panel_a"]
        root["points"]["calls"]["feature_panel"] = "other"
    elif change == "sample_id":
        root["points"]["calls"]["sample_id"] = "other"
    else:
        root["feature_panels"]["panel_a"]["features_by_class"]["Endogenous"].append("ZzzGene")
    with pytest.raises(ValueError, match="recompute summarize_points"):
        summarize_points_by_feature(sdata, summary=summary, features="GeneA")


def test_reference_coordinates_must_match_its_edges():
    sdata = _sdata()
    summary = summarize_points(sdata, "calls", bin_size=2)
    # Equal dimensions are not enough: these bin locations were shifted.
    summary.spatial_counts.coords["x"] = summary.spatial_counts.x + 1
    with pytest.raises(ValueError, match="coordinates do not match"):
        summarize_points_by_feature(sdata, summary=summary, features="GeneA")
