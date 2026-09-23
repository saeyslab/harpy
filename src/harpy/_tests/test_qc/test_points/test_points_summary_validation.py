from dataclasses import replace

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._summarize_points import PointsSummary
from harpy.qc.points._summarize_points_by_feature import FeaturePointsSummary


def _summary(kind):
    """Build two planes over unequal-width bins; the feature mask retains even the zero bin."""
    axis = "feature_class" if kind == "class" else "feature"
    grid = xr.DataArray(
        np.array([[[1, 0]], [[0, 0]]], dtype=np.uint64),
        dims=(axis, "y", "x"),
        coords={axis: ["A", "B"], "y": [1.0], "x": [1.0, 2.5]},
    )
    metadata = PointsSummaryMetadata(
        points_name="calls",
        sample_id=None,
        feature_panel="panel",
        to_coordinate_system="global",
        crd=None,
        microns_per_unit=None,
        bin_size=2,
        x_edges=(0, 2, 3),
        y_edges=(0, 2),
    )
    kwargs = {"metadata": metadata, "per_feature": pd.DataFrame(), "spatial_counts": grid, "spatial_bins": None}
    if kind == "class":
        return PointsSummary(**kwargs, per_class=pd.DataFrame())
    mask = xr.DataArray([[True, True]], dims=("y", "x"), coords={"y": [1.0], "x": [1.0, 2.5]})
    return FeaturePointsSummary(**kwargs, retained_bin_mask=mask)


@pytest.fixture(params=["class", "feature"])
def summary(request):
    return _summary(request.param)


@pytest.mark.parametrize("binned", [False, True])
def test_construction_validates_structure_without_count_reductions(summary, binned, monkeypatch):
    """Neither valid unbinned results nor geometry validation need count scans or derived statistics."""
    changes = {}
    if not binned:
        changes = {
            "spatial_counts": None,
            "metadata": replace(summary.metadata, bin_size=None, x_edges=None, y_edges=None),
        }
        if isinstance(summary, FeaturePointsSummary):
            changes["retained_bin_mask"] = None

    def forbidden(*args, **kwargs):
        pytest.fail("Constructor validation must not compute counts or a class mask")

    monkeypatch.setattr(dask, "compute", forbidden)
    monkeypatch.setattr(xr.DataArray, "any", forbidden)
    monkeypatch.setattr(xr.DataArray, "sum", forbidden)
    result = replace(summary, **changes)
    assert result.spatial_counts is (summary.spatial_counts if binned else None)
    assert result.per_feature is summary.per_feature
    if isinstance(result, FeaturePointsSummary):
        assert result.retained_bin_mask is (summary.retained_bin_mask if binned else None)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dimensions", "dimensions"),
        ("missing_names", "identify"),
        ("duplicate_names", "unique strings"),
        ("lazy_counts", "in-memory"),
        ("nonfinite_edges", "x_edges"),
        ("unordered_edges", "x_edges"),
        ("wrong_edge_count", "x_edges"),
        ("shifted_centers", "bin centers"),
    ],
)
def test_malformed_geometry_fails_at_construction(summary, change, message):
    """Reject inconsistent geometry before any consumer receives the summary."""
    grid, metadata = summary.spatial_counts, summary.metadata
    axis = grid.dims[0]
    if change == "dimensions":
        grid = grid.transpose(axis, "x", "y")
    elif change == "missing_names":
        grid = grid.drop_vars(axis)
    elif change == "duplicate_names":
        grid = grid.assign_coords({axis: ["A", "A"]})
    elif change == "lazy_counts":
        grid = grid.chunk()
    elif change == "nonfinite_edges":
        metadata = replace(metadata, x_edges=(0, 2, np.inf))
    elif change == "unordered_edges":
        metadata = replace(metadata, x_edges=(0, 2, 2))
    elif change == "wrong_edge_count":
        metadata = replace(metadata, x_edges=(0, 2))
    elif change == "shifted_centers":
        grid = grid.assign_coords(x=[2, 3.5])
    with pytest.raises(ValueError, match=message):
        replace(summary, spatial_counts=grid, metadata=metadata)


def test_grid_and_binning_metadata_must_be_present_together(summary):
    with pytest.raises(ValueError, match="unbinned summary"):
        replace(summary, spatial_counts=None)
    with pytest.raises(ValueError, match="metadata.bin_size"):
        replace(summary, metadata=replace(summary.metadata, bin_size=None))
    with pytest.raises(ValueError, match="x_edges"):
        replace(summary, metadata=replace(summary.metadata, x_edges=None))


@pytest.mark.parametrize("change", ["missing", "non_boolean", "misaligned", "wrong_dimensions"])
def test_feature_constructor_checks_inherited_mask_alignment(change):
    summary = _summary("feature")
    mask = summary.retained_bin_mask
    if change == "missing":
        mask = None
    elif change == "non_boolean":
        mask = mask.astype(int)
    elif change == "misaligned":
        mask = mask[:, ::-1]
    elif change == "wrong_dimensions":
        mask = mask.transpose("x", "y")
    with pytest.raises(ValueError, match="retained_bin_mask"):
        replace(summary, retained_bin_mask=mask)
