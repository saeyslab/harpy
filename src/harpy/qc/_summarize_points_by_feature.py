"""Read-only feature totals and optional spatial summaries for selected panel features."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from spatialdata import SpatialData

from harpy._spatial_bounds import SpatialBounds
from harpy.qc._points_reduction import _reduce_points
from harpy.qc._points_summary_metadata import PointsSummaryMetadata
from harpy.qc._spatial_bin_summary import FeatureSpatialBinSummary, _summarize_feature_bins


@dataclass(frozen=True)
class FeaturePointsSummary:
    """Feature totals and optional spatial statistics for explicitly selected panel features.

    Attributes
    ----------
    metadata
        One :class:`~harpy.qc.PointsSummaryMetadata` holding source/panel/sample
        identity, coordinate system, crop, physical calibration and bin edges.
        Nested arrays and dataframes do not duplicate this context in ``attrs``.
    per_feature
        One row per requested panel feature, including zero detections, in
        panel order. Columns are ``feature``, ``feature_class`` and uint64
        ``n_points``: total points after feature and spatial selection,
        independent of binning. No complete-class statistics or within-class
        fractions are computed for this requested subset.
    spatial_counts
        Optional in-memory uint64 DataArray containing one XY count grid per
        selected feature. Dimensions are ``(feature, y, x)``: ``feature`` identifies
        the selected feature names, and each value counts that feature's points
        in one spatial bin. Includes empty bins and undetected selected features.
        Even a single feature keeps the feature dimension.
        Bin-center x/y coordinates use the requested
        coordinate system; counts are raw, unsmoothed and not area-normalized.
        None when binning was not requested.
    spatial_bins
        Optional :class:`~harpy.qc.FeatureSpatialBinSummary` with ``per_bin``
        measurements and ``per_feature`` statistics. All selected features share bins with
        any selected-feature points, including each feature's individual zeros.
        Its ``per_feature`` describes the distribution across bins, whereas
        the top-level table provides totals regardless of binning.
        None when binning was not requested.
    """

    metadata: PointsSummaryMetadata
    per_feature: pd.DataFrame
    spatial_counts: xr.DataArray | None
    spatial_bins: FeatureSpatialBinSummary | None

    @property
    def retained_bin_mask(self) -> xr.DataArray | None:
        """Return an uncached XY mask of bins with requested-feature points, or None without binning.

        Preserve grid coordinates. Editing the grid changes this mask but does
        not recalculate the stored spatial-bin statistics.
        """
        if self.spatial_counts is None:
            return None
        return self.spatial_counts.any(dim="feature")


def summarize_points_by_feature(
    sdata: SpatialData,
    points_name: str,
    *,
    features: str | Sequence[str],
    bin_size: float | None = None,
    max_grid_bytes: int | None = 1024**3,
    to_coordinate_system: str = "global",
    microns_per_unit: float | None = None,
    crd: SpatialBounds | tuple[float, ...] | None = None,
) -> FeaturePointsSummary:
    """Summarize selected panel features with optional spatial counts and bin statistics.

    Requires a feature panel referenced by the points element in
    ``sdata.attrs["harpy"]``; register one with :func:`harpy.pt.add_feature_panel`
    if missing. The panel defines feature/class columns and assignments.
    Validate every source feature/class assignment before feature or spatial
    filtering, so selection cannot hide invalid assignments. No segmentation,
    AnnData construction, plotting, or changes to the source store occur.

    Parameters
    ----------
    sdata
        SpatialData containing the points and their panel metadata. Both
        in-memory and Zarr-backed objects are supported.
    points_name
        One source points element; independent samples are not pooled.
    features
        Exact panel feature name or nonempty sequence of distinct names.
        Features may belong to different classes; results retain panel order
        (class order, then feature order), regardless of request order.
        Only selected features determine automatic extent and bin inclusion.
        Unknown features raise. Undetected features retain zero totals and,
        when binning, all-zero planes if another requested feature has points.
    bin_size
        Positive finite bin width in ``to_coordinate_system`` units. None
        returns feature totals without spatial counts or bin summaries.
        Without a crop, full-width bins extend beyond observed maxima to include
        every selected point. Cropped terminal bins may be narrower.
    max_grid_bytes
        Maximum final count-grid bytes, default 1 GiB; None disables the limit.
        Check ``n_features * n_y_bins * n_x_bins * 8`` before allocating edges
        or counting (after extent discovery when needed). This is not a peak-
        memory budget: intermediate reductions and ``spatial_bins.per_bin``
        require additional memory. The latter has one row per selected feature
        and retained bin: 20 features over 100,000 retained bins mean two million rows.
        The limit is not enforced when ``bin_size=None``.
    to_coordinate_system
        Registered coordinate system for the grid and crop. Transform the full
        XY/XYZ coordinates before cropping, then project to XY for binning.
        Registration is only required when cropping or binning is requested.
    microns_per_unit
        Optional positive finite micrometers per coordinate unit. Use 1.0 for
        micron coordinates. Adds actual ``bin_area_um2`` measurements without
        changing coordinates or counts; None omits physical areas.
    crd
        Optional :class:`harpy.SpatialBounds` or tuple
        ``(xmin, xmax, ymin, ymax[, zmin, zmax])`` in the requested coordinate
        system. All intervals apply after transformation. Optional z bounds
        require 3D points; spatial counts, when requested, remain an XY grid.
        Cropping also applies to feature totals without binning. Metadata retains
        the normalized bounds object. See :class:`harpy.SpatialBounds` for bounds rules.

    Returns
    -------
    FeaturePointsSummary
        Feature totals and shared metadata, plus optional spatial counts and
        bin statistics. Raise ValueError if no requested-feature points remain
        after cropping, regardless of binning or an explicit crop.

    Notes
    -----
    Request features together to compare them over the same bins. For
    ``features=["EPCAM", "VIM"]``, EPCAM=0 and VIM=5 retains the bin and
    contributes a zero to EPCAM's statistics. Requesting only EPCAM excludes
    that bin. Plotting one feature afterwards keeps the original population.
    This is detection-based bin inclusion, not a tissue annotation.

    All requested features share one partition-wise count pass, with a
    preliminary extent reduction only when binning without ``crd``. Bin statistics
    use the completed grid without rereading points. A separate call reads source
    points again: feature planes cannot be reconstructed from class-level grids.

    See Also
    --------
    harpy.qc.summarize_points
    harpy.qc.spatial_bin_histogram_by_feature
    harpy.pt.add_feature_panel

    Examples
    --------
    .. code-block:: python

        totals = hp.qc.summarize_points_by_feature(
            sdata, "transcripts", features=["EPCAM", "VIM"]
        )
        totals.per_feature  # no spatial grid is computed

        summary = hp.qc.summarize_points_by_feature(
            sdata, "transcripts", features=["EPCAM", "VIM"], bin_size=10,
            to_coordinate_system="sample_micron", microns_per_unit=1.0,
        )
        summary.per_feature  # totals, also available without binning
        summary.spatial_counts.sel(feature="EPCAM")
        summary.spatial_bins.per_feature
        hp.qc.spatial_bin_histogram_by_feature(summary, feature="EPCAM")
    """
    result = _reduce_points(
        sdata,
        points_name,
        selected_names=features,
        summary_axis="feature",
        bin_size=bin_size,
        max_grid_bytes=max_grid_bytes,
        to_coordinate_system=to_coordinate_system,
        microns_per_unit=microns_per_unit,
        crd=crd,
    )
    class_by_feature = result.panel.class_by_feature
    per_feature = pd.DataFrame(
        {
            "feature": result.selected_names,
            "feature_class": [class_by_feature[feature] for feature in result.selected_names],
            "n_points": result.feature_counts.reindex(result.selected_names, fill_value=0).to_numpy(dtype=np.uint64),
        }
    )
    bins = (
        None
        if result.spatial_counts is None
        else _summarize_feature_bins(result.spatial_counts, metadata=result.metadata, panel=result.panel)
    )
    return FeaturePointsSummary(
        metadata=result.metadata, per_feature=per_feature, spatial_counts=result.spatial_counts, spatial_bins=bins
    )
