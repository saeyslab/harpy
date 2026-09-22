"""Read-only feature totals and optional spatial summaries for selected panel features."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import xarray as xr
from spatialdata import SpatialData

from harpy._feature_panels import _resolve_points_feature_panel
from harpy.qc.points._points_binning import _check_grid_budget
from harpy.qc.points._points_reduction import (
    _compute_point_reductions,
    _resolve_point_coordinate_transform,
    _spatial_count_array,
    _validate_point_columns,
)
from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._points_summary_schema import _FEATURE_CLASS_KEY, _FEATURE_KEY, _N_POINTS_KEY
from harpy.qc.points._spatial_bin_summary import FeatureSpatialBinSummary, _summarize_feature_bins
from harpy.qc.points._summarize_points import PointsSummary


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
        measurements and ``per_feature`` statistics over the reference
        ``PointsSummary``'s retained bins, including each feature's zeros.
        Its ``per_feature`` describes the distribution across bins, whereas
        the top-level table provides totals regardless of binning.
        None when binning was not requested.
    retained_bin_mask
        Boolean XY mask inherited from the reference summary, or None without
        binning. This population is independent of the requested features:
        even bins where all requested features have zero counts remain included.
        It is not recalculated when the feature count grid is edited.
    """

    metadata: PointsSummaryMetadata
    per_feature: pd.DataFrame
    spatial_counts: xr.DataArray | None
    spatial_bins: FeatureSpatialBinSummary | None
    retained_bin_mask: xr.DataArray | None


def summarize_points_by_feature(
    sdata: SpatialData,
    *,
    summary: PointsSummary,
    features: str | Sequence[str],
    max_grid_bytes: int | None = 1024**3,
) -> FeaturePointsSummary:
    """Summarize features over an existing points summary's spatial population.

    The reference summary determines the source, spatial settings and retained
    bins. Changing the requested features does not change another feature's
    grid or statistics. No source data or metadata are modified.

    Parameters
    ----------
    sdata
        SpatialData containing the reference summary's source points and panel.
        Both in-memory and Zarr-backed objects are supported. The source and
        its transformations must remain unchanged since computing ``summary``;
        source identity checks do not establish snapshot freshness.
    summary
        Result of :func:`harpy.qc.summarize_points`. Its metadata supplies the
        points name, coordinate system, crop (including z), calibration and
        exact bin edges. Its retained-bin mask defines the statistical population.
        Without binning, return the requested feature totals from this summary
        without reading source points. With binning, count features from source
        points on the existing grid; no extent discovery is performed.
    features
        Exact feature name or nonempty sequence of distinct names from the
        classes represented in ``summary``. Results retain panel order
        (class order, then feature order), not request order. Unknown features
        or features from unrepresented classes raise. Undetected features
        retain zero totals and, with binning, all-zero planes and bin rows,
        even when none of the requested features have detections.
    max_grid_bytes
        Maximum final count-grid bytes, default 1 GiB; None disables the limit.
        Check ``n_features * n_y_bins * n_x_bins * 8`` before counting. This is
        not a peak-memory budget: reductions and ``spatial_bins.per_bin``
        require additional memory. The latter has one row per selected feature
        and retained bin: 20 features over 100,000 retained bins mean two million rows.
        The limit is not enforced for an unbinned reference summary.

    Returns
    -------
    FeaturePointsSummary
        Feature totals and inherited metadata, plus optional spatial counts,
        bin statistics and the reference population's mask.

    Notes
    -----
    If an endogenous reference retains a bin containing VIM but no EPCAM,
    that bin contributes zero to EPCAM's statistics even when requesting
    only EPCAM. Bin inclusion is based on the reference classes' detections,
    not a tissue annotation or the requested features' detections.

    With binning, all requested features share one partition-wise count pass.
    It validates every source feature/class assignment before selection, so
    filtering cannot hide invalid assignments. Feature planes cannot be
    reconstructed from class-level grids. Without binning, reuse the reference
    totals and their prior validation; no source-content scan is performed.

    See Also
    --------
    harpy.qc.summarize_points
    harpy.qc.spatial_bin_histogram_by_feature
    harpy.pt.add_feature_panel

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(
            sdata, "transcripts", feature_classes="Endogenous", bin_size=10,
            to_coordinate_system="sample_micron", microns_per_unit=1.0,
        )
        features = hp.qc.summarize_points_by_feature(
            sdata, summary=summary, features=["EPCAM", "VIM"]
        )
        features.spatial_counts.sel(feature="EPCAM")
        features.spatial_bins.per_feature
        hp.qc.spatial_bin_histogram_by_feature(features, feature="EPCAM")
    """
    if not isinstance(summary, PointsSummary):
        raise TypeError("summary must be a PointsSummary returned by hp.qc.summarize_points().")
    return _reduce_points_by_feature(sdata, summary=summary, features=features, max_grid_bytes=max_grid_bytes)


def _reduce_points_by_feature(
    sdata: SpatialData, *, summary: PointsSummary, features: str | Sequence[str], max_grid_bytes: int | None
) -> FeaturePointsSummary:
    """Reduce requested features within an existing reference and assemble their summary.

    Validate source identity and feature selection once. Unbinned references
    already contain the totals, so return them without reading source points.
    Otherwise inherit exact edges, metadata and bin inclusion, then use
    ``_compute_point_reductions()`` for the shared partition validation/count
    pass. Entirely undetected features remain valid on this reference grid.
    """
    # Check that the panel key and sample ID saved in summary.metadata still
    # match the current metadata for this points element in sdata.
    # This does not detect changes to point rows or coordinate transformations.
    metadata = summary.metadata
    if metadata.points_name not in sdata.points:
        raise ValueError(f"Points element {metadata.points_name!r} does not exist.")
    panel_name, panel, points_record = _resolve_points_feature_panel(sdata, metadata.points_name)
    if panel_name != metadata.feature_panel or points_record.get("sample_id") != metadata.sample_id:
        raise ValueError("The reference summary's source metadata no longer matches; recompute summarize_points().")
    # Compare assignments only for classes represented in the supplied summary.
    # For example, an Endogenous-only summary might contain:
    # {"EPCAM": "Endogenous", "VIM": "Endogenous"}.
    # The current panel must contain the same assignments for these classes.
    # Features belonging to classes not included in the supplied summary
    # are excluded from this comparison.
    reference_classes = set(summary.per_class[_FEATURE_CLASS_KEY])
    reference_assignments = dict(
        zip(summary.per_feature[_FEATURE_KEY], summary.per_feature[_FEATURE_CLASS_KEY], strict=True)
    )
    expected_assignments = {
        feature: feature_class
        for feature, feature_class in panel.class_by_feature.items()
        if feature_class in reference_classes
    }
    if reference_assignments != expected_assignments:
        raise ValueError("The reference summary's features no longer match the panel; recompute summarize_points().")

    if not isinstance(features, (str, Sequence)):
        raise ValueError("features must contain one or more exact feature names from the reference summary.")
    requested = (features,) if isinstance(features, str) else tuple(features)
    if not requested or any(not isinstance(feature, str) for feature in requested):
        raise ValueError("features must contain one or more exact feature names from the reference summary.")
    if len(set(requested)) != len(requested):
        raise ValueError("features must not contain duplicate names.")
    # Features must belong to the supplied summary, not merely the full panel.
    # For an Endogenous-only summary, "EPCAM" is allowed, but "Negative1"
    # is rejected even if it belongs to the panel.
    unknown = set(requested) - set(reference_assignments)
    if unknown:
        raise ValueError(f"Unknown features in the reference summary: {sorted(unknown)}; select represented classes.")
    # Use the panel's class/feature order, not the request order.
    selected_names = tuple(feature for feature in panel.class_by_feature if feature in requested)
    if max_grid_bytes is not None and (
        isinstance(max_grid_bytes, bool) or not isinstance(max_grid_bytes, Integral) or max_grid_bytes < 1
    ):
        raise ValueError("max_grid_bytes must be a positive integer or None.")

    per_feature = (
        summary.per_feature.set_index(_FEATURE_KEY)
        .loc[list(selected_names), [_FEATURE_CLASS_KEY, _N_POINTS_KEY]]
        .reset_index()
    )
    if metadata.bin_size is None:
        return FeaturePointsSummary(
            metadata=metadata, per_feature=per_feature, spatial_counts=None, spatial_bins=None, retained_bin_mask=None
        )
    if summary.spatial_counts is None or metadata.x_edges is None or metadata.y_edges is None:
        raise ValueError("The reference summary is missing its spatial grid or edges; recompute summarize_points().")
    retained_bin_mask = summary.retained_bin_mask
    if retained_bin_mask is None or retained_bin_mask.dims != ("y", "x"):
        raise ValueError("The reference summary must contain an XY retained-bin mask; recompute summarize_points().")
    # Match physical bin locations, not just array shape. Keep the parent mask
    # as a snapshot; requested-feature zeros must not redefine this population.
    for axis, edges in (("x", metadata.x_edges), ("y", metadata.y_edges)):
        centers = np.asarray(edges[:-1]) + np.diff(edges) / 2
        if not np.array_equal(retained_bin_mask.coords[axis].values, centers):
            raise ValueError("The reference grid coordinates do not match its bin edges; recompute summarize_points().")
    points = sdata.points[metadata.points_name]
    _validate_point_columns(points, points_name=metadata.points_name, panel=panel)
    axes, matrix = _resolve_point_coordinate_transform(
        points, points_name=metadata.points_name, to_coordinate_system=metadata.to_coordinate_system, crd=metadata.crd
    )
    _check_grid_budget(
        (len(selected_names), len(metadata.y_edges) - 1, len(metadata.x_edges) - 1), max_grid_bytes=max_grid_bytes
    )
    edges = (np.asarray(metadata.x_edges), np.asarray(metadata.y_edges))
    feature_counts, bin_counts = _compute_point_reductions(
        points,
        panel=panel,
        points_name=metadata.points_name,
        selected_names=selected_names,
        summary_axis=_FEATURE_KEY,
        axes=axes,
        matrix=matrix,
        crd=metadata.crd,
        edges=edges,
    )
    # The reference defines a population even when all requested features are
    # undetected. Zero-fill their planes; do not reject the empty reduction.
    grid = _spatial_count_array(bin_counts, selected_names=selected_names, summary_axis=_FEATURE_KEY, edges=edges)
    per_feature[_N_POINTS_KEY] = feature_counts.reindex(selected_names, fill_value=0).to_numpy(dtype=np.uint64)
    bins = _summarize_feature_bins(grid, metadata=metadata, panel=panel, retained_bin_mask=retained_bin_mask)
    return FeaturePointsSummary(
        metadata=metadata,
        per_feature=per_feature,
        spatial_counts=grid,
        spatial_bins=bins,
        retained_bin_mask=retained_bin_mask,
    )
