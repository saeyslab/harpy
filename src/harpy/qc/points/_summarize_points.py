"""Read-only, panel-aware summaries of original points."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import xarray as xr
from spatialdata import SpatialData

from harpy._feature_panels import _FeaturePanelContract
from harpy._spatial_bounds import SpatialBounds
from harpy.qc.points._points_reduction import _reduce_points_by_class
from harpy.qc.points._points_summary_metadata import PointsSummaryMetadata
from harpy.qc.points._points_summary_schema import _FEATURE_CLASS_KEY, _FEATURE_KEY, _N_FEATURES_KEY, _N_POINTS_KEY
from harpy.qc.points._points_summary_validation import _validate_spatial_summary
from harpy.qc.points._spatial_bin_summary import SpatialBinSummary, _summarize_spatial_bins


@dataclass(frozen=True)
class PointsSummary:
    """Computed summaries of one original points element and its feature panel.

    Construction validates spatial structure, not numerical summaries.
    In-place edits to the contained arrays or tables do not trigger validation.

    Attributes
    ----------
    metadata
        One :class:`PointsSummaryMetadata` record containing source identity,
        coordinate system, crop, calibration, and bin edges.
    per_feature
        One row per selected panel feature, including zero detections. Columns
        are ``feature``, ``feature_class``, ``n_points``, and
        ``within_class_fraction``: this feature's point count divided by the
        total point count of its own class, after selection. Values range
        from 0 to 1; NaN when the class has no points.
    per_class
        One row per selected class: ``n_features``, ``n_zero_features``,
        ``pct_zero_features`` (0–100), ``n_points``,
        ``mean_points_per_feature``, ``median_points_per_feature``,
        ``p95_points_per_feature``, ``top_n`` (requested N), ``n_top_features``
        (min(N, panel size)), and ``pct_points_top_n_features`` (0–100,
        missing for a zero-point class). All feature statistics include zeros.
    spatial_counts
        Optional in-memory uint64 DataArray with dimensions
        ``(feature_class, y, x)`` and bin-center x/y coordinates. These
        coordinates use ``metadata.to_coordinate_system`` and its units,
        after transforming the source points. Coordinates are not rebased
        to zero or replaced by bin indices. See ``metadata.x_edges`` and
        ``metadata.y_edges`` for grid boundaries. This is not a registered
        SpatialData image element.
        Counts are raw, unsmoothed, and not normalized by area or panel size.
        Empty bins remain in the grid; ``retained_bin_mask`` identifies bins
        included in ``spatial_bins``.
    spatial_bins
        Optional :class:`SpatialBinSummary` with per-bin measurements and
        statistics across bins containing at least one point from the selected
        feature classes.
        Its ``per_class`` describes spatial bins, unlike this container's
        feature-level ``per_class``. None when binning was not requested.

    Notes
    -----
    Retain this parent result when interpreting detached tables or arrays;
    source and geometry context live only in ``metadata``, not in nested
    ``attrs`` or repeated source-identity columns. Add identity columns
    explicitly if an exported table needs them. Selected panel sizes
    are available through the derived ``panel_feature_counts`` property.
    """

    metadata: PointsSummaryMetadata
    per_feature: pd.DataFrame
    per_class: pd.DataFrame
    spatial_counts: xr.DataArray | None
    spatial_bins: SpatialBinSummary | None

    def __post_init__(self) -> None:
        _validate_spatial_summary(self.spatial_counts, metadata=self.metadata, axis=_FEATURE_CLASS_KEY)

    @property
    def panel_feature_counts(self) -> dict[str, int]:
        """Return selected panel sizes from ``per_class.n_features``, including undetected features."""
        return {
            str(name): int(count)
            for name, count in zip(self.per_class[_FEATURE_CLASS_KEY], self.per_class[_N_FEATURES_KEY], strict=True)
        }

    @property
    def retained_bin_mask(self) -> xr.DataArray | None:
        """Return a boolean XY mask of bins with any selected-class points, or None without binning.

        Preserve the grid's x/y coordinates. Derive the mask from the current
        ``spatial_counts`` on each access, without caching or reading source
        points. Grid edits affect this mask but do not recalculate the stored
        ``spatial_bins`` statistics.
        """
        if self.spatial_counts is None:
            return None
        return self.spatial_counts.any(dim=_FEATURE_CLASS_KEY)


def summarize_points(
    sdata: SpatialData,
    points_name: str,
    *,
    feature_classes: str | Sequence[str] | None = None,
    bin_size: float | None = None,
    max_grid_bytes: int | None = 1024**3,
    to_coordinate_system: str = "global",
    microns_per_unit: float | None = None,
    crd: SpatialBounds | tuple[float, ...] | None = None,
    top_n: int = 20,
) -> PointsSummary:
    """Summarize original points by panel feature/class and optional spatial bins.

    Requires the selected points element to reference a feature panel in
    ``sdata.attrs["harpy"]``. Points elements without this metadata are not
    supported, including when ``feature_classes=None``. Register a complete
    assay panel explicitly with ``hp.pt.add_feature_panel()`` before calling
    this function; no panel is inferred or registered here.

    Include every selected panel feature, even with no detected points. This
    is independent of segmentation: points outside cells are included, and no
    labels, images, or AnnData table are required. The operation is read-only.
    Raise ValueError if no points remain after class and spatial selection,
    regardless of ``bin_size`` or whether ``crd`` was supplied.

    Parameters
    ----------
    sdata
        SpatialData containing the points and their Harpy feature-panel metadata.
        Both in-memory and Zarr-backed objects are supported.
    points_name
        One points element to summarize; independent samples are not pooled.
    feature_classes
        Exact panel class name or non-empty sequence of distinct names. None
        selects all classes, including endogenous features. Results retain
        panel class/feature order, regardless of selection order.
        Only selected classes contribute to counts, inferred extent, and bin
        inclusion. Retain bins containing at least one point from the selected
        classes, including each class's zeros within that shared population.
    bin_size
        Positive finite bin width in ``to_coordinate_system`` units. None
        skips binning and returns ``spatial_counts=None`` and ``spatial_bins=None``.
    max_grid_bytes
        Maximum bytes for the final dense uint64 count grid; defaults to 1 GiB.
        A positive integer, or None to disable the limit. When binning is
        requested, raise ValueError if ``n_classes * n_y_bins * n_x_bins * 8``
        exceeds this limit, before allocating bin edges or reducing counts.
        Without ``crd``, the selected-class extent calculation happens first.
        This bounds only the final count array, not total peak memory: pandas
        summaries (including ``spatial_bins.per_bin``), the occupancy mask,
        coordinates, and other intermediate objects require more.
        The limit is not enforced when ``bin_size=None``.
    to_coordinate_system
        Registered coordinate system used for crop and bins. Apply the full
        same-dimensional affine transformation before projecting to XY. The
        registration is only required when cropping or binning is requested.
    microns_per_unit
        Positive finite micrometers per unit of ``to_coordinate_system``,
        shared by x and y. Adds ``bin_area_um2`` to ``spatial_bins.per_bin``
        when binning; None omits it. Use 1.0 for micron coordinates or the
        calibrated micrometers per pixel for pixel coordinates. Units are
        not inferred. ``bin_area`` remains width × height in coordinate units
        squared. Coordinates, bins, and count statistics are unchanged.
    crd
        Optional :class:`harpy.SpatialBounds`, for example
        ``hp.SpatialBounds(x=(xmin, xmax), y=(ymin, ymax), z=(zmin, zmax))``.
        The optional z interval restricts 3D points; None leaves z unrestricted.
        Tuples ``(xmin, xmax, ymin, ymax)`` and
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` are also accepted.
        All bounds use ``to_coordinate_system`` and apply after the full
        coordinate transformation, before projecting to XY for binning.
        Supplying z bounds for 2D points raises ValueError.
        Every output uses this crop. Last bins may be narrower than bin_size.
        Result metadata retains a validated ``SpatialBounds`` object, regardless
        of the input form; None when no crop was supplied.
    top_n
        Positive integer used only for the top-N concentration statistic.
        It does not truncate per-feature results or other class statistics.

    Returns
    -------
    PointsSummary
        Computed per-feature and per-class dataframes and optional raw-count
        spatial grid and bin summaries. See :class:`PointsSummary` for output
        columns. Only reduced results reach driver memory, not the original points.

    Notes
    -----
    The authoritative panel is resolved through::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
            -> sdata.attrs["harpy"]["feature_panels"][panel_name]

    The panel defines source ``feature_key``, categorical ``feature_class_key``,
    classes, and complete feature lists. Every source point's feature must
    occur in the panel and its observed class must match. These checks run
    partition-wise before class, crop, or z filtering, so selection cannot hide
    invalid feature/class assignments.

    Separate class-only calls can use different grids and bin populations.
    For comparisons over one population, compute the relevant classes together
    and select from the returned summaries. An explicit crop fixes the grid,
    not bin inclusion.
    This is not a tissue mask: residual background detections from selected
    classes remain included, and empty tissue bins are excluded. Bin areas
    include the full geometric bin, not only tissue.
    No smoothing or plotting is performed.

    See Also
    --------
    harpy.qc.summarize_points_by_feature
    harpy.qc.spatial_bin_histogram

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(
            sdata,
            "transcripts",
            feature_classes=["Negative", "SystemControl"],
            bin_size=200,
            to_coordinate_system="sample_micron",
            microns_per_unit=1.0,
        )
        summary.per_feature  # includes panel features with zero detections
        summary.spatial_counts  # raw (feature_class, y, x) bin counts
        summary.spatial_bins.per_bin  # prepared histogram measurements
        summary.spatial_bins.per_class  # overview across included bins
        summary.metadata.to_coordinate_system  # shared coordinate context
    """
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    result = _reduce_points_by_class(
        sdata,
        points_name,
        feature_classes=feature_classes,
        bin_size=bin_size,
        max_grid_bytes=max_grid_bytes,
        to_coordinate_system=to_coordinate_system,
        microns_per_unit=microns_per_unit,
        crd=crd,
    )
    per_feature, per_class = _summary_frames(
        result.feature_counts, panel=result.panel, classes=result.selected_names, top_n=int(top_n)
    )
    spatial_bins = (
        None
        if result.spatial_counts is None
        else _summarize_spatial_bins(result.spatial_counts, metadata=result.metadata)
    )
    return PointsSummary(
        metadata=result.metadata,
        per_feature=per_feature,
        per_class=per_class,
        spatial_counts=result.spatial_counts,
        spatial_bins=spatial_bins,
    )


def _summary_frames(
    counts: pd.Series, *, panel: _FeaturePanelContract, classes: tuple[str, ...], top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-fill against the selected panel, then derive statistics without rescanning points."""
    feature_frames, class_rows = [], []
    for name in classes:
        features = panel.features_by_class[name]
        values = counts.reindex(features, fill_value=0).to_numpy(dtype=np.uint64)
        total = values.sum(dtype=np.uint64)
        feature_frames.append(
            pd.DataFrame(
                {
                    _FEATURE_KEY: features,
                    _FEATURE_CLASS_KEY: name,
                    _N_POINTS_KEY: values,
                    "within_class_fraction": values / total if total else np.full(len(values), np.nan),
                }
            )
        )
        n_top = min(top_n, len(values))
        n_zero = int((values == 0).sum())
        class_rows.append(
            {
                _FEATURE_CLASS_KEY: name,
                _N_FEATURES_KEY: len(features),
                "n_zero_features": n_zero,
                "pct_zero_features": 100 * n_zero / len(features),
                _N_POINTS_KEY: total,
                "mean_points_per_feature": float(values.mean()),
                "median_points_per_feature": float(np.median(values)),
                "p95_points_per_feature": float(np.percentile(values, 95)),
                "top_n": top_n,
                "n_top_features": n_top,
                "pct_points_top_n_features": 100 * (np.sort(values)[-n_top:].sum() / total) if total else np.nan,
            }
        )
    return pd.concat(feature_frames, ignore_index=True), pd.DataFrame(class_rows)
