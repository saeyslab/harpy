"""Read-only rendering of precomputed class and feature count grids."""

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from harpy.qc.points._points_summary_schema import _FEATURE_CLASS_KEY, _FEATURE_KEY
from harpy.qc.points._summarize_points import PointsSummary
from harpy.qc.points._summarize_points_by_feature import FeaturePointsSummary


def plot_points_density(
    summary: PointsSummary | FeaturePointsSummary,
    *,
    feature_class: str | None = None,
    features: str | Sequence[str] | None = None,
    normalization: Literal["per_area", "per_panel_feature", "per_panel_feature_per_area"] | None = None,
    cmap: str | Colormap = "cividis",
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot precomputed point counts or densities at their original bin resolution.

    No source points are read and no data are modified. Bins outside the
    summary's retained population are transparent; zeros inside it remain
    visible. Feature summaries retain the population of their parent class
    summary, even where the displayed features have no detections.

    Parameters
    ----------
    summary
        Binned :class:`harpy.qc.PointsSummary` or
        :class:`harpy.qc.FeaturePointsSummary`. Its metadata supplies bin
        boundaries, coordinate system and optional physical calibration.
    feature_class
        Exact class in a ``PointsSummary`` grid. None sums all computed classes.
        Not supported for ``FeaturePointsSummary``.
    features
        Exact feature name or nonempty sequence of distinct names in a
        ``FeaturePointsSummary`` grid. A sequence sums the selected planes.
        None selects the sole feature; multiple features require a selection.
        Not supported for ``PointsSummary``.
    normalization
        None displays points per bin. ``"per_area"`` divides by each bin's
        actual area in µm² and requires ``metadata.microns_per_unit``.
        For class summaries, ``"per_panel_feature"`` divides by the displayed
        classes' complete panel size, including undetected features;
        ``"per_panel_feature_per_area"`` divides by both panel size and area.
        Pooled classes use summed counts divided by their combined panel size.
    cmap
        Matplotlib colormap. Excluded bins are transparent regardless of its
        configured bad-value color.
    colorbar
        Whether to show a colorbar labelled with the selection and units.
    vmin, vmax
        Optional color limits, useful for comparing multiple maps on one scale.
        These do not normalize or alter the counts. All-zero maps default to
        a 0–1 color scale when neither limit is supplied.
    alpha
        Density-layer opacity, between 0 and 1.
    figsize
        Figure size in inches when creating axes, using Matplotlib's configured
        DPI. Ignored when ``ax`` is supplied. Neither setting changes binning.
    title
        Optional axes title. None leaves the title unchanged.
    ax
        Axes to reuse, or None to create a figure. Empty axes are initialized
        with the grid extent, equal aspect and downward-increasing y, as for
        images. Existing plots retain their limits, aspect and orientation.
        For image overlays, the caller must render the image in
        ``summary.metadata.to_coordinate_system``; this function neither reads
        the image nor applies another coordinate transformation.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the density map, or an empty-state message when no
        bins are retained. An all-zero selected plane alone is not empty.

    See Also
    --------
    harpy.qc.summarize_points
    harpy.qc.summarize_points_by_feature
    harpy.pl.plot_sdata

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(
            sdata, "transcripts", bin_size=10,
            to_coordinate_system="sample_micron", microns_per_unit=1,
        )
        hp.pl.plot_points_density(summary, feature_class="Endogenous")

        ax = hp.pl.plot_sdata(
            sdata, image_name="DAPI", channel="DAPI",
            to_coordinate_system=summary.metadata.to_coordinate_system,
        )
        hp.pl.plot_points_density(summary, feature_class="Endogenous", ax=ax, alpha=0.5)
    """
    if not isinstance(summary, (PointsSummary, FeaturePointsSummary)):
        raise TypeError("summary must be a PointsSummary or FeaturePointsSummary, not source data or a bare array.")
    if summary.spatial_counts is None:
        raise ValueError(
            "Spatial counts are missing. Call hp.qc.summarize_points() with bin_size; "
            "for feature maps, pass that summary to hp.qc.summarize_points_by_feature()."
        )
    if normalization not in (None, "per_area", "per_panel_feature", "per_panel_feature_per_area"):
        raise ValueError(f"Unknown normalization: {normalization!r}.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    axis = _FEATURE_CLASS_KEY if isinstance(summary, PointsSummary) else _FEATURE_KEY
    grid = summary.spatial_counts
    # Grid/edge/mask consistency belongs to the summary's constructor.
    x_edges = np.asarray(summary.metadata.x_edges, dtype=float)
    y_edges = np.asarray(summary.metadata.y_edges, dtype=float)
    retained = np.asarray(summary.retained_bin_mask)
    names, selection_label = _density_selection(grid, axis=axis, feature_class=feature_class, features=features)
    # Copy into display values: normalization must never modify stored counts.
    values = grid.sel({axis: list(names)}).sum(dim=axis).to_numpy().astype(float)
    units = "Points"
    if normalization in ("per_panel_feature", "per_panel_feature_per_area"):
        if not isinstance(summary, PointsSummary):
            raise ValueError("Panel-feature normalization is supported only for PointsSummary, not feature grids.")
        panel_sizes = summary.panel_feature_counts
        if any(name not in panel_sizes or panel_sizes[name] <= 0 for name in names):
            raise ValueError("Panel-feature normalization requires a positive panel size for every displayed class.")
        values /= sum(panel_sizes[name] for name in names)
        units += " per panel feature"
    if normalization in ("per_area", "per_panel_feature_per_area"):
        calibration = summary.metadata.microns_per_unit
        if calibration is None or not np.isfinite(calibration) or calibration <= 0:
            raise ValueError("Area normalization requires a positive, finite summary.metadata.microns_per_unit.")
        # Actual widths preserve the area of narrower terminal bins after a crop.
        values /= np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :] * calibration**2
        units += " per µm²"
    else:
        units += " per bin"

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    preserve_view = ax.has_data() or not ax.get_autoscalex_on() or not ax.get_autoscaley_on()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    autoscale_x, autoscale_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()

    if retained.any():
        # Matplotlib otherwise expands a constant-zero colorbar below zero.
        if vmin is None and vmax is None and not values[retained].any():
            vmin, vmax = 0, 1
        # Keep per-plane zeros visible. Only the shared population determines transparency.
        colors = plt.get_cmap(cmap).copy()
        colors.set_bad(alpha=0)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.array(values, mask=~retained),
            shading="flat",
            cmap=colors,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            zorder=max((image.get_zorder() for image in ax.images), default=0) + 1,
            rasterized=True,
        )
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, shrink=0.75, label=f"{selection_label} — {units}")
    else:
        ax.text(0.5, 0.5, "No retained spatial bins", ha="center", va="center", transform=ax.transAxes)

    if preserve_view:
        # Adding a mesh can autoscale an existing image's viewport; restore it without flipping axes.
        ax.set_xlim(xlim, auto=autoscale_x)
        ax.set_ylim(ylim, auto=autoscale_y)
    else:
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[-1], y_edges[0])
        ax.set_aspect("equal")
        ax.set_xlabel(f"x ({summary.metadata.to_coordinate_system})")
        ax.set_ylabel(f"y ({summary.metadata.to_coordinate_system})")
    if title is not None:
        ax.set_title(title)
    if created_ax:
        ax.figure.tight_layout()
    return ax


def _density_selection(
    grid: xr.DataArray, *, axis: str, feature_class: str | None, features: str | Sequence[str] | None
) -> tuple[tuple[str, ...], str]:
    """Select already computed planes; pooling never changes the retained-bin population."""
    available = tuple(grid[axis].values.tolist())
    if axis == _FEATURE_CLASS_KEY:
        if features is not None:
            raise ValueError("Use feature_class with PointsSummary; features requires FeaturePointsSummary.")
        if feature_class is not None and not isinstance(feature_class, str):
            raise ValueError("feature_class must be one exact class name or None.")
        names = available if feature_class is None else (feature_class,)
    else:
        if feature_class is not None:
            raise ValueError("Use features with FeaturePointsSummary, not feature_class.")
        if features is None:
            if len(available) != 1:
                raise ValueError("Select features explicitly when the summary contains multiple feature planes.")
            names = available
        elif isinstance(features, str):
            names = (features,)
        elif isinstance(features, Sequence):
            names = tuple(features)
        else:
            raise ValueError("features must be one exact feature name or a nonempty sequence of distinct names.")
        if not names or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("features must be one exact feature name or a nonempty sequence of distinct names.")
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Requested {axis} names are absent from spatial_counts: {sorted(unknown)}.")
    label = names[0] if len(names) == 1 else f"Combined: {' + '.join(names)}"
    return names, label
