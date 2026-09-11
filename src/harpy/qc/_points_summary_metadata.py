"""Single-owner source and geometry context for computed point summaries."""

from dataclasses import dataclass

from harpy._spatial_bounds import SpatialBounds


@dataclass(frozen=True, kw_only=True)
class PointsSummaryMetadata:
    """Immutable context owned by a :class:`~harpy.qc.PointsSummary`.

    Attributes
    ----------
    points_name, sample_id, feature_panel
        Source points name, optional sample identity, and referenced panel key.
        These describe the computation's inputs, not a live metadata lookup.
    to_coordinate_system
        Requested coordinate system, used whenever cropping or binning occurs.
        All bin coordinates and edges use this frame and its units.
    crd
        Validated :class:`~harpy.SpatialBounds` with named ``x``, ``y``, and
        optional ``z`` intervals, or None. Bounds are half-open and use
        ``to_coordinate_system`` units. The crop need not equal the grid extent.
    microns_per_unit
        Optional physical calibration of the selected coordinate system.
        It changes area/density reporting, not coordinates or raw counts.
    bin_size
        Requested bin width, or None when binning was not requested.
    x_edges, y_edges
        Tuples of complete grid boundaries in the selected coordinate system,
        including excluded bins. Both are None without binning. Terminal bins
        can be narrower after cropping. ``extent`` is derived from these edges.
    """

    points_name: str
    sample_id: str | None
    feature_panel: str
    to_coordinate_system: str
    crd: SpatialBounds | None
    microns_per_unit: float | None
    bin_size: float | None
    x_edges: tuple[float, ...] | None
    y_edges: tuple[float, ...] | None

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """Return ``(xmin, xmax, ymin, ymax)`` from grid edges, or None without bins."""
        if self.x_edges is None or self.y_edges is None:
            return None
        return self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1]
