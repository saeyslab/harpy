"""Shared, validated spatial bounds independent of a particular Harpy operation."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, kw_only=True)
class SpatialBounds:
    """Axis-aligned spatial bounds with named x/y and optional z intervals.

    Parameters
    ----------
    x
        ``(xmin, xmax)`` bounds for the x axis.
    y
        ``(ymin, ymax)`` bounds for the y axis.
    z
        Optional ``(zmin, zmax)`` bounds. None leaves z unrestricted.

    Notes
    -----
    The x and y axes are required; all axes must be supplied by keyword.
    Construction validates that each supplied axis has two finite bounds with
    minimum strictly less than maximum. Bounds are normalized to immutable
    float tuples, so changing an input list cannot change the bounds later.

    This type describes coordinate bounds, not integer raster slices. The
    consuming function defines their coordinate system and boundary behavior.
    Currently :func:`harpy.qc.summarize_points` accepts this type: all bounds
    use ``to_coordinate_system`` units and are applied after the full coordinate
    transformation, with minima included and maxima excluded. Its output grid
    remains XY, even when z bounds restrict which points are counted.

    Examples
    --------
    >>> bounds = SpatialBounds(x=(0, 1000), y=(200, 800))
    >>> bounds.as_tuple()
    (0.0, 1000.0, 200.0, 800.0)
    >>> SpatialBounds(x=(0, 1000), y=(200, 800), z=(9, 11)).as_tuple()
    (0.0, 1000.0, 200.0, 800.0, 9.0, 11.0)
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            bounds = getattr(self, axis)
            if axis == "z" and bounds is None:
                continue
            message = f"Spatial bounds for axis {axis!r} must be two finite numbers with minimum < maximum."
            if isinstance(bounds, (str, bytes)):
                raise ValueError(message)
            try:
                lower, upper = bounds
                lower, upper = float(lower), float(upper)
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(message) from e
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError(message)
            object.__setattr__(self, axis, (lower, upper))

    def as_tuple(self) -> tuple[float, ...]:
        """Return ``(xmin, xmax, ymin, ymax)`` with ``zmin, zmax`` appended when present."""
        return (*self.x, *self.y, *(self.z if self.z is not None else ()))


def _normalize_spatial_bounds(crd: SpatialBounds | tuple[float, ...] | None) -> SpatialBounds | None:
    """Normalize four/six-value extents to validated, named axis bounds."""
    if crd is None:
        return None
    if isinstance(crd, SpatialBounds):
        return crd
    try:
        if isinstance(crd, (str, bytes)):
            raise ValueError("String extents are not supported.")
        values = tuple(crd)
        if len(values) not in (4, 6):
            raise ValueError("Expected four XY bounds or six XYZ bounds.")
        return SpatialBounds(x=values[:2], y=values[2:4], z=values[4:] if len(values) == 6 else None)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid crd: expected SpatialBounds or (xmin, xmax, ymin, ymax[, zmin, zmax]). {e}") from e
