from dataclasses import dataclass

import numpy as np
from dask.dataframe import DataFrame
from spatialdata.models import get_axes_names
from spatialdata.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
)
from xarray import DataArray


@dataclass(frozen=True)
class _PointToLabelsTransform:
    """Immutable affine mapping from intrinsic points to intrinsic labels coordinates."""

    point_axes: tuple[str, ...]
    labels_axes: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.point_axes != self.labels_axes or self.point_axes not in {("x", "y"), ("x", "y", "z")}:
            raise ValueError(
                "Point-to-label transformations require matching ('x', 'y') or ('x', 'y', 'z') axes, "
                f"found points axes {self.point_axes!r} and labels axes {self.labels_axes!r}."
            )
        matrix = np.asarray(self.matrix, dtype=np.float64)
        expected_shape = (len(self.point_axes) + 1, len(self.point_axes) + 1)
        if matrix.shape != expected_shape:
            raise ValueError(f"Point-to-label affine matrix must have shape {expected_shape}, found {matrix.shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError("Point-to-label affine matrix must contain only finite values.")
        expected_last_row = np.zeros(expected_shape[1], dtype=np.float64)
        expected_last_row[-1] = 1.0
        if not np.allclose(matrix[-1], expected_last_row):
            raise ValueError("Point-to-label affine matrix must use homogeneous coordinates.")
        try:
            np.linalg.solve(matrix[:-1, :-1], np.eye(len(self.point_axes)))
        except np.linalg.LinAlgError as e:
            raise ValueError("Point-to-label affine matrix must be invertible.") from e

    @property
    def affine_matrix(self) -> np.ndarray:
        """Return the small homogeneous matrix as a NumPy array."""
        return np.asarray(self.matrix, dtype=np.float64)


def _resolve_point_to_labels_transform(
    points: DataFrame,
    labels: DataArray,
    *,
    to_coordinate_system: str,
) -> _PointToLabelsTransform:
    """Resolve an affine map from intrinsic point coordinates to labels pixels.

    The points and labels transformations both map into
    ``to_coordinate_system``. If their homogeneous matrices are ``M_points``
    and ``M_labels``, respectively, the returned matrix is calculated as
    ``solve(M_labels, M_points)``. Applying it therefore maps source point
    coordinates directly into the intrinsic pixel frame of ``labels`` without
    resampling the labels raster.
    """
    point_axes = tuple(get_axes_names(points))
    labels_dimensions = tuple(labels.dims)
    if labels_dimensions == ("y", "x"):
        labels_axes = ("x", "y")
    elif labels_dimensions == ("z", "y", "x"):
        labels_axes = ("x", "y", "z")
    else:
        raise ValueError(f"Labels dimensions must be ('y', 'x') or ('z', 'y', 'x'), found {labels_dimensions!r}.")
    if point_axes != labels_axes:
        raise ValueError(
            "Points and labels must use the same spatial dimensionality, "
            f"found points axes {point_axes!r} and labels dimensions {labels_dimensions!r}."
        )

    try:
        point_transformation = get_transformation(points, to_coordinate_system=to_coordinate_system)
    except ValueError as e:
        raise ValueError(f"Points element does not define coordinate system {to_coordinate_system!r}.") from e
    try:
        labels_transformation = get_transformation(labels, to_coordinate_system=to_coordinate_system)
    except ValueError as e:
        raise ValueError(f"Labels element does not define coordinate system {to_coordinate_system!r}.") from e

    point_matrix = _invertible_affine_matrix(
        point_transformation,
        axes=point_axes,
        element_kind="Points",
    )
    labels_matrix = _invertible_affine_matrix(
        labels_transformation,
        axes=labels_axes,
        element_kind="Labels",
    )
    try:
        point_to_labels = np.linalg.solve(labels_matrix, point_matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Labels transformation to coordinate system {to_coordinate_system!r} must be invertible."
        ) from e
    if not np.isfinite(point_to_labels).all():
        raise ValueError(
            f"Point-to-label transformation through coordinate system {to_coordinate_system!r} "
            "contains non-finite values."
        )

    return _PointToLabelsTransform(
        point_axes=point_axes,
        labels_axes=labels_axes,
        matrix=tuple(tuple(float(value) for value in row) for row in point_to_labels),
    )


def _invertible_affine_matrix(
    transformation: BaseTransformation,
    *,
    axes: tuple[str, ...],
    element_kind: str,
) -> np.ndarray:
    """Return and structurally validate one element-to-coordinate-system matrix."""
    _validate_same_dimensional_axes(transformation, axes=axes, element_kind=element_kind)
    try:
        matrix = np.asarray(
            transformation.to_affine_matrix(input_axes=axes, output_axes=axes),
            dtype=np.float64,
        )
    except (AssertionError, TypeError, ValueError) as e:
        raise ValueError(
            f"{element_kind} transformation cannot be represented as a same-dimensional affine matrix "
            f"for axes {axes!r}."
        ) from e

    expected_shape = (len(axes) + 1, len(axes) + 1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{element_kind} transformation matrix must have shape {expected_shape}, found {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{element_kind} transformation matrix must contain only finite values.")
    expected_last_row = np.zeros(expected_shape[1], dtype=np.float64)
    expected_last_row[-1] = 1.0
    if not np.allclose(matrix[-1], expected_last_row):
        raise ValueError(f"{element_kind} transformation matrix must use homogeneous coordinates.")
    try:
        np.linalg.solve(matrix[:-1, :-1], np.eye(len(axes)))
    except np.linalg.LinAlgError as e:
        raise ValueError(f"{element_kind} transformation matrix must be invertible.") from e
    return matrix


def _validate_same_dimensional_axes(
    transformation: BaseTransformation,
    *,
    axes: tuple[str, ...],
    element_kind: str,
) -> None:
    """Reject transformation components that introduce or remove spatial axes."""
    expected = set(axes)
    if isinstance(transformation, Identity):
        return
    if isinstance(transformation, (Scale, Translation)):
        valid = set(transformation.axes).issubset(expected)
    elif isinstance(transformation, Affine):
        valid = set(transformation.input_axes) == expected and set(transformation.output_axes) == expected
    elif isinstance(transformation, MapAxis):
        mapped = tuple(transformation.map_axis.get(axis, axis) for axis in axes)
        valid = (
            set(transformation.map_axis).issubset(expected)
            and set(mapped) == expected
            and len(set(mapped)) == len(axes)
        )
    elif isinstance(transformation, Sequence):
        for component in transformation.transformations:
            _validate_same_dimensional_axes(component, axes=axes, element_kind=element_kind)
        return
    else:
        valid = False
    if not valid:
        raise ValueError(
            f"{element_kind} transformation must preserve spatial axes {axes!r}; "
            "dimension-adding or dimension-dropping mappings are not supported."
        )


def _get_translation_values(translation: Affine | Sequence | Translation | Identity) -> tuple[float | int, float | int]:
    transform_matrix = translation.to_affine_matrix(
        input_axes=(
            "c",
            "z",
            "x",
            "y",
        ),
        output_axes=("c", "z", "x", "y"),
    )

    assert (
        transform_matrix.shape == (5, 5)
        and np.array_equal(transform_matrix[:-1, :-1], np.eye(4))  # no scaling or rotation
        and np.array_equal(transform_matrix[-1], np.array([0, 0, 0, 0, 1]))  # maintaining homogeneity
        and np.array_equal(transform_matrix[:2, -1], np.array([0, 0]))  # no translation allowed in z and c
    ), f"The provided transform matrix {transform_matrix} represents more than just a translation in 'y' and 'x'."

    return tuple(transform_matrix[2:4:, -1])


def _identity_check_transformations_points(ddf: DataFrame, to_coordinate_system: str = "global"):
    """Check that the points element has no transformations associated with it other than an Identity transformation in `to_coordinate_system`."""
    transformations = get_transformation(ddf, get_all=True)

    if to_coordinate_system not in [*transformations]:
        raise ValueError(
            f"Coordinate system '{to_coordinate_system}' does not appear to be a coordinate system of the spatial element. "
            f"Please choose a coordinate system from this list: {[*transformations]}."
        )
    transformation = transformations[to_coordinate_system]

    if not isinstance(transformation, Identity):
        raise ValueError(
            f"Currently we do not provide support for defining transformations "
            f"other than the Identity transformation on a points element for coordinate system {to_coordinate_system}."
        )
