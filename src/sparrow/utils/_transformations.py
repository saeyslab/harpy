import numpy as np
from dask.dataframe import DataFrame
from spatialdata.transformations import Identity, Sequence, Translation, get_transformation


def _get_translation_values(translation: Sequence | Translation | Identity) -> tuple[float | int, float | int]:
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
    """Check that points layer has not other transformations associated that an Identity transformation."""
    transformations = get_transformation(ddf, get_all=True)

    if to_coordinate_system not in [*transformations]:
        raise ValueError(
            f"Coordinate system '{to_coordinate_system}' does not appear to be a coordinate system of the spatial element. "
            f"Please choose a coordinate system from this list: {[*transformations]}."
        )
    transformation = transformations[to_coordinate_system]

    if not isinstance(transformation, Identity):
        raise ValueError(
            "Currently we do not provide support for defining transformations other than the Identity transformation on a points layer."
        )
