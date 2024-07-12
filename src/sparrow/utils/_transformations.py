import numpy as np
from spatialdata.transformations.transformations import Identity, Sequence, Translation


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
