import numpy as np


def extract_index(
    from_input: np.ndarray, value: np.ndarray, along_axis: int = 0
) -> np.ndarray:
    """

    :param along_axis:
    :param from_input:
    :param value:
    :return:
    """

    assert (
        type(from_input) is np.ndarray and type(value) is np.ndarray
    ), f"Expected to have input type 'np.ndarray' got {type(from_input), type(value)}"
    assert from_input.ndim == 3 and (from_input.shape[-2], from_input.shape[-1]) == (
        2,
        2,
    ), f"Expected from_input coordinates to be either '[n_from_input, 2, 2]' got {from_input.ndim, from_input.shape}"

    assert (
        0 <= along_axis <= from_input.ndim
    ), f"Expected along_axis to be in range ['0' and '%s'] got {from_input.ndim, along_axis}"

    if value.ndim == 2 and from_input.ndim == 3:
        value = value[np.newaxis, :, :]

    assert (
        from_input.ndim == value.ndim
    ), f"Expected to have same number of dimensions got {from_input.ndim, value.ndim}"

    return np.unique(np.where(from_input == value)[along_axis])


def get_dimension(input_array: np.ndarray) -> int:
    return input_array.ndim


def get_coordinate_structure(input_array: np.ndarray) -> tuple:
    return input_array.shape[-2], input_array.shape[-1]


def is_input_dimension_constraint_satisfied(n_dim: int) -> bool:
    return True if (n_dim == 2 or n_dim == 3) else False


def is_point_structure_constraint_satisfied(input_shape: tuple) -> bool:
    return True if (input_shape == (1, 2)) else False


def is_value_structure_constraint_satisfied(input_shape: tuple) -> bool:
    return True if (input_shape == (1, 1)) else False


def is_line_segment_structure_constraint_satisfied(input_shape: tuple) -> bool:
    return True if (input_shape == (2, 2)) else False


def is_input_a_point(points):
    return (
        True
        if (
            is_input_dimension_constraint_satisfied(get_dimension(points))
            and is_point_structure_constraint_satisfied(
                get_coordinate_structure(points)
            )
        )
        else False
    )


def is_input_a_value(values):
    return (
        True
        if (
            is_input_dimension_constraint_satisfied(get_dimension(values))
            and is_value_structure_constraint_satisfied(
                get_coordinate_structure(values)
            )
        )
        else False
    )


def is_input_a_line_segment(line_segments):
    return (
        True
        if (
            is_input_dimension_constraint_satisfied(get_dimension(line_segments))
            and is_line_segment_structure_constraint_satisfied(
                get_coordinate_structure(line_segments)
            )
        )
        else False
    )


def get_point_assertion_message(points: np.ndarray):
    return (
        f"Expected points to be either '[n_points, 1, 2]' for n_dim == 3 or '[1, 2]' for n_dim == 2,"
        f"got {points.shape} for n_dim == {points.ndim}"
    )


def get_value_assertion_message(values: np.ndarray):
    return (
        f"Expected values to be either '[n_values, 1, 1]' for n_dim == 3 or '[1, 1]' for n_dim == 2, "
        f"got {values.shape} for n_dim == {values.ndim}"
    )


def get_line_segment_assertion_message(line_segments: np.ndarray):
    return (
        f"Expected line segments to be either '[n_values, 2, 2]' for n_dim == 3 or '[2, 2]' for n_dim == 2, "
        f"got {line_segments.shape} for n_dim == {line_segments.ndim}"
    )
