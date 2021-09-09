from typing import Union

import numpy as np

from py_gis_utility.ops.math_ops import (
    new_perpendicular_point_to_line_segment,
    new_coordinate_based_on_angle_and_distance,
    new_point_after_certain_distance,
    euclidean,
    euclidean_between_two_sets,
)
from py_gis_utility.helper import is_line_segment_3d, is_point_3d


def get_new_perpendicular_point_with_common_distance_all_to_line_segment(
    line_segments: np.ndarray,
    distance_from_the_line: Union[int, float] = 10,
    return_pandas: bool = True,
):
    """
    :param return_pandas:
    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_the_line: how far the new point to create from the reference
    :return:

    """

    assert type(line_segments) is np.ndarray and type(distance_from_the_line) in [
        float,
        int,
    ], (
        f"Expected to have input type ['np.ndarray', '[int, float]'] "
        f"got {type(line_segments), type(distance_from_the_line)}"
    )

    assert is_line_segment_3d(line_segments), (
        f"Expected line segments to be either '[n_values, 2, 2]' for n_dim == 3 "
        f"got {line_segments.shape} for n_dim == {line_segments.ndim}"
    )

    assert (
        type(distance_from_the_line) in [float, int] and distance_from_the_line >= 0.0
    ), (
        "Expected distance_from_the_line to be of type 'float or int' and 'non zero'"
        f"got {type(distance_from_the_line), distance_from_the_line}"
    )

    common_distance_from_the_line = (
        np.ones((line_segments.shape[0], 1, 1)) * distance_from_the_line
    )

    return new_perpendicular_point_to_line_segment(
        line_segments, common_distance_from_the_line
    )


def get_new_perpendicular_point_with_custom_distance_to_every_line_segment(
    line_segments: np.ndarray, distance_from_the_line: np.ndarray
):
    """
    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_the_line: how far the new point to create from the reference
    :return:

    """
    return new_perpendicular_point_to_line_segment(
        line_segments, distance_from_the_line
    )


def get_new_coordinates_with_custom_angle_and_distance_for_every_point(
    points: np.ndarray, angle_in_degree: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """
    :param points: array of shape [number_of_line_segments, 1, 2]
    :param angle_in_degree: array of shape [number_of_points, 1, 1]
    :param distance: array of shape [number_of_points, 1, 1]
    :return:
    """
    return new_coordinate_based_on_angle_and_distance(points, angle_in_degree, distance)


def get_new_coordinates_with_common_angle_and_distance_to_all_points(
    points: np.ndarray, angle_in_degree: Union[float, int], distance: Union[float, int]
) -> np.ndarray:
    """
    :param points: array of shape [number_of_line_segments, 1, 2]
    :param angle_in_degree:
    :param distance:
    :return:
    """

    assert (
        type(points) is np.ndarray
        and type(distance) in [float, int]
        and type(angle_in_degree) in [float, int]
    ), (
        "Expected to have input type ['np.ndarray', '[float, int]', '[float, int]']"
        f"got {type(points), type(distance), type(angle_in_degree)}"
    )

    assert is_point_3d(points), (
        f"Expected points to be either '[n_points, 1, 2]' for n_dim == 3"
        f"got {points.shape} for n_dim == {points.ndim}"
    )

    return new_coordinate_based_on_angle_and_distance(
        points,
        (np.ones((points.shape[0], 1, 1)) * angle_in_degree),
        (np.ones((points.shape[0], 1, 1)) * distance),
    )


def get_points_with_custom_distance_for_every_line_segments(
    line_segments: np.ndarray, distance_from_start: np.ndarray
) -> np.ndarray:
    """

    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_start: array specifying the distance to compute the point
            shape [number_of_line_segments, 1, 1]
    :return:
    """
    return new_point_after_certain_distance(line_segments, distance_from_start)


def get_new_points_with_same_distance_for_all_line_segments(
    line_segments: np.ndarray, distance_from_start: Union[float, int]
) -> np.ndarray:
    """

    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_start:
    :return:
    """
    assert type(line_segments) is np.ndarray and type(distance_from_start) in [
        float,
        int,
    ], f"Expected to have input type ['np.ndarray', '[int, float]'] got {type(line_segments), type(distance_from_start)}"

    assert is_line_segment_3d(line_segments), (
        f"Expected line segments to be either '[n_values, 2, 2]' for n_dim == 3"
        f"got {line_segments.shape} for n_dim == {line_segments.ndim}"
    )

    assert type(distance_from_start) in [float, int] and distance_from_start >= 0.0, (
        "Expected distance_from_the_line to be of type 'float or int' and 'non zero'"
        f"got {type(distance_from_start), distance_from_start}"
    )

    common_distance_from_start = (
        np.ones((line_segments.shape[0], 1, 1)) * distance_from_start
    )

    return new_point_after_certain_distance(line_segments, common_distance_from_start)


def get_euclidean(coordinates_a: np.ndarray) -> np.ndarray:
    """

    :param coordinates_a:
    :return:
    """
    return euclidean(coordinates_a)


def get_euclidean_within_two(
    coordinates_a: np.ndarray, coordinates_b: np.ndarray
) -> np.ndarray:
    """

    :param coordinates_a:
    :param coordinates_b:
    :return:
    """
    return euclidean_between_two_sets(coordinates_a, coordinates_b)
