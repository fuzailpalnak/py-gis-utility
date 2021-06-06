from typing import Union

import numpy as np
from pandas import DataFrame

from gis_maths.compute.functions import (
    new_perpendicular_point_to_line_segment,
    new_coordinate_based_on_angle_and_distance,
    new_point_after_certain_distance,
    euclidean,
    euclidean_between_two_sets,
    perpendicular_distance_from_point_to_line_segment_in_2d,
)
from gis_maths.compute.utils import (
    is_line_segment_3d,
    is_point_3d,
    create_frame_with_columns,
)


def get_new_perpendicular_point_with_common_distance_all_to_line_segment(
    line_segments: np.ndarray,
    distance_from_the_line: Union[int, float] = 10,
) -> DataFrame:
    """
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

    (
        points_at_start_of_segment,
        points_at_the_end_of_segment,
    ) = new_perpendicular_point_to_line_segment(
        line_segments, common_distance_from_the_line
    )

    output_data_frame = create_frame_with_columns(
        line_segments=line_segments.tolist(),
        distance_computed=distance_from_the_line,
        start=points_at_start_of_segment.tolist(),
        end=points_at_the_end_of_segment.tolist(),
    )

    return output_data_frame


def get_new_perpendicular_point_with_custom_distance_to_every_line_segment(
    line_segments: np.ndarray, distance_from_the_line: np.ndarray
) -> DataFrame:
    """
    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_the_line: how far the new point to create from the reference
    :return:

    """
    (
        points_at_start_of_segment,
        points_at_the_end_of_segment,
    ) = new_perpendicular_point_to_line_segment(line_segments, distance_from_the_line)

    output_data_frame = create_frame_with_columns(
        line_segments=line_segments.tolist(),
        distance_computed=distance_from_the_line.tolist(),
        start=points_at_start_of_segment.tolist(),
        end=points_at_the_end_of_segment.tolist(),
    )

    return output_data_frame


def get_new_coordinates_with_custom_angle_and_distance_for_every_point(
    points: np.ndarray, angle_in_degree: np.ndarray, distance: np.ndarray
) -> DataFrame:
    """
    :param points: array of shape [number_of_line_segments, 1, 2]
    :param angle_in_degree: array of shape [number_of_points, 1, 1]
    :param distance: array of shape [number_of_points, 1, 1]
    :return:
    """
    new_coordinates = new_coordinate_based_on_angle_and_distance(
        points, angle_in_degree, distance
    )
    return create_frame_with_columns(
        points=points.tolist(),
        angle_in_degree=angle_in_degree.tolist(),
        distance=distance.tolist(),
        new_coordinates=new_coordinates.tolist(),
    )


def get_new_coordinates_with_common_angle_and_distance_to_all_points(
    points: np.ndarray, angle_in_degree: Union[float, int], distance: Union[float, int]
) -> DataFrame:
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

    new_coordinates = new_coordinate_based_on_angle_and_distance(
        points,
        (np.ones((points.shape[0], 1, 1)) * angle_in_degree),
        (np.ones((points.shape[0], 1, 1)) * distance),
    )
    return create_frame_with_columns(
        points=points.tolist(),
        angle_in_degree=angle_in_degree,
        distance=distance,
        new_coordinates=new_coordinates.tolist(),
    )


def get_points_with_custom_distance_for_every_line_segments(
    line_segments: np.ndarray, distance_from_start: np.ndarray
) -> DataFrame:
    """

    :param line_segments: array of shape [number_of_line_segments, 2, 2]
    :param distance_from_start: array specifying the distance to compute the point
            shape [number_of_line_segments, 1, 1]
    :return:
    """
    new_points = new_point_after_certain_distance(line_segments, distance_from_start)
    return create_frame_with_columns(
        line_segments=line_segments.tolist(),
        distance_from_start=distance_from_start.tolist(),
        points=new_points.tolist(),
    )


def get_new_points_with_same_distance_for_all_line_segments(
    line_segments: np.ndarray, distance_from_start: Union[float, int]
) -> DataFrame:
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

    new_points = new_point_after_certain_distance(
        line_segments, common_distance_from_start
    )
    return create_frame_with_columns(
        line_segments=line_segments.tolist(),
        distance_from_start=common_distance_from_start.tolist(),
        points=new_points.tolist(),
    )


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


def get_perpendicular_distance_from_point_to_line_segment_in_2d(
    line_segment: np.ndarray, coordinates: np.ndarray
) -> DataFrame:
    distances = perpendicular_distance_from_point_to_line_segment_in_2d(
        line_segment, coordinates
    )
    return create_frame_with_columns(
        line_segment=line_segment.tolist(),
        coordinates=coordinates.tolist(),
        distances=distances.tolist(),
    )
