import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_euclidean_between_two_sets(
    coordinates_a: np.ndarray, coordinates_b: np.ndarray
) -> np.ndarray:
    """
    This function will return set of distances from coordinates_b to coordinates_a
    ex -
        coordinates_b = [[0, 0], [1, 1]]
        coordinates_a = [[2, 1], [3, 1], [2, 4], [5, 2]]

        return
            sets of distances from [0, 0] to all the coordinates present in coordinates_a
            sets of distances from [1, 1] to all the coordinates present in coordinates_a

    visually -
        | distance from [0, 0] to [2, 1]   | distance from [1, 1] to [2, 1] |
        | distance from [0, 0] to [3, 1]   | distance from [1, 1] to [3, 1] |
        | distance from [0, 0] to [2, 4]   | distance from [1, 1] to [2, 4] |
        | distance from [0, 0] to [5, 2]   | distance from [1, 1] to [5, 2] |

    resulting in shape of [n_coordinates_a, n_coordinates_b]

    :param coordinates_a: array of shape [n_coordinates_a, 2]
    :param coordinates_b: array of shape [n_coordinates_b, 2]
    :return: array of shape [n_coordinates_a, n_coordinates_b]
    """
    assert coordinates_b.shape[-1] == 2 and coordinates_b.ndim == 2, (
        "Expected orientation to have shape '[number of points, 2]'" "got %s",
        (coordinates_b.shape,),
    )

    assert coordinates_a.shape[-1] == 2 and coordinates_a.ndim == 2, (
        "Expected input_coordinates to have shape '[number of points, 2]'" "got %s",
        (coordinates_a.shape,),
    )

    distances = np.linalg.norm(
        np.concatenate(
            [coordinates_a[np.newaxis, :, :]] * coordinates_b.shape[0], axis=0
        )
        - coordinates_b[:, np.newaxis, :],
        axis=-1,
    ).T
    return distances


def compute_euclidean_between_self(coordinates_a: np.ndarray) -> np.ndarray:
    """
    This function will return set of distances from coordinates_a to coordinates_a
    ex -
        coordinates_a = [[0, 0], [1, 1]]

        return
            sets of distances from [0, 0] to all the coordinates present in coordinates_a
            sets of distances from [1, 1] to all the coordinates present in coordinates_a

    visually -
        | distance from [0, 0] to [0, 0]   | distance from [1, 1] to [0, 0] |
        | distance from [0, 0] to [1, 1]   | distance from [1, 1] to [1, 1] |

    resulting in shape of [n_coordinates_a, n_coordinates_a]

    :param coordinates_a: array of shape [n_coordinates_a, 2]
    :return: array of shape [n_coordinates_a, n_coordinates_a]
    """

    assert coordinates_a.shape[-1] == 2 and coordinates_a.ndim == 2, (
        "Expected input_coordinates to have shape '[number of points, 2]'" "got %s",
        (coordinates_a.shape,),
    )

    return squareform(pdist(coordinates_a))


def get_minimum_in_matrix(input_matrix: np.ndarray, find_minimum_in_axis=1):
    """

    :param input_matrix:
    :param find_minimum_in_axis:
    :return:
    """
    assert input_matrix.shape[-1] == 2 and input_matrix.ndim == 2, (
        "Expected input_coordinates to have shape '[number of points, 2]'" "got %s",
        (input_matrix.shape,),
    )

    assert find_minimum_in_axis in [1, 2], (
        "Expected shortest_distance_axis to be in '[1, 2]'" "got %s",
        (find_minimum_in_axis,),
    )
    minimum = np.argmin(input_matrix, axis=find_minimum_in_axis)

    return minimum, input_matrix[minimum]


def compute_perpendicular_distance_from_point_to_line_segment_in_2d(
    line_segment: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """
    The function will compute perpendicular distance for the coordinates value present in coordinates to the given
    input line segment

    :param line_segment: array of shape [number_of_line_segments, 2, 2] or [2, 2]

            If there is just one line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [2, 2] are [
                                                [start_line_segment_x, start_line_segment_y],
                                                [end_line_segment_x,   end_line_segment_y]
                                            ]
            If there is multiple  line segment to which perpendicular distances are to be computed then pass it as
            follows :
                -- the dimension [number_of_line_segments, 2, 2] are [
                                                                        [
                                                                        [start_line_segment_1_x, start_line_segment_1_y],
                                                                        [end_line_segment_1_x,   end_line_segment_1_y]
                                                                        ],
                                                                        [
                                                                        [start_line_segment_2_x, start_line_segment_2_y],
                                                                        [end_line_segment_2_x,   end_line_segment_2_y]
                                                                        ],
                                                                        ....
                                                                        ...
                                                                        [
                                                                        [start_line_segment_n_x, start_line_segment_n_y],
                                                                        [end_line_segment_n_x,   end_line_segment_n_y]
                                                                        ],
                                                                    ]
    :param coordinates: array of shape [number of points, 2]
    :return: array of shape [number of segments, number of points]

    if single line segment is passed for computation, i.e. input with dim [2, 2] then expect the output in
        format:
            | distance from line_Segment_1 to coordinate[0]    | distance from line_Segment_1 to coordinate[1] |

        if multiple line segments are passed for computation, i.e. input with dim [n_segments, 2, 2] then expect
         the output in format:
            | distance from line_Segment_1 to coordinate[0]    | distance from line_Segment_1 to coordinate[1] |
            | distance from line_Segment_2 to coordinate[0]    | distance from line_Segment_2 to coordinate[1] |
            ....
            ...
            | distance from line_Segment_n to coordinate[0]    | distance from line_Segment_n to coordinate[1] |




    """

    # https://stackoverflow.com/a/53176074/7984359
    # https://math.stackexchange.com/questions/1300484/distance-between-line-and-a-point
    # https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

    assert coordinates.ndim == 2 and coordinates.shape[-1] == 2, (
        "Expected coordinates to be '2' dimensional and have shape '[number of point, 2]'"
        "got %s, %s",
        (
            coordinates.ndim,
            coordinates.shape,
        ),
    )
    assert (
        line_segment.ndim == 2
        and (line_segment.shape[-2], line_segment.shape[-1]) == (2, 2)
    ) or (
        line_segment.ndim == 3
        and (line_segment.shape[-2], line_segment.shape[-1]) == (2, 2)
    ), (
        "Expected line segment coordinates to be either '[n_line_segments, 2, 2]' or '[2, 2]'"
        "got %s, %s",
        (
            line_segment.ndim,
            line_segment.shape,
        ),
    )

    if line_segment.ndim == 2:
        line_segment = line_segment[np.newaxis, :, :]

    coordinates_repeat_copy = np.concatenate(
        [coordinates[np.newaxis, :, :]] * line_segment.shape[0], axis=0
    )

    dp = line_segment[:, 1:2, :] - line_segment[:, 0:1, :]
    st = dp[:, :, 0:1] ** 2 + dp[:, :, 1:2] ** 2

    u = (
        (coordinates_repeat_copy[:, :, 0:1] - line_segment[:, 0:1, 0:1]) * dp[:, :, 0:1]
        + (coordinates_repeat_copy[:, :, 1:2] - line_segment[:, 0:1, 1:2])
        * dp[:, :, 1:2]
    ) / st

    u[u > 1.0] = 1.0
    u[u < 0.0] = 0.0

    dx = (line_segment[:, 0:1, 0:1] + u * dp[:, :, 0:1]) - coordinates_repeat_copy[
        :, :, 0:1
    ]
    dy = (line_segment[:, 0:1, 1:2] + u * dp[:, :, 1:2]) - coordinates_repeat_copy[
        :, :, 1:2
    ]

    return np.squeeze(np.sqrt(dx ** 2 + dy ** 2), axis=-1)


def compute_perpendicular_point_to_line_segment(
    line_segment: np.ndarray, distance_from_the_line: int = 10
):
    """

    :param line_segment:
    :param distance_from_the_line:
    :return:
    """

    assert (
        line_segment.ndim == 2
        and (line_segment.shape[-2], line_segment.shape[-1]) == (2, 2)
    ) or (
        line_segment.ndim == 3
        and (line_segment.shape[-2], line_segment.shape[-1]) == (2, 2)
    ), (
        "Expected line segment coordinates to be either '[n_line_segments, 2, 2]' or '[2, 2]'"
        "got %s, %s",
        (
            line_segment.ndim,
            line_segment.shape,
        ),
    )

    assert type(distance_from_the_line) is int and distance_from_the_line > 0, (
        "Expected distance_from_the_line to be of type 'int' and 'non zero'"
        "got %s, %s",
        (
            type(distance_from_the_line),
            distance_from_the_line,
        ),
    )
    if line_segment.ndim == 2:
        line_segment = line_segment[np.newaxis, :, :]

    x1, y1 = line_segment[:, 0:1, 0:1], line_segment[:, 0:1, 1:2]
    x2, y2 = line_segment[:, 1:2, 0:1], line_segment[:, 1:2, 1:2]

    dx = x1 - x2
    dy = y1 - y2

    dist = np.linalg.norm(np.array([dx, dy]), axis=0)

    x_perpendicular = distance_from_the_line * (dx / dist)
    y_perpendicular = distance_from_the_line * (dy / dist)

    point_with_start_as_reference = [
        [
            np.array(x1 + y_perpendicular).squeeze(axis=-1),
            np.array(y1 - x_perpendicular).squeeze(axis=-1),
        ],
        [
            np.array(x1 - y_perpendicular).squeeze(axis=-1),
            np.array(y1 + x_perpendicular).squeeze(axis=-1),
        ],
    ]

    point_with_end_as_reference = [
        [
            np.array(x2 - y_perpendicular).squeeze(axis=-1),
            np.array(y2 + x_perpendicular).squeeze(axis=-1),
        ],
        [
            np.array(x2 + y_perpendicular).squeeze(axis=-1),
            np.array(y2 - x_perpendicular).squeeze(axis=-1),
        ],
    ]

    perpendicular_with_start = (
        np.dstack(
            (
                np.array(point_with_start_as_reference[0]).squeeze(axis=-1),
                np.array(point_with_start_as_reference[1]).squeeze(axis=-1),
            )
        )
        .swapaxes(-1, 1)
        .T
    )

    perpendicular_with_end = (
        np.dstack(
            (
                np.array(point_with_end_as_reference[0]).squeeze(axis=-1),
                np.array(point_with_end_as_reference[1]).squeeze(axis=-1),
            )
        )
        .swapaxes(-1, 1)
        .T
    )
    return perpendicular_with_start, perpendicular_with_end

