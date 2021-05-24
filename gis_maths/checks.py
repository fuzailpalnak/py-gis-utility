def raise_value_assertion(values):
    raise AssertionError(
        "Expected values to be either '[n_line_segments, 1, 1]' or '[1, 1]'"
        "got %s, %s",
        (
            values.ndim,
            values.shape,
        ),
    )


def raise_point_assertion(points):
    raise AssertionError(
        "Expected points to be either '[n_line_segments, 1, 2]' or '[1, 2]'"
        "got %s, %s",
        (
            points.ndim,
            points.shape,
        ),
    )


def raise_line_segment_assertion(line_segments):
    raise AssertionError(
        "Expected line segment coordinates to be either '[n_line_segments, 2, 2]' or '[2, 2]'"
        "got %s, %s",
        (
            line_segments.ndim,
            line_segments.shape,
        ),
    )


def is_points_structure(points):
    return (
        True
        if (points.ndim == 2 or points.ndim == 3)
        and ((points.shape[-2], points.shape[-1]) == (1, 2))
        else False
    )


def is_value_structure(values):
    return (
        True
        if (values.ndim == 2 or values.ndim == 3)
        and ((values.shape[-2], values.shape[-1]) == (1, 1))
        else False
    )


def is_line_segment_structure(line_segments):
    return (
        True
        if (line_segments.ndim == 2 or line_segments.ndim == 3)
        and ((line_segments.shape[-2], line_segments.shape[-1]) == (2, 2))
        else False
    )
