from typing import Union

import affine
import geopandas
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from rasterio.io import BufferedDatasetWriter, DatasetWriter
from shapely.geometry import mapping
from stitch_n_split.geo_info import get_affine_transform
from stitch_n_split.split.mesh import (
    mesh_from_geo_transform,
    ImageNonOverLapMesh,
    ImageOverLapMesh,
)


def read_data_frame(path: str) -> GeoDataFrame:
    return geopandas.read_file(path)


def create_mesh(
    mesh_bounds: tuple,
    grid_size: tuple,
    pixel_resolution: tuple,
    is_overlap: bool = False,
) -> Union[ImageNonOverLapMesh, ImageOverLapMesh]:
    """

    :param mesh_bounds:
    :param grid_size:
    :param pixel_resolution:
    :param is_overlap: if set to true will find overlapping grid if any
    :return:
    """
    assert len(mesh_bounds) == 4, (
        f"Expected mesh_bounds to be in format (minx, miny, maxx, maxy) but got "
        f"{mesh_bounds} of size {len(mesh_bounds)}"
    )

    assert len(grid_size) == 2, (
        f"Expected grid_size to be in format (h x w) but got "
        f"{grid_size} of size {len(grid_size)}"
    )

    assert (
        len(pixel_resolution) == 2
    ), f"Expected pixel_resolution to have size 2 but got {len(grid_size)}"

    mesh = mesh_from_geo_transform(
        grid_size=grid_size,
        transform=get_affine_transform(
            mesh_bounds[0], mesh_bounds[-1], *pixel_resolution
        ),
        mesh_bounds=mesh_bounds,
        overlap=is_overlap,
    )
    return mesh


def extract_geometry_from_data_frame_row(row: GeoDataFrame) -> list:
    if "geometry" not in list(row.keys()):
        raise KeyError("Missing Keys, Must have keys ['geometry']")

    feature_geometry = list()

    for geometry in list(mapping(row["geometry"])["features"]):
        feature_geometry.append(geometry["geometry"])

    return feature_geometry


def save_image_with_geo_transform(
    save_path: str, image: np.ndarray, transform: affine.Affine
):
    """

    :param save_path:
    :param image:
    :param transform:
    :return:
    """

    bands = 1 if image.ndim == 2 else image.shape[-1]
    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        dtype=rasterio.uint8,
        count=bands,
        width=image.shape[0],
        height=image.shape[1],
        transform=transform,
    ) as dst:
        dst.write(image, indexes=1)


def read_image_with_geo_transform(
    path: str,
) -> Union[BufferedDatasetWriter, DatasetWriter]:
    """

    :param path:
    :return:
    """
    return rasterio.open(path)


def convert_2d_input_to_3d_single_batch_format(input_array: np.ndarray) -> np.ndarray:
    input_array = input_array[np.newaxis, :, :]
    return input_array


def convert_1d_coordinates_to_2d_single_batch_format(
    input_coordinates: np.ndarray,
) -> np.ndarray:
    input_coordinates = input_coordinates[np.newaxis, :]
    return input_coordinates


def minimum_in_matrix(input_matrix: np.ndarray, find_minimum_in_axis=1):
    """

    :param input_matrix:
    :param find_minimum_in_axis:
    :return:
    """
    assert (
        input_matrix.shape[-1] == 2 and input_matrix.ndim == 2
    ), f"Expected input_coordinates to have shape '[number of points, 2]'got {input_matrix.shape}"

    assert find_minimum_in_axis in [
        1,
        2,
    ], f"Expected shortest_distance_axis to be in '[1, 2]' got {find_minimum_in_axis}"
    minimum = np.argmin(input_matrix, axis=find_minimum_in_axis)

    return minimum, input_matrix[minimum]


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


def is_input_3d(input_array: np.ndarray) -> bool:
    return True if get_dimension(input_array) == 3 else False


def is_input_2d(input_array: np.ndarray) -> bool:
    return True if get_dimension(input_array) == 2 else False


def is_line_segment(input_array: np.ndarray) -> bool:
    return True if (get_coordinate_structure(input_array) == (2, 2)) else False


def is_value(input_array: np.ndarray) -> bool:
    return True if (get_coordinate_structure(input_array) == (1, 1)) else False


def is_point(input_array: np.ndarray) -> bool:
    return True if (get_coordinate_structure(input_array) == (1, 2)) else False


def is_line_segment_3d(line_segments: np.ndarray) -> bool:
    return (
        True
        if (is_input_3d(line_segments) and is_line_segment(line_segments))
        else False
    )


def is_value_3d(values: np.ndarray) -> bool:
    return True if (is_input_3d(values) and is_value(values)) else False


def is_point_3d(points: np.ndarray) -> bool:
    return True if (is_input_3d(points) and is_point(points)) else False


def is_line_segment_2d(line_segments: np.ndarray) -> bool:
    return (
        True
        if (is_input_2d(line_segments) and is_line_segment(line_segments))
        else False
    )


def is_value_2d(values: np.ndarray) -> bool:
    return True if (is_input_2d(values) and is_value(values)) else False


def is_point_2d(points: np.ndarray) -> bool:
    return True if (is_input_2d(points) and is_point(points)) else False
