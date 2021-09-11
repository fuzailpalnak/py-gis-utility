import cv2
from typing import Dict, Union

from geopandas import GeoDataFrame
from rasterio.io import BufferedDatasetWriter, DatasetWriter

from py_gis_utility.helper import (
    read_image_with_geo_transform,
    extract_geometry_from_data_frame_row,
    read_data_frame,
    create_mesh,
)
from py_gis_utility.ops.image_ops import (
    image_to_collection_generator,
    copy_geo_reference_to_image,
    create_bitmap,
)


def image_obj_to_coordinates_generator(
    image: Union[BufferedDatasetWriter, DatasetWriter], **kwargs
) -> Dict:
    """

    :param image:
    :param kwargs:
    :return:
    """
    return image_to_collection_generator(
        image.read(), image.transform, is_shape=False, crs=image.crs, **kwargs
    )


def image_path_to_coordinates_generator(image_path: str, **kwargs) -> Dict:
    """

    :param image_path:
    :return:
    """
    return image_obj_to_coordinates_generator(
        read_image_with_geo_transform(image_path), **kwargs
    )


def image_obj_to_shape_generator(
    image: Union[BufferedDatasetWriter, DatasetWriter], **kwargs
) -> Dict:
    """

    :param image:
    :return:
    """
    return image_to_collection_generator(
        image.read(), image.transform, is_shape=True, crs=image.crs, **kwargs
    )


def image_path_to_shape_generator(image_path: str, **kwargs) -> Dict:
    """

    :param image_path:
    :return:
    """
    return image_obj_to_shape_generator(
        read_image_with_geo_transform(image_path), **kwargs
    )


def copy_geo_reference(copy_from_path: str, copy_to_path: str, save_to: str):
    """

    :param save_to:
    :param copy_from_path:
    :param copy_to_path:
    :return:
    """
    copy_geo_reference_to_image(
        read_image_with_geo_transform(copy_from_path),
        cv2.cvtColor(cv2.imread(copy_to_path), cv2.COLOR_BGR2RGB),
        save_to,
    )


def shape_geometry_to_bitmap_from_data_frame_generator(
    data_frame: GeoDataFrame,
    output_image_size: tuple,
    pixel_resolution: tuple,
    allow_output_to_overlap: bool = True,
):
    """

    :param data_frame:
    :param output_image_size:
    :param pixel_resolution:
    :param allow_output_to_overlap:
    :return:
    """
    bitmap_generator = create_bitmap(
        create_mesh(
            data_frame.total_bounds,
            output_image_size,
            pixel_resolution,
            is_overlap=allow_output_to_overlap,
        ),
        list(extract_geometry_from_data_frame_row(data_frame)),
    )

    return bitmap_generator


def shape_geometry_to_bitmap_from_shape_geometry_path_generator(
    shape_geometry_file_path: str,
    output_image_size: tuple,
    pixel_resolution: tuple,
    allow_output_to_overlap: bool = True,
):
    """

    :param shape_geometry_file_path:
    :param output_image_size:
    :param pixel_resolution:
    :param allow_output_to_overlap:
    :return:
    """
    return shape_geometry_to_bitmap_from_data_frame_generator(
        read_data_frame(shape_geometry_file_path),
        output_image_size,
        pixel_resolution,
        allow_output_to_overlap,
    )
