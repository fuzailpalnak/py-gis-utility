from typing import List, Dict

from shapely.geometry import shape

from py_gis_utility.helper import read_image_with_geo_transform
from py_gis_utility.image_utils.image_ops import convert_image_to_collection


def from_image_to_polygon_geometries(
    image_path: str, **kwargs
) -> List[Dict[shape, Dict]]:
    """

    :param image_path:
    :return:
    """
    image_info = read_image_with_geo_transform(image_path)
    return convert_image_to_collection(
        image_info.read(),
        image_info.transform,
        is_shape=False,
        crs=image_info.crs,
        **kwargs
    )


def from_image_to_polygon_coordinates(image_path: str, **kwargs) -> List[Dict]:
    """

    :param image_path:
    :return:
    """
    image_info = read_image_with_geo_transform(image_path)
    return convert_image_to_collection(
        image_info.read(),
        image_info.transform,
        is_shape=True,
        crs=image_info.crs,
        **kwargs
    )
