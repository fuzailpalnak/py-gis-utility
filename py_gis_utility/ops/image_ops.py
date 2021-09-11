from dataclasses import dataclass
from typing import List, Dict, Union

import affine
import rasterio
import numpy as np

from rasterio.features import shapes
from rasterio.io import BufferedDatasetWriter, DatasetWriter
from shapely.geometry import shape
from stitch_n_split.geo_info import get_affine_transform, get_pixel_resolution
from stitch_n_split.split.mesh import ImageNonOverLapMesh, ImageOverLapMesh


@dataclass
class Bitmap:
    array: np.ndarray
    transform: affine.Affine


def create_bitmap(
    mesh: Union[ImageNonOverLapMesh, ImageOverLapMesh], geometry_collection: list
):
    """

    :param mesh:
    :param geometry_collection:
    :return:
    """
    intermediate_bitmap = np.empty((0,))
    intermediate_bitmap = np.concatenate(
        (intermediate_bitmap, geometry_collection), axis=0
    )

    for grid in mesh.extent():
        transform = get_affine_transform(
            grid["extent"][0],
            grid["extent"][-1],
            *get_pixel_resolution(mesh.mesh_transform),
        )
        bitmap_array = rasterio.features.rasterize(
            ((g, 255) for g in intermediate_bitmap),
            out_shape=mesh.grid_size,
            transform=transform,
        )
        yield Bitmap(bitmap_array, transform)


def image_to_collection_generator(
    image: np.ndarray, transform: affine.Affine, is_shape=False, **kwargs
) -> Dict:
    """

    :param is_shape:
    :param transform:
    :param image:
    :return:
    """
    for i, (s, v) in enumerate(
        shapes(
            image.astype(rasterio.uint8),
            mask=None,
            connectivity=8,
            transform=transform,
        )
    ):
        yield {
            "geometry": shape(s) if is_shape else s,
            "properties": {"id": v, **kwargs},
        }


def copy_geo_reference_to_image(
    copy_from: Union[BufferedDatasetWriter, DatasetWriter],
    copy_to: np.ndarray,
    save_to: str,
):
    """

    :param copy_from:
    :param copy_to:
    :param save_to:
    :return:
    """
    bands = copy_to.ndim if copy_to.ndim > 2 else 1
    geo_referenced_image = rasterio.open(
        save_to,
        mode="w",
        driver=copy_from.driver,
        width=copy_from.width,
        height=copy_from.height,
        crs=copy_from.crs,
        transform=copy_from.transform,
        dtype=copy_to.dtype,
        count=bands,
    )
    if bands > 2:
        for band in range(copy_to.shape[2]):
            geo_referenced_image.write(copy_to[:, :, band], band + 1)
    else:
        geo_referenced_image.write(copy_to, 1)

    copy_from.close()
    geo_referenced_image.close()
