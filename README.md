# py-gis-utility

A GIS utility library which contains some regularly required math and image operations.

[![Downloads](https://static.pepy.tech/personalized-badge/py-gis-utility?period=month&units=international_system&left_color=yellowgreen&right_color=green&left_text=Downloads)](https://pepy.tech/project/py-gis-utility)

## Installation
    
    pip install py-gis-utility
    
    
## Requirements

- *_Geopandas - [installation](https://anaconda.org/conda-forge/geopandas)_*
- *_Rasterio - [installation](https://anaconda.org/conda-forge/rasterio)_*
- *_GDAL 2.4.4 - [installation](https://anaconda.org/conda-forge/gdal)_*
- *_Fiona -  [installation](https://anaconda.org/conda-forge/fiona)_*
- *_Shapely -  [installation](https://anaconda.org/conda-forge/shapely)_*

 ## Math Operations
 
1. Get perpendicular point with reference to start and end point of the segment 
2. Get perpendicular distance from point to line_segment
3. Given a Point find a new point at an given 'angle' with given 'distance'
4. Calculate a new point on the line segment given the distance from the start
5. Euclidean computation

## Image Operations
- ### Save Multi Band Imagery
```python
import numpy as np
from affine import Affine
from py_gis_utility.image_func import save_16bit_multi_band, save_8bit_multi_band

image = np.zeros((512, 512, 6))
transform = Affine(1.0, 0.0, 3422098.682455578,
       0.0, -1.0, 5289611.291479621)

# Save 8bit
save_8bit_multi_band(image, transform, 26910, r"8bit.tiff")

# Save 16bit
save_16bit_multi_band(image, transform, 26910, r"16bit.tiff")

```

- ### Generate bitmap from shape file

![Animation](https://user-images.githubusercontent.com/24665570/132937989-0a77de62-2c55-4369-a155-35326b21c82d.gif)

```python
from py_gis_utility.helper import (
    read_data_frame,
    save_image_with_geo_transform,
)
from py_gis_utility.image_func import shape_geometry_to_bitmap_from_data_frame_generator

data_frame = read_data_frame(r"path_to_geometry_file")
bitmap_gen = shape_geometry_to_bitmap_from_data_frame_generator(data_frame, (50, 50), (1, 1),
 allow_output_to_overlap=True)

for i, bitmap in enumerate(bitmap_gen):
    save_image_with_geo_transform(f"{i}.tiff", bitmap.array, bitmap.transform)
```

- ### Generate shape geometry from geo reference bitmap

```python

from py_gis_utility.helper import (read_image_with_geo_transform,
)
from py_gis_utility.image_func import image_obj_to_coordinates_generator, image_obj_to_shape_generator


img_obj = read_image_with_geo_transform(r"path_to_geo_referenced_file")

# output in format {'geometry': <shapely.geometry.polygon.Polygon object at 0x0000022009E5EC08>, 'properties': {'id': 255.0, 'crs': CRS.from_epsg(4326)}}
shape_gen = image_obj_to_shape_generator(img_obj)
for g in shape_gen:
    print(g)

# output in format {'geometry': {'type': 'Polygon', 'coordinates': [[(621000.0, 3349500.0), .... ,(621000.0, 3349489.5)]]}, 'properties': {'id': 255.0, 'crs': CRS.from_epsg(4326)}}
co_ord_gen = image_obj_to_coordinates_generator(img_obj)
for g in co_ord_gen:
    print(g)
```

## Support Me

<a href='https://ko-fi.com/fuzailpalnak' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>



        
