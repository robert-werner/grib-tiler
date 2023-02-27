from datetime import datetime

import pytz
import rasterio
from pyrfc3339.generator import generate

def explode(coords):
    """Explode a GeoJSON geometry's coordinates object and yield coordinate tuples.
    As long as the input is conforming, the type of the geometry doesn't matter."""
    for e in coords:
        if isinstance(e, (float, int)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f

def fiona_bbox(f):
    x, y = zip(*list(explode(f['geometry']['coordinates'])))
    return min(x), min(y), max(x), max(y)

def get_rfc3339nano_time():
    return generate(datetime.now().replace(tzinfo=pytz.utc), microseconds=True)


def seek_by_meta_value(input_fn, **meta_term):
    results = {}
    with rasterio.open(input_fn) as input_rio:
        for bidx in input_rio.indexes:
            tags = input_rio.tags(bidx)
            for k, v in meta_term.items():
                if k in tags:
                    for _v in v:
                        if tags[k] == _v:
                            if 'meta' in meta_term:
                                if meta_term['meta']:
                                    if _v in results:
                                        results[_v].append(bidx)
                                    else:
                                        results[_v] = []
                                        results[_v].append(bidx)
                            else:
                                if _v in results:
                                    results[_v].append({bidx: tags})
                                else:
                                    results[_v] = []
                                    results[_v].append({bidx: tags})
    return results


def get_driver_extension(driver):
    driver_to_extension = {
        'GTiff': '.tiff',
        'GRIB': '.grib2',
        'VRT': '.vrt'
    }
    return driver_to_extension[driver]
