import os.path

import morecantile
from pyproj import CRS

CUSTOM_TMS = {
    'EPSG:3575': {
        'crs': CRS.from_epsg(3575),
        'extent': [180.0 / 512,
                   -90.0 / 512,
                   -180.0 / 512,
                   90.0 / 512],
        'extent_crs': CRS.from_epsg(4326)}
}


def load_tms(output_crs, tilesize):
    if output_crs in CUSTOM_TMS:
        tms_params = CUSTOM_TMS[output_crs]
        tms_params['tile_width'] = tilesize
        tms_params['tile_height'] = tilesize
        tms = morecantile.TileMatrixSet.custom(**tms_params)
    else:
        tms_params = {'extent': list(CRS.from_user_input(output_crs).area_of_use.bounds),
                      'crs': CRS.from_user_input(output_crs), 'extent_crs': CRS.from_epsg(4326),
                      'tile_width': tilesize,
                      'tile_height': tilesize}
        tms = morecantile.TileMatrixSet.custom(**tms_params)
    return tms
