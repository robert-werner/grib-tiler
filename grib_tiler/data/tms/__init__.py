import os.path

import morecantile
from pyproj import CRS

CUSTOM_TMS = {
    'EPSG:3575': morecantile.TileMatrixSet.custom(crs=CRS.from_epsg(3575), extent=[180.0 / 512,
                                                                                   -90.0 / 512,
                                                                                   -180.0 / 512,
                                                                                       90.0 / 512],
                                                  extent_crs=CRS.from_epsg(4326))
}


def load_tms(output_crs):
    if output_crs in CUSTOM_TMS:
        tms = CUSTOM_TMS[output_crs]
    else:
        tms = morecantile.TileMatrixSet.custom(
            extent=list(CRS.from_user_input(output_crs).area_of_use.bounds),
            crs=CRS.from_user_input(output_crs),
            extent_crs=CRS.from_epsg(4326)
        )
    return tms
