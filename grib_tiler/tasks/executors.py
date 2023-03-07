import multiprocessing
import os
import random
import threading
import geopandas as gpd

import fiona
import numpy
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.apps.contour import build_contour
from rasterio.apps.translate import translate
from rasterio.apps.vrt import build_vrt
from rasterio.apps.warp import warp
from rasterio.cutils.min_max import min_max
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.utils import render
from shapely import simplify
from shapely.geometry import shape, mapping

from grib_tiler.tasks import WarpTask, InRangeTask, RenderTileTask, TranslateTask, VirtualTask, IsolinesTask
from grib_tiler.utils import fiona_bbox

simplify_coeff = 0.0
lock = threading.Lock()


def concatenate_raster(virtual_task: VirtualTask):
    return build_vrt(source_filenames=virtual_task.input_filename,
                     dest_filename=virtual_task.output_filename,
                     resample_algorithm='bilinear')


def warp_raster(warp_task: WarpTask):
    return warp(
        source_filename=warp_task.input_filename,
        dest_filename=warp_task.output_filename,
        output_crs=warp_task.output_crs,
        multi_mode=warp_task.multithreading,
        cutline_filename=warp_task.cutline_filename,
        resample_algorithm='bilinear',
        cutline_layer=warp_task.cutline_layer_name,
        output_format=warp_task.output_format,
        source_nodata=warp_task.source_nodata,
        dest_nodata=warp_task.destination_nodata,
        flush_to_disk=warp_task.write_flush,
        target_extent_bbox=warp_task.target_extent,
        target_extent_crs=warp_task.target_extent_crs,
        crop_to_cutline=warp_task.crop_to_cutline)


def _inrange_calculator(rio_ds, band):
    with rasterio.open(rio_ds) as rio_ds:
        try:
            statistics = rio_ds.statistics(band, approx=True, clear_cache=True)
            return statistics.min, statistics.max
        except rasterio._err.CPLE_AppDefinedError:
            statistics = rio_ds.statistics(band, approx=False, clear_cache=True)
            return statistics.min, statistics.max


def in_range_calculator(inrange_task: InRangeTask):
    in_ranges = []
    bands = inrange_task.bands
    if not bands:
        with rasterio.open(inrange_task.input_filename) as input_rio:
            bands = input_rio.indexes
    else:
        if len(bands) == 1:
            bands = [1]
    with rasterio.open(inrange_task.input_filename) as input_rio:
        for band in bands:
            try:
                statistics = input_rio.statistics(band, approx=True, clear_cache=True)
                in_ranges.append((statistics.min, statistics.max))
            except rasterio._err.CPLE_AppDefinedError:
                statistics = input_rio.statistics(band, approx=False, clear_cache=True)
                in_ranges.append((statistics.min, statistics.max))
    return tuple(in_ranges)


def render_tile(render_tile_task: RenderTileTask):
    with Reader(input=render_tile_task.input_filename,
                tms=render_tile_task.tms) as input_file_rio:
        try:
            tile = input_file_rio.tile(tile_z=render_tile_task.z,
                                       tile_y=render_tile_task.y,
                                       tile_x=render_tile_task.x,
                                       tilesize=render_tile_task.tilesize,
                                       resampling_method='bilinear')
            if isinstance(render_tile_task.nodata_mask, np.ndarray):
                tile.mask = render_tile_task.nodata_mask
            if tile.data.shape[0] > 4:
                render_tile_task.image_format = 'GTIFF'
            if render_tile_task.image_format == 'JPEG':
                if tile.data.shape[0] == 2:
                    tile_mask = numpy.reshape(numpy.expand_dims(tile.mask, axis=-1), (1, render_tile_task.tilesize,
                                                                                      render_tile_task.tilesize))
                    tile.data = np.concatenate((tile.data, tile_mask), axis=0)
            tile_bytes = tile.render(img_format=render_tile_task.image_format)
            del tile
        except TileOutsideBounds:
            if render_tile_task.image_format == 'JPEG':
                if len(render_tile_task.bands) == 2:
                    tile_bytes = render(data=numpy.zeros(
                        shape=(
                            3, render_tile_task.tilesize, render_tile_task.tilesize),
                        dtype='uint8'), img_format=render_tile_task.image_format)
                else:
                    tile_bytes = render(data=numpy.zeros(
                        shape=(
                            len(input_file_rio.dataset.indexes), render_tile_task.tilesize, render_tile_task.tilesize),
                        dtype='uint8'), img_format=render_tile_task.image_format)
            else:
                tile_bytes = render(data=numpy.zeros(
                    shape=(
                        len(input_file_rio.dataset.indexes), render_tile_task.tilesize, render_tile_task.tilesize),
                    dtype='uint8'), img_format=render_tile_task.image_format)
    with open(render_tile_task.output_filename, 'wb') as tile_file:
        tile_file.write(tile_bytes)


def isolines_from_band(isolines_task: IsolinesTask):
    return build_contour(source_raster_filename=isolines_task.input_filename,
                         output_vector_filename=isolines_task.output_filename,
                         elevation_interval=isolines_task.elevation_interval)


def translate_raster(translate_task: TranslateTask):
    return translate(source_filename=translate_task.input_filename,
                     dest_filename=translate_task.output_filename,
                     bands_list=translate_task.bands,
                     output_format=translate_task.output_format,
                     min_max_list=translate_task.scale,
                     resample_algorithm='bilinear',
                     output_dtype=translate_task.output_dtype)


def vrt_to_raster(args):
    input_filename = args[0]
    output_directory = args[1]
    filename = f'{os.path.splitext(input_filename)[0]}_{int(random.randint(0, 1000000))}.tiff'
    output_filename = os.path.join(output_directory, filename)
    translate_task = TranslateTask(input_filename=input_filename,
                                   output_filename=output_filename,
                                   output_format='GTiff',
                                   output_dtype='Byte')
    return translate_raster(translate_task)


def calculate_band_minmax(args):
    input_filename = args
    return min_max(input_filename)


def extract_isoline_properties(feature):
    global simplify_coeff
    with lock:
        isoline = {
            "value": None,
            "bbox": None,
            "points": []
        }
        geometry_coordinates = feature["geometry"]["coordinates"]
        if simplify_coeff != 0.0:
            shapely_geom = shape(feature["geometry"])
            shapely_geom = simplify(shapely_geom, tolerance=simplify_coeff)
            geometry = mapping(shapely_geom)
            geometry_coordinates = geometry['coordinates']

        for x, y in geometry_coordinates:
            isoline["points"].append(float(format(y, ".2f")))
            isoline["points"].append(float(format(x, ".2f")))
        isoline["bbox"] = list(fiona_bbox(geometry_coordinates))
        isoline["value"] = int(feature["properties"]["ELEV"])
    return isoline


def band_isolines(args):
    global simplify_coeff
    isolines_json = {
        "isoline": [

        ]
    }
    input_filename = args[0]
    output_directory = args[1]
    elevation_interval = args[2]
    simplify_epsilon = args[3]
    simplify_coeff = simplify_epsilon
    filename = f'{os.path.splitext(input_filename)[0]}_isolines_{int(random.randint(0, 1000000))}.gpkg'
    output_filename = os.path.join(output_directory, filename)
    isolines_task = IsolinesTask(input_filename=input_filename,
                                 output_filename=output_filename,
                                 elevation_interval=elevation_interval)
    isolines_filename = isolines_from_band(isolines_task)
    with fiona.open(isolines_filename) as isolines_vds:
        with multiprocessing.Pool(os.cpu_count()) as isoline_extract_pool:
            for result in isoline_extract_pool.map(extract_isoline_properties, isolines_vds):
                isolines_json["isoline"].append(result)
    return isolines_json


def warp_band(args):
    input_filename = args[0]
    output_crs = args[1]
    output_crs_bounds = args[2]
    bounds_crs = args[3]
    cutline_filename = args[4]
    cutline_layer = args[5]
    output_directory = args[6]
    crop_to_cutline = args[7]
    warp_task = WarpTask(input_filename=input_filename,
                             output_directory=output_directory,
                             output_crs=output_crs,
                             target_extent=output_crs_bounds,
                             target_extent_crs=bounds_crs,
                             cutline_filename=cutline_filename,
                             cutline_layer_name=cutline_layer,
                             output_format='VRT',
                             crop_to_cutline=crop_to_cutline)
    return warp_raster(warp_task)


def extract_band(args):
    input_filename = args[0]
    band = args[1]
    output_directory = args[2]
    filename = f'{os.path.splitext(input_filename)[0].replace(" ", "_")}_{int(random.randint(0, 1000000))}.vrt'
    output_filename = os.path.join(output_directory, filename)
    translate_task = TranslateTask(input_filename=input_filename,
                                   output_filename=output_filename,
                                   band=band,
                                   output_dtype=None)
    return translate_raster(translate_task)


def precut_bands(args):
    input_filename = args[0]
    band = args[1]
    output_directory = args[2]
    output_crs = 'EPSG:4326'
    target_extent = CRS.from_epsg(4326).area_of_use.bounds,
    target_extent_crs = 'EPSG:4326'
    warp_task = WarpTask(input_filename=input_filename,
                         output_directory=output_directory,
                         output_crs=output_crs,
                         target_extent=target_extent,
                         target_extent_crs=target_extent_crs,
                         output_format='VRT')
    return warp_raster(warp_task), band, output_directory


def warp_bands(args):
    input_filename = args[0]
    band = args[1]
    output_directory = args[2]
    output_crs = args[3]
    target_extent = args[4]
    target_extent_crs = args[5]
    cutline_filename = args[6]
    cutline_layer = args[7]
    warp_task = WarpTask(input_filename=input_filename,
                         output_directory=output_directory,
                         output_crs=output_crs,
                         target_extent=target_extent,
                         target_extent_crs='EPSG:4326',
                         cutline_filename=cutline_filename,
                         cutline_layer_name=cutline_layer,
                         output_format='VRT')
    return warp_raster(warp_task), band, output_directory


def calculate_inrange_bands(args):
    input_filename = args[0]
    bands = args[1]
    inrange_task = InRangeTask(input_filename=input_filename,
                               bands=bands)
    return input_filename, bands, in_range_calculator(inrange_task)


def transalte_bands_to_byte(args):
    input_filename = args[0]
    scale = args[1]
    output_directory = args[2]
    filename = f'{os.path.splitext(input_filename)[0]}_byte.vrt'
    output_filename = os.path.join(output_directory, filename)
    translate_task = TranslateTask(input_filename=input_filename,
                                   output_filename=output_filename,
                                   scale=scale,
                                   output_dtype='Byte')
    return translate_raster(translate_task)


def concatenate_bands(args):
    input_filenames = args[0]
    output_directory = args[1]
    filename = f'{str(random.randint(0, 1000000))}_conc.vrt'
    output_filename = os.path.join(output_directory, filename)
    vrt_task = VirtualTask(input_filename=input_filenames,
                           output_filename=output_filename)
    return concatenate_raster(vrt_task)
