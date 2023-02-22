import os
import random

import numpy
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.apps.translate import translate
from rasterio.apps.warp import warp
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.utils import render

from grib_tiler.tasks import WarpTask, InRangeTask, RenderTileTask, TranslateTask


def warp_raster(warp_task: WarpTask):
    warp(
        src_ds=warp_task.input_filename,
        dst_ds=warp_task.output_filename,
        output_crs=warp_task.output_crs,
        multi=warp_task.multithreading,
        cutline_fn=warp_task.cutline_filename,
        resample_algo='bilinear',
        cutline_layer=warp_task.cutline_layer_name,
        output_format=warp_task.output_format,
        src_nodata=warp_task.source_nodata,
        dst_nodata=warp_task.destination_nodata,
        write_flush=warp_task.write_flush,
        target_extent=warp_task.target_extent,
        target_extent_crs=warp_task.target_extent_crs)
    return warp_task.output_filename


def in_range_calculator(inrange_task: InRangeTask):
    in_ranges = []
    bands = inrange_task.bands
    if isinstance(inrange_task.bands, int):
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
    os.makedirs(os.path.join(render_tile_task.output_directory,
                             str(render_tile_task.z), str(render_tile_task.x)),
                exist_ok=True)
    with Reader(input=render_tile_task.input_filename,
                tms=render_tile_task.tms) as input_file_rio:
        try:
            tile = input_file_rio.tile(tile_z=render_tile_task.z,
                                       tile_y=render_tile_task.y,
                                       tile_x=render_tile_task.x,
                                       tilesize=render_tile_task.tilesize,
                                       nodata=render_tile_task.nodata,
                                       indexes=render_tile_task.bands)
            if isinstance(render_tile_task.nodata_mask, np.ndarray):
                tile.mask = render_tile_task.nodata_mask
            if render_tile_task.image_format == 'JPEG':
                if tile.data.shape[0] == 2:
                    tile_mask = numpy.reshape(numpy.expand_dims(tile.mask, axis=-1), (1, render_tile_task.tilesize,
                                                                                      render_tile_task.tilesize))
                    tile.data = np.concatenate((tile.data, tile_mask), axis=0)
            tile_bytes = tile.render(img_format=render_tile_task.image_format)
            del tile
        except TileOutsideBounds:
            tile_bytes = render(data=numpy.zeros(
                shape=(len(input_file_rio.dataset.indexes), render_tile_task.tilesize, render_tile_task.tilesize),
                dtype='uint8'), img_format=render_tile_task.image_format)
    with open(render_tile_task.output_filename, 'wb') as tile_file:
        tile_file.write(tile_bytes)


def translate_raster(translate_task: TranslateTask):
    translate(src_ds=translate_task.input_filename,
              dst_ds=translate_task.output_filename,
              bands=translate_task.bands,
              output_format=translate_task.output_format,
              scale=translate_task.scale,
              output_dtype=translate_task.output_dtype)
    return translate_task.output_filename

def extract_band(args):
    input_filename = args[0]
    band = args[1]
    output_directory = args[2]
    filename = f'{os.path.splitext(input_filename)[0]}_{int(random.randint(0, 1000000))}.vrt'
    output_filename = os.path.join(output_directory, filename)
    translate_task = TranslateTask(input_filename=input_filename,
                                   output_filename=output_filename,
                                   band=band,
                                   output_dtype=None)
    return translate_raster(translate_task), band, output_directory

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