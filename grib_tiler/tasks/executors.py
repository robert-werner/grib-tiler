import os
import numpy
import rasterio
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
    with rasterio.open(inrange_task.input_filename) as input_rio:
        if len(input_rio.indexes) == 1:
            try:
                statistics = input_rio.statistics(1, approx=True, clear_cache=True)
                return (statistics.min, statistics.max)
            except rasterio._err.CPLE_AppDefinedError:
                statistics = input_rio.statistics(1, approx=False, clear_cache=True)
                return (statistics.min, statistics.max)
        for band_index in input_rio.indexes:
            try:
                statistics = input_rio.statistics(band_index, approx=True, clear_cache=True)
                in_ranges.append((statistics.min, statistics.max))
            except rasterio._err.CPLE_AppDefinedError:
                statistics = input_rio.statistics(band_index, approx=False, clear_cache=True)
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
                                       indexes=render_tile_task.bands).post_process(
                in_range=render_tile_task.in_range,
                out_dtype='uint8')
            if render_tile_task.nodata_mask:
                tile.mask = render_tile_task.nodata_mask
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
              output_format=translate_task.output_format)
    return translate_task.output_filename

