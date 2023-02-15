import json
import os

import numpy
import numpy as np
import rasterio
from rasterio.apps.translate import translate
from rasterio.apps.warp import warp
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.utils import render

from grib_tiler.utils.step_namedtuples import RenderTileTask

META_INFO = {
    "meta": {
        "common": [
            {"rstep": 0, "rmin": 0},
            {"gstep": 0, "gmin": 0},
            {"bstep": 0, "bmin": 0},
            {"astep": 0, "amin": 0}
        ]
    }
}


def extract_band(translate_task):
    if translate_task.band:
        band_postfix = f'_b{translate_task.band}'
    else:
        band_postfix = ''
    input_filename = os.path.splitext(os.path.basename(translate_task.input_fn))[0]
    if not translate_task.output_format_extension:
        output_format_extension = os.path.splitext(os.path.basename(translate_task.input_fn))[1]
    else:
        output_format_extension = translate_task.output_format_extension
    output_fn = translate_task.output_fn
    if not translate_task.output_fn:
        output_fn = os.path.join(translate_task.output_dir, f'{input_filename}{band_postfix}{output_format_extension}')
    translate(
        src_ds=translate_task.input_fn,
        dst_ds=output_fn,
        bands=translate_task.band,
        output_format=translate_task.output_format
    )
    return output_fn


def warp_band(warp_task):
    input_filename = os.path.splitext(os.path.basename(warp_task.input_fn))[0]
    input_file_extension = os.path.splitext(os.path.basename(warp_task.input_fn))[1]
    warped_fn = os.path.join(warp_task.output_dir, f'{input_filename}_warped{input_file_extension}')
    warp(
        src_ds=warp_task.input_fn,
        dst_ds=warped_fn,
        output_crs=warp_task.output_crs,
        multi=warp_task.multithreaded,
        cutline_fn=warp_task.cutline_fn,
        cutline_layer=warp_task.cutline_layername,
        output_format=warp_task.output_format,
        src_nodata=warp_task.src_nodata,
        dst_nodata=warp_task.dst_nodata,
        write_flush=warp_task.write_flush,
        target_extent=warp_task.target_extent,
        target_extent_crs=warp_task.target_extent_crs
    )
    return warped_fn


def get_in_ranges(inrange_task):
    with rasterio.open(inrange_task.input_fn) as input_rio:
        try:
            statistics = input_rio.statistics(inrange_task.band, approx=True, clear_cache=True)
            return statistics.min, statistics.max
        except rasterio._err.CPLE_AppDefinedError:
            statistics = input_rio.statistics(inrange_task.band, approx=False, clear_cache=True)
            return statistics.min, statistics.max


def get_render_task(input_fn,
                    z, x, y,
                    tms,
                    nodata,
                    in_range,
                    tilesize,
                    img_format):
    return RenderTileTask(input_fn, z, x, y, tms, nodata, in_range, tilesize, img_format)


def concatenate_tiles(tiles):
    return np.dstack(tiles)


def get_tile_extension(tile_img_format):
    img_to_ext = {
        'PNG': '.png',
        'JPEG': '.jpg',
        'TIFF': '.tiff'
    }
    return img_to_ext[tile_img_format]


def render_tile(render_task):
    os.makedirs(os.path.join(render_task.output_dir, render_task.subdirectory_name), exist_ok=True)
    with Reader(input=render_task.input_fn, tms=render_task.tms) as input_file_rio:
        indexes = list(
            range(1, len(input_file_rio.dataset.indexes) + 1))
        in_range = render_task.in_range_calculator
        if len(indexes) == 1:
            indexes = indexes[0]
            in_range = [render_task.in_range_calculator, ]
        try:
            tile = input_file_rio.tile(tile_z=render_task.z,
                                       tile_y=render_task.y,
                                       tile_x=render_task.x,
                                       tilesize=render_task.tilesize,
                                       nodata=render_task.nodata,
                                       indexes=indexes).post_process(
                in_range=in_range,
                out_dtype='uint8')
            if isinstance(render_task.zero_mask, numpy.ndarray):
                tile.mask = render_task.zero_mask
            tile_bytes = tile.render(img_format=render_task.img_format)
        except TileOutsideBounds:
            tile_bytes = render(data=numpy.zeros(
                shape=(len(input_file_rio.dataset.indexes), render_task.tilesize, render_task.tilesize),
                dtype='uint8'), img_format=render_task.img_format)
    os.makedirs(os.path.join(render_task.output_dir, render_task.band_name, f'{render_task.z}/{render_task.x}'),
                exist_ok=True)
    with open(os.path.join(render_task.output_dir, render_task.band_name,
                           f'{render_task.z}/{render_task.x}/{render_task.y}{get_tile_extension(render_task.img_format)}'),
              'wb') as tile_file:
        tile_file.write(tile_bytes)


def write_metainfo(meta_info_task):
    meta_info = META_INFO
    in_range = meta_info_task.in_range_calculator
    if isinstance(in_range[0], float):
        in_range = [in_range]
    for idx, band, in_range in zip(range(0, len(meta_info_task.in_range_calculator) + 1),
                                   ['r', 'g', 'b', 'a'][0:len(meta_info_task.in_range_calculator)], in_range):
        meta_info['meta']['common'][idx][f'{band}step'] = (in_range[1] - in_range[0]) / 255
        meta_info['meta']['common'][idx][f'{band}min'] = in_range[0]
    with open(meta_info_task.output_dir, 'w') as meta_json:
        json.dump(meta_info, meta_json, indent=4)


def grep_meta(input_fn, meta_key):
    results = []
    with rasterio.open(input_fn) as input_rio:
        for bidx in input_rio.indexes:
            tags = input_rio.tags(bidx)
            if meta_key in tags:
                results.append(tags[meta_key])
    results = list(set(results))
    return results[0]
