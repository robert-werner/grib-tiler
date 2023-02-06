import json
import os
import tempfile
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor

import click
import mercantile
import morecantile
import numpy
import rasterio
import rasterio.mask
import tqdm
from pyproj import CRS
from rasterio._io import unlink_vsi
from rasterio.apps.translate import translate
from rasterio.apps.warp import warp
from rasterio.rio import options
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.utils import render
from tqdm.contrib.concurrent import process_map

from utils import click_options
from utils.click_handlers import bands_handler, zooms_handler

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = '/dev/null'

META_DICT = {
  "meta": {
    "common": [
      { "rstep": 0, "rmin": 0 },
      { "gstep": 0, "gmin": 0 },
      { "bstep": 0, "bmin": 0 },
      { "astep": 0, "amin": 0 }
    ]
  }
}



def render_tile(input_file,
                output_path,
                tile,
                tms,
                nodata,
                in_range,
                tilesize,
                img_format):
    with threading.Lock():
        try:
            if in_range:
                with Reader(input=input_file, tms=tms) as input_file_rio:
                    output_tile_bytes = input_file_rio.tile(tile_z=tile.z,
                                                      tile_y=tile.y,
                                                      tile_x=tile.x,
                                                      tilesize=tilesize,
                                                      nodata=nodata,
                                                      indexes=1).post_process(
                        in_range=in_range,
                        out_dtype='uint8').render(nodata=nodata, img_format=img_format)
            else:
                output_tile_bytes = render(data=numpy.zeros(shape=(tilesize, tilesize),
                                                            dtype='uint8'),
                                           nodata=nodata,
                                           img_format=img_format)
        except TileOutsideBounds:
            output_tile_bytes = render(data=numpy.zeros(shape=(tilesize, tilesize),
                                                        dtype='uint8'),
                                       nodata=nodata,
                                       img_format=img_format)
        with open(output_path, 'wb') as tile_file:
            tile_file.write(output_tile_bytes)


def render_wrapper(render_kwargs):
    if not render_kwargs:
        return None
    return render_tile(**render_kwargs)


def get_tile_extension(tile_img_format):
    img_to_ext = {
        'PNG': 'png',
        'JPEG': 'jpg',
        'GTiff': 'tiff'
    }
    return img_to_ext[tile_img_format]


def process_raster(src_ds=None,
                   dst_ds=None,
                   output_crs=None,
                   multi=None,
                   cutline_fn=None,
                   cutline_layer=None,
                   output_format=None,
                   src_nodata=None,
                   dst_nodata=None,
                   bands=None,
                   temp_dir=None,
                   write_flush=None,
                   tms=None,
                   tilesize=None,
                   img_format=None,
                   tile_folder=None,
                   tile_extension=None,
                   tiles_list=None):
    extracted_ds_fn = translate(src_ds=src_ds,
                                dst_ds=dst_ds,
                                bands=bands,
                                output_format=output_format)
    input_filename = os.path.splitext(os.path.basename(extracted_ds_fn))[0]
    warped_dst_fn = os.path.join(temp_dir.name, f'{input_filename}_warped.tiff')
    warped_ds_fn = warp(
        src_ds=extracted_ds_fn,
        dst_ds=warped_dst_fn,
        output_crs=output_crs,
        multi=multi,
        cutline_fn=cutline_fn,
        cutline_layer=cutline_layer,
        output_format='GTiff',
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        write_flush=write_flush
    )
    meta_path = os.path.join(tile_folder, str(bands[0]))
    os.makedirs(meta_path, exist_ok=True)
    with rasterio.open(warped_ds_fn) as wds_rio:
        try:
            stats = wds_rio.statistics(bidx=1, approx=True, clear_cache=True)
            in_range = ((stats.min, stats.max),)
            meta_dict = META_DICT
            meta_dict['meta']['common'][0]['rstep'] = (stats.max - stats.min) / 255
            meta_dict['meta']['common'][0]['rmin'] = stats.min
            with open(os.path.join(meta_path, 'meta.json'), 'w') as meta_file:
                json.dump(meta_dict, meta_file)
        except rasterio._err.CPLE_AppDefinedError:
            in_range = None

    if 'vsimem' in extracted_ds_fn:
        unlink_vsi(extracted_ds_fn)
    else:
        os.remove(extracted_ds_fn)

    tile_params = []
    for tile in tiles_list:
        output_path_directory = os.path.join(os.path.join(tile_folder, str(bands[0])), f'{tile.z}/{tile.x}')
        output_path = os.path.join(output_path_directory,
                                   f'{tile.y}.{tile_extension}')
        os.makedirs(output_path_directory, exist_ok=True)

        if os.path.exists(output_path):
            tile_params.append(False)
        else:
            tile_params.append(
                {
                    'input_file': warped_ds_fn,
                    'output_path': output_path,
                    'tile': tile,
                    'tms': tms,
                    'nodata': dst_nodata,
                    'in_range': in_range,
                    'tilesize': tilesize,
                    'img_format': img_format,
                }
            )
    return tile_params


def process_raster_wrapper(kwargs):
    return process_raster(**kwargs)


@click.command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@options.file_in_arg
@click_options.file_out_arg
@click_options.bidx_magic_opt
@click_options.img_format_opt
@click_options.cutline_opt
@click_options.cutline_layer_opt
@click_options.tile_dimension_opt
@click_options.out_crs_opt
@click_options.threads_opt
@click_options.zooms_opt
@click_options.coeff_opt
@click_options.nodata_opt
def grib_tiler(input,
               output,
               band_idx,
               img_format,
               cutline,
               cutline_layer,
               dimension,
               output_crs,
               threads,
               zooms,
               coeff,
               nodata):
    tms = morecantile.TileMatrixSet.custom(extent=[coord / coeff for coord in [180.0, -90.0, -180.0, 90.0]],
                                           extent_crs=CRS.from_epsg(4326),
                                           crs=CRS.from_user_input(output_crs))
    tilesize = dimension
    tile_extension = get_tile_extension(img_format)

    os.makedirs(output, exist_ok=True)
    band_idx = bands_handler(band_idx)
    zooms = zooms_handler(zooms)
    with rasterio.open(input) as input_rio:
        if not band_idx:
            band_idx = input_rio.indexes
        input_driver = input_rio.driver
        render_bounds = input_rio.bounds

    tiles_list = list(mercantile.tiles(*render_bounds, zooms))

    input_path_splitext = os.path.splitext(input)
    input_path_filename, input_extension = input_path_splitext[0], input_path_splitext[1]
    input_filename = os.path.splitext(os.path.basename(input))[0]

    pp_executor = ProcessPoolExecutor(max_workers=threads)

    temp_dir = tempfile.TemporaryDirectory()
    band_pipeline_tasks = [
        {
            'src_ds': input,
            'dst_ds': os.path.join(temp_dir.name, f'{input_filename}_b{bidx}{input_extension}'),
            'output_crs': output_crs,
            'multi': True,
            'cutline_fn': cutline,
            'cutline_layer': cutline_layer,
            'output_format': input_driver,
            'src_nodata': nodata,
            'dst_nodata': nodata,
            'bands': [bidx],
            'temp_dir': temp_dir,
            'write_flush': True,
            'tms': tms,
            'tilesize': tilesize,
            'img_format': img_format,
            'tile_folder': output,
            'tile_extension': tile_extension,
            'tiles_list': tiles_list
        }
        for bidx in band_idx
    ]
    results = process_map(process_raster_wrapper, band_pipeline_tasks, max_workers=6,
                          desc='Обрезка и перепроецирование тайлов')
    tiling_pbar = tqdm.tqdm(total=(len(band_idx) * len(tiles_list)), desc='Тайлирование')
    for result in results:
        for _ in pp_executor.map(render_wrapper, result):
            tiling_pbar.update(1)
        if result[0]:
            os.remove(result[0]['input_file'])
    temp_dir.cleanup()


if __name__ == '__main__':
    grib_tiler()
