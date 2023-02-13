import os
import os
import sys
import tempfile
import warnings

import click
import fiona
import mercantile
import morecantile
import numpy
import rasterio
import rasterio.mask
from pyproj import CRS
from rasterio.apps.vrt import build_vrt
from tqdm.contrib.concurrent import process_map

from grib_tiler.data.tms import load_tms
from grib_tiler.steps import extract_band, warp_band, get_in_ranges, render_tile, \
    write_metainfo
from grib_tiler.utils import click_options
from grib_tiler.utils.click_handlers import bands_handler, zooms_handler
from grib_tiler.utils.step_namedtuples import TranslateTask, WarpTask, InRangeTask, RenderTileTask, MetaInfoTask

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = '/dev/null'
os.environ['GRIB_ADJUST_LONGITUDE_RANGE'] = 'NO'

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

temp_dir = tempfile.TemporaryDirectory()
temp_dir_name = '/root/development/temp'


def buildvrt_wrapper(vrtask):
    return build_vrt(vrtask.src_ds_s, vrtask.band)


@click.command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.files_in_arg
@click_options.file_out_arg
@click_options.uv_opt
@click_options.bands_opt
@click_options.img_format_opt
@click_options.cutline_opt
@click_options.cutline_layer_opt
@click_options.tile_dimension_opt
@click_options.out_crs_opt
@click_options.threads_opt
@click_options.zooms_opt
@click_options.nodata_opt
@click_options.zero_mask_flag
def grib_tiler(inputs,
               output,
               bands,
               img_format,
               cutline,
               cutline_layer,
               tilesize,
               output_crs,
               threads,
               zooms,
               nodata,
               uv,
               zero_mask
               ):
    tms = load_tms(output_crs)
    os.makedirs(output, exist_ok=True)

    zooms = zooms_handler(zooms)

    if uv:
        uv_in_ranges, uv_translated = uv_handler(cutline, cutline_layer, inputs, output_crs, temp_dir_name,
                                                 threads)
        metainfo_writer(output, uv_in_ranges, uv_translated)
        render_tile_task = []
        if zero_mask:
            zero_mask = numpy.zeros((tilesize, tilesize), dtype="uint8")
        with rasterio.open(inputs[0]) as input:
            render_bounds = input.bounds
            tiles_list = list(mercantile.tiles(*render_bounds, zooms))
        for tile in tiles_list:
            for filename, in_range in zip(uv_translated, uv_in_ranges):
                render_tile_task.append(
                    RenderTileTask(input_fn=filename, z=tile.z, x=tile.x, y=tile.y, tms=tms, nodata=nodata,
                                   in_range=in_range, tilesize=tilesize,
                                   img_format=img_format, band_name=os.path.splitext(os.path.basename(filename))[0],
                                   output_dir=output, zero_mask=zero_mask)
                )
        process_map(render_tile, render_tile_task, max_workers=threads, desc='Рендеринг тайлов')
    else:
        inputs = inputs[0]
        in_ranges, translated = band_handler(cutline, cutline_layer, inputs, output_crs, temp_dir_name, threads, bands)
        metainfo_writer(output, in_ranges, translated)
        render_tile_task = []
        if zero_mask:
            zero_mask = numpy.zeros((tilesize, tilesize), dtype="uint8")
        with rasterio.open(translated[0]) as input:
            render_bounds = input.bounds
            tiles_list = list(mercantile.tiles(*render_bounds, zooms))
        for tile in tiles_list:
            for filename, in_range in zip(translated, in_ranges):
                render_tile_task.append(
                    RenderTileTask(input_fn=filename, z=tile.z, x=tile.x, y=tile.y, tms=tms, nodata=nodata,
                                   in_range=in_range, tilesize=tilesize,
                                   img_format=img_format, band_name=os.path.splitext(os.path.basename(filename))[0],
                                   output_dir=output,
                                   zero_mask=zero_mask)
                )
        process_map(render_tile, render_tile_task, max_workers=threads, desc='Рендеринг тайлов')
    temp_dir.cleanup()


def metainfo_writer(output_dir, in_ranges, translated_input):
    metainfo_tasks = []
    for filename, in_range in zip(translated_input, in_ranges):
        os.makedirs(os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0]), exist_ok=True)
        metainfo_tasks.append(
            MetaInfoTask(
                in_range,
                os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0], 'meta.json')
            )
        )
    process_map(write_metainfo, metainfo_tasks, max_workers=6, desc='Генерация метаинформации к каналам')


def band_handler(cutline, cutline_layer, input, output_crs, temp_dir, threads, bands):
    if bands:
        bands = bands_handler(bands)
    else:
        with rasterio.open(input) as input:
            bands = input.indexes
    extracted_bands = process_map(extract_band,
                                  [TranslateTask(input_fn=input, output_dir=temp_dir,
                                                 band=band, output_format='GRIB', output_format_extension='.grib2')
                                   for band in bands],
                                  max_workers=threads,
                                  desc='Извлечение выбранных каналов')
    in_ranges = process_map(get_in_ranges,
                            [InRangeTask(input_fn=input, band=1) for input in extracted_bands],
                            max_workers=threads,
                            desc='Вычисление мин/макс выбранных каналов')
    warped_bands = process_map(warp_band,
                               [WarpTask(input_fn=u_fn, output_dir=temp_dir,
                                         output_crs=output_crs, multithreaded=True,
                                         cutline_fn=cutline,
                                         cutline_layername=cutline_layer, output_format='GTiff',
                                         src_nodata=0, dst_nodata=0,
                                         write_flush=True) for u_fn in extracted_bands],
                               max_workers=threads,
                               desc='Перепроецирование выбранных каналов')
    return in_ranges, warped_bands


def uv_handler(cutline, cutline_layer, inputs, output_crs, temp_dir, threads):
    if len(inputs) != 1:
        click.echo('Допускается на вход только один файл, содержащий U- и V- компоненты')
        raise click.Abort()
    input = inputs[0]
    uv_bands = {}
    with rasterio.open(input) as grib_input:
        for band_idx in grib_input.indexes:
            band_tags = grib_input.tags(band_idx)
            if 'UGRD' == band_tags['GRIB_ELEMENT']:
                uv_bands[band_tags['GRIB_SHORT_NAME']] = []
                uv_bands[band_tags['GRIB_SHORT_NAME']].append(band_idx)
            if 'VGRD' == band_tags['GRIB_ELEMENT']:
                uv_bands[band_tags['GRIB_SHORT_NAME']].append(band_idx)
    uv_shortnames = list(uv_bands.keys())
    uv_filenames = process_map(extract_band,
                               [TranslateTask(input_fn=input, output_dir=temp_dir, band=uv,
                                              output_format='GTiff',
                                              output_format_extension='.tiff',
                                              output_fn=os.path.join(temp_dir, f'{uv_shortname}.tiff')) for
                                uv_shortname, uv in uv_bands.items()],
                               max_workers=threads,
                               desc='Извлечение UV-компонент ветра')
    prewarped_uv_filenames = process_map(warp_band,
                                         [WarpTask(input_fn=uv_fn, output_dir=temp_dir,
                                                   output_crs='EPSG:4326', multithreaded=True,
                                                   cutline_fn=None,
                                                   cutline_layername=None, output_format='GTiff',
                                                   src_nodata=None, dst_nodata=None,
                                                   write_flush=True,
                                                   output_fn=None,
                                                   target_extent=EPSG_4326.to_string(),
                                                   target_extent_crs=EPSG_4326_BOUNDS) for uv_idx, uv_fn in
                                          enumerate(uv_filenames)],
                                         max_workers=threads,
                                         desc='Предварительное перепроецирование UV-компонент ветра')
    cut_uv_filenames = process_map(warp_band,
                                   [WarpTask(input_fn=uv_fn, output_dir=temp_dir,
                                             output_crs='EPSG:4326', multithreaded=True,
                                             cutline_fn=cutline,
                                             cutline_layername=cutline_layer, output_format='GTiff',
                                             src_nodata=None, dst_nodata=None,
                                             write_flush=True,
                                             output_fn=None,
                                             target_extent=None,
                                             target_extent_crs=None) for uv_idx, uv_fn in
                                    enumerate(prewarped_uv_filenames)],
                                   max_workers=threads,
                                   desc='Обрезка UV-компонент ветра')
    u_inranges = process_map(get_in_ranges,
                             [InRangeTask(input_fn=input, band=1) for input in cut_uv_filenames],
                             max_workers=threads,
                             desc='Вычисление мин/макс U-компонент')
    v_inranges = process_map(get_in_ranges,
                             [InRangeTask(input_fn=input, band=2) for input in cut_uv_filenames],
                             max_workers=threads,
                             desc='Вычисление мин/макс V-компонент')
    uv_inranges = list(zip(u_inranges, v_inranges))
    if output_crs:
        uv_filenames = process_map(warp_band,
                                   [WarpTask(input_fn=uv_fn, output_dir=temp_dir,
                                             output_crs=output_crs, multithreaded=True,
                                             cutline_fn=None,
                                             cutline_layername=None,
                                             output_format='GTiff',
                                             src_nodata=None, dst_nodata=None,
                                             write_flush=True,
                                             target_extent=None,
                                             target_extent_crs=None,
                                             output_fn=None)
                                    for uv_idx, uv_fn in enumerate(cut_uv_filenames)],
                                   max_workers=threads,
                                   desc='Перепроецирование UV-компонент ветра')
    return uv_inranges, uv_filenames


if __name__ == '__main__':
    try:
        grib_tiler()
    except KeyboardInterrupt:
        temp_dir.cleanup()
        sys.exit()
