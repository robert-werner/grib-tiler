import os
import sys
import tempfile
import warnings

import click
import fiona
import mercantile
import numpy as np
import rasterio
from pyproj import CRS
from tqdm.contrib.concurrent import process_map

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks import WarpTask, InRangeTask, TranslateTask, RenderTileTask
from grib_tiler.tasks.executors import warp_raster, in_range_calculator, translate_raster, render_tile
from grib_tiler.utils import click_options, seek_by_meta_value
from grib_tiler.utils.click_handlers import zooms_handler, bands_handler

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = '/dev/null'

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

temp_dir = tempfile.TemporaryDirectory()


def warp_input(input_subsets, cutline_filename, cutline_layer, output_crs, threads):
    warp_subsets = []
    warp_subsets_tasks = []
    if cutline_filename:
        if output_crs:
            for input_subset in input_subsets:
                warp_subsets_task = WarpTask(input_filename=input_subset,
                                             output_directory=temp_dir.name,
                                             output_crs=output_crs,
                                             cutline_filename=cutline_filename,
                                             cutline_layer_name=cutline_layer,
                                             output_format='VRT')
                warp_subsets_tasks.append(warp_subsets_task)
        else:
            for input_subset in input_subsets:
                warp_subsets_task = WarpTask(input_filename=input_subset,
                                             output_directory=temp_dir.name,
                                             cutline_filename=cutline_filename,
                                             cutline_layer_name=cutline_layer,
                                             output_format='VRT')
                warp_subsets_tasks.append(warp_subsets_task)
    else:
        if output_crs:
            if output_crs in ['EPSG:3857']:
                sub_warp_subsets_tasks = []
                for input_subset in input_subsets:
                    sub_warp_subsets_task = WarpTask(input_filename=input_subset,
                                                     output_directory=temp_dir.name,
                                                     output_crs='EPSG:4326',
                                                     target_extent=EPSG_4326_BOUNDS,
                                                     target_extent_crs='EPSG:4326',
                                                     output_format='VRT')
                    sub_warp_subsets_tasks.append(sub_warp_subsets_task)
                input_subsets = process_map(warp_raster,
                                            sub_warp_subsets_tasks,
                                            max_workers=threads,
                                            desc='Предварительное перепроецирование каналов')
                for input_subset in input_subsets:
                    warp_subsets_task = WarpTask(input_filename=input_subset,
                                                 output_directory=temp_dir.name,
                                                 output_crs=output_crs,
                                                 target_extent=list(CRS.from_epsg(3857).area_of_use.bounds),
                                                 target_extent_crs='EPSG:4326',
                                                 output_format='VRT')
                    warp_subsets_tasks.append(warp_subsets_task)
            else:
                for input_subset in input_subsets:
                    warp_subsets_task = WarpTask(input_filename=input_subset,
                                                 output_directory=temp_dir.name,
                                                 output_crs=output_crs,
                                                 target_extent=list(
                                                     CRS.from_user_input(output_crs).area_of_use.bounds),
                                                 target_extent_crs='EPSG:4326',
                                                 output_format='VRT')
                    warp_subsets_tasks.append(warp_subsets_task)
    warp_subsets = process_map(warp_raster,
                               warp_subsets_tasks,
                               max_workers=threads,
                               desc='Финальное перепроецирование каналов')
    return warp_subsets


def prepare_for_tiling(warped_subsets, threads):
    rasters_for_tiling_tasks = []
    if warped_subsets:
        for warped_subset in warped_subsets:
            output_filename = warped_subset.replace('vrt', 'tiff').replace('_warped', '')
            rasters_for_tiling_task = TranslateTask(input_filename=warped_subset,
                                                    output_filename=output_filename,
                                                    output_format='GTiff')
            rasters_for_tiling_tasks.append(rasters_for_tiling_task)
    else:
        for warped_subset in warped_subsets:
            output_filename = warped_subset.replace('vrt', 'tiff').replace('_warped', '')
            rasters_for_tiling_task = TranslateTask(input_filename=warped_subset,
                                                    output_filename=output_filename,
                                                    output_format='GTiff')
            rasters_for_tiling_tasks.append(rasters_for_tiling_task)
    rasters_for_tiling = process_map(translate_raster,
                                     rasters_for_tiling_tasks,
                                     max_workers=threads,
                                     desc='Рендеринг преобразованных каналов для тайлирования')
    return rasters_for_tiling


def prepare_in_ranges(rasters_for_tiling, threads):
    in_range_tasks = []
    for raster_for_tiling in rasters_for_tiling:
        in_range_task = InRangeTask(input_filename=raster_for_tiling)
        in_range_tasks.append(in_range_task)
    in_ranges = process_map(in_range_calculator,
                            in_range_tasks,
                            max_workers=threads,
                            desc='Вычисление мин/макс значений каналов')
    return in_ranges


def prepare_tiling_tasks(rasters_for_tiling, in_ranges,
                         tiles, output,
                         tms, tilesize,
                         image_format, generate_nodata_mask=True):
    tiling_tasks = []
    nodata_mask = None
    if generate_nodata_mask:
        nodata_mask = np.zeros((tilesize, tilesize), dtype='uint8')
    for raster_for_tiling, in_range in zip(rasters_for_tiling, in_ranges):
        for tile in tiles:
            tiling_task = RenderTileTask(input_filename=raster_for_tiling,
                                         output_directory=output,
                                         z=tile.z,
                                         x=tile.x,
                                         y=tile.y,
                                         tms=tms,
                                         tilesize=tilesize,
                                         in_range=in_range,
                                         image_format=image_format,
                                         subdirectory_name=os.path.splitext(os.path.basename(raster_for_tiling))[0],
                                         nodata_mask_array=nodata_mask)
            tiling_tasks.append(tiling_task)
    return tiling_tasks


@click.command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.files_in_arg
@click_options.file_out_arg
@click_options.wind_opt
@click_options.bands_opt
@click_options.img_format_opt
@click_options.cutline_opt
@click_options.cutline_layer_opt
@click_options.tile_dimension_opt
@click_options.out_crs_opt
@click_options.threads_opt
@click_options.zooms_opt
def grib_tiler(input_filename,
               output,
               band_numbers,
               image_format,
               cutline_filename,
               cutline_layer,
               tilesize,
               output_crs,
               threads,
               zooms,
               wind):
    tms = None
    if output_crs:
        tms = load_tms(output_crs)
    zooms = zooms_handler(zooms)

    if wind:
        uv_seek_results = seek_by_meta_value(input_filename, GRIB_ELEMENT=['UGRD', 'VGRD'])
        u_bands_list = uv_seek_results['UGRD']
        v_bands_list = uv_seek_results['VGRD']
        if len(u_bands_list) != len(v_bands_list):
            click.echo('Входной GRIB-файл непригоден для генерации UV-тайлов')
            raise click.Abort()
        uv_bands_list = list(zip(u_bands_list, v_bands_list))
        uv_subsets_tasks = []
        for uv_bands in uv_bands_list:
            uv_bands_indexes = [list(uv_bands[0].keys())[0], list(uv_bands[1].keys())[0]]
            uv_band_comment = list(uv_bands[0].values())[0]['GRIB_SHORT_NAME']
            output_filename = os.path.join(temp_dir.name, f'{uv_band_comment}.vrt')
            uv_subsets_task = TranslateTask(input_filename=input_filename,
                                            output_filename=output_filename,
                                            output_format='VRT',
                                            bands=uv_bands_indexes)
            uv_subsets_tasks.append(uv_subsets_task)
        uv_subsets = process_map(translate_raster,
                                 uv_subsets_tasks,
                                 max_workers=threads,
                                 desc='Извлечение каналов ветра')
        if cutline_filename:
            with fiona.open(cutline_filename) as cutline_fio_ds:
                tiles = list(mercantile.tiles(*cutline_fio_ds.bounds, zooms))
        else:
            with rasterio.open(input_filename) as cut_warped_rio_ds:
                extent = cut_warped_rio_ds.bounds
                tiles = list(mercantile.tiles(*extent, zooms))
        warp_uv_subsets = warp_input(uv_subsets, cutline_filename, cutline_layer, output_crs, threads)
        if warp_uv_subsets:
            rasters_for_tiling = prepare_for_tiling(warp_uv_subsets, threads)
        else:
            rasters_for_tiling = prepare_for_tiling(uv_subsets, threads)
        in_ranges = prepare_in_ranges(rasters_for_tiling, threads)
        tiling_tasks = prepare_tiling_tasks(rasters_for_tiling, in_ranges, tiles, output, tms, tilesize, image_format)
    else:
        band_numbers = bands_handler(band_numbers)
        if not band_numbers:
            with rasterio.open(input_filename) as input_rio_ds:
                band_numbers = input_rio_ds.indexes
        band_subsets_tasks = []
        for band_number in band_numbers:
            output_filename = os.path.join(temp_dir.name, f'{band_number}.vrt')
            band_subsets_task = TranslateTask(input_filename=input_filename,
                                              output_filename=output_filename,
                                              output_format='VRT',
                                              bands=[band_number])
            band_subsets_tasks.append(band_subsets_task)
        band_subsets = process_map(translate_raster,
                                   band_subsets_tasks,
                                   max_workers=threads,
                                   desc='Извлечение выбранных каналов GRIB-файла')
        tiles = None
        warp_band_subsets = None
        if cutline_filename:
            with fiona.open(cutline_filename) as cutline_fio_ds:
                tiles = list(mercantile.tiles(*cutline_fio_ds.bounds, zooms))
        else:
            with rasterio.open(input_filename) as cut_warped_rio_ds:
                extent = cut_warped_rio_ds.bounds
                tiles = list(mercantile.tiles(*extent, zooms))
        warp_band_subsets = warp_input(band_subsets, cutline_filename, cutline_layer, output_crs, threads)
        if warp_band_subsets:
            rasters_for_tiling = prepare_for_tiling(warp_band_subsets, threads)
        else:
            rasters_for_tiling = prepare_for_tiling(band_subsets, threads)
        in_ranges = prepare_in_ranges(rasters_for_tiling, threads)
        tiling_tasks = prepare_tiling_tasks(rasters_for_tiling, in_ranges, tiles, output, tms, tilesize, image_format,
                                            generate_nodata_mask=False)
    process_map(render_tile, tiling_tasks,
                max_workers=threads,
                desc='Рендеринг тайлов')
    temp_dir.cleanup()


if __name__ == '__main__':
    try:
        grib_tiler()
    except KeyboardInterrupt:
        temp_dir.cleanup()
        sys.exit()
