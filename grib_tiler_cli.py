import json
import os
import sys
import tempfile
import warnings

import click
import mercantile
import numpy as np
import rasterio
from click import echo, UsageError
from cloup import command
from pyproj import CRS
from tqdm.contrib.concurrent import process_map

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks import WarpTask, InRangeTask, TranslateTask, RenderTileTask
from grib_tiler.tasks.executors import warp_raster, in_range_calculator, translate_raster
from grib_tiler.utils import click_options
from grib_tiler.utils.click_handlers import zooms_handler
from grib_tiler.utils.click_options import  multiband_constraint
from utils.click_handlers import bands_handler

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = '/dev/null'

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

temp_dir = tempfile.TemporaryDirectory()

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


def warp_input(input_subsets, cutline_filename, cutline_layer, output_crs, threads):
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
    warp_subsets = []
    warp_subsets = process_map(warp_raster,
                               warp_subsets_tasks,
                               max_workers=threads,
                               desc='Финальное перепроецирование каналов')
    return warp_subsets


def prepare_for_tiling(warped_subsets, in_ranges, threads):
    rasters_for_tiling_tasks = []
    if warped_subsets:
        for warped_subset, in_range in zip(warped_subsets, in_ranges):
            output_filename = warped_subset.replace('vrt', 'tiff').replace('_warped', '')
            rasters_for_tiling_task = TranslateTask(input_filename=warped_subset,
                                                    output_filename=output_filename,
                                                    output_format='GTiff',
                                                    scale=in_range)
            rasters_for_tiling_tasks.append(rasters_for_tiling_task)
    else:
        for warped_subset, in_range in zip(warped_subsets, in_ranges):
            output_filename = warped_subset.replace('vrt', 'tiff').replace('_warped', '')
            rasters_for_tiling_task = TranslateTask(input_filename=warped_subset,
                                                    output_filename=output_filename,
                                                    output_format='GTiff',
                                                    scale=in_range)
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


def prepare_tiling_tasks(rasters_for_tiling,
                         tiles, output,
                         tms, tilesize,
                         image_format, in_ranges, generate_nodata_mask=True):
    tiling_tasks = []
    nodata_mask = None
    if generate_nodata_mask:
        nodata_mask = np.zeros((tilesize, tilesize), dtype='uint8')
    for raster_for_tiling, in_ranges in zip(rasters_for_tiling, in_ranges):
        subdirectory_name = os.path.splitext(os.path.basename(raster_for_tiling))[0]
        os.makedirs(os.path.join(output, subdirectory_name), exist_ok=True)
        meta_json_filename = os.path.join(output, subdirectory_name, 'meta.json')
        meta_info = META_INFO
        _in_ranges = in_ranges
        if isinstance(_in_ranges[0], float):
            _in_ranges = [_in_ranges]
        for idx, band, in_range in zip(range(0, len(_in_ranges) + 1),
                                       ['r', 'g', 'b', 'a'][0:len(_in_ranges)], _in_ranges):
            meta_info['meta']['common'][idx][f'{band}step'] = (in_range[1] - in_range[0]) / 255
            meta_info['meta']['common'][idx][f'{band}min'] = in_range[0]
        with open(meta_json_filename, 'w') as meta_json:
            json.dump(meta_info, meta_json, indent=4)
        for tile in tiles:
            tiling_task = RenderTileTask(input_filename=raster_for_tiling,
                                         output_directory=output,
                                         z=tile.z,
                                         x=tile.x,
                                         y=tile.y,
                                         tms=tms,
                                         tilesize=tilesize,
                                         image_format=image_format,
                                         subdirectory_name=os.path.splitext(os.path.basename(raster_for_tiling))[0],
                                         nodata_mask_array=nodata_mask)
            tiling_tasks.append(tiling_task)
    return tiling_tasks

@command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.files_in_arg
@click_options.file_out_arg
@click_options.bands_opt
@click_options.img_format_opt
@click_options.cutline_opt
@click_options.cutline_layer_opt
@click_options.tile_dimension_opt
@click_options.out_crs_opt
@click_options.threads_opt
@click_options.zooms_opt
@click_options.multiband_opt
def grib_tiler(input,
               output,
               band_numbers,
               image_format,
               cutline_filename,
               cutline_layer,
               tilesize,
               output_crs,
               threads,
               zooms,
               multiband):
    input_files_quantity = len(input)
    band_numbers = bands_handler(band_numbers)
    band_numbers_quantity = len(band_numbers)
    click.echo(band_numbers_quantity)
    inputs = []
    if input_files_quantity > 1:
        if multiband:
            if not band_numbers:
                for input_file in input:
                    with rasterio.open(input_file) as input_rio_ds:
                        band_numbers.extend(input_rio_ds.indexes)
                        band_numbers_quantity += len(band_numbers)
                        inputs.extend([input_file] * len(band_numbers))
            elif band_numbers:
                if not all(elem == band_numbers[0] for elem in band_numbers):
                    inputs.extend(input)
                    if band_numbers_quantity != input_files_quantity:
                        raise UsageError('Количество каналов должно соответствовать количеству входных файлов')
                else:
                    inputs.extend(input)
                    band_numbers.extend([band_numbers[0]] * input_files_quantity)
        else:
            raise UsageError('Создавать одноканальные тайлы из множества (больше одного) входных файлов запрещено')
    else:
        if multiband:
            if band_numbers_quantity == 1:
                raise UsageError('Мультиканальные тайлы с одним каналом создавать запрещено')
            elif not band_numbers:
                with rasterio.open(input[0]) as input_rio_ds:
                    band_numbers.extend(input_rio_ds.indexes) # Оборачиваем в ещё один массив для удобства
                    band_numbers_quantity += len(band_numbers)
                    inputs.extend([input[0]] * band_numbers_quantity)
            elif band_numbers:
                inputs.extend([input[0]] * band_numbers_quantity)
        else:
            if not band_numbers:
                with rasterio.open(input[0]) as input_rio_ds:
                    band_numbers.extend(input_rio_ds.indexes) # Оборачиваем в ещё один массив для удобства
                    band_numbers_quantity += len(band_numbers)
                    inputs.extend([input[0]] * band_numbers_quantity)
            else:
                inputs.extend([input[0]] * band_numbers_quantity)
    extract_parameters = list(zip(inputs, band_numbers))



    tms = None
    if output_crs:
        tms, output_crs = load_tms(output_crs)
    if not zooms:
        zooms = list(range(0, 5))
    else:
        zooms = zooms_handler(zooms)
    with rasterio.open(input[0]) as input_rio_ds:
        extent = input_rio_ds.bounds
        echo(f'Генерация списка тайлов с {zooms[0]} по {zooms[-1]} уровни увеличения, пожалуйста, подождите...')
        tiles = list(mercantile.tiles(*extent, zooms))



    temp_dir.cleanup()


if __name__ == '__main__':
    try:
        grib_tiler()
    except:
        temp_dir.cleanup()
        sys.exit()
