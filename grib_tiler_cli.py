import os
import sys
import tempfile
import time
import warnings

import mercantile
import rasterio
from click import echo, UsageError, command
from pyproj import CRS
from tqdm.contrib.concurrent import process_map
from utils.click_handlers import bands_handler

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks.executors import extract_band, precut_bands, warp_bands, calculate_inrange_bands
from grib_tiler.utils import click_options
from grib_tiler.utils.click_handlers import zooms_handler

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
    tms, output_crs, output_crs_name = load_tms(output_crs)
    zooms = zooms_handler(zooms)
    echo(f'Генерация списка тайлов с {zooms[0]} по {zooms[-1]} уровни увеличения, пожалуйста, подождите...')
    tiles = list(mercantile.tiles(*EPSG_4326_BOUNDS, zooms))


    input_files_quantity = len(input)
    band_numbers = bands_handler(band_numbers)
    band_numbers_quantity = len(band_numbers)
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
                    band_numbers.extend([input_rio_ds.indexes]) # Оборачиваем в ещё один массив для удобства
                    band_numbers_quantity += len(band_numbers)
                    inputs.extend([input[0]] * band_numbers_quantity)
            elif band_numbers:
                band_numbers = [band_numbers] # Оборачиваем в ещё один массив для удобства
                inputs.extend([input[0]] * band_numbers_quantity)
        else:
            if not band_numbers:
                with rasterio.open(input[0]) as input_rio_ds:
                    band_numbers.extend(input_rio_ds.indexes) # Оборачиваем в ещё один массив для удобства
                    band_numbers_quantity += len(band_numbers)
                    inputs.extend([input[0]] * band_numbers_quantity)
            else:
                inputs.extend([input[0]] * band_numbers_quantity)
    if multiband:
        generation_time = str(int(time.time()))
        result_directory = f'{generation_time}_multiband'
        tile_output_directories = [
            os.path.join(output, result_directory)
        ]
    else:
        generation_time = str(int(time.time()))
        result_directory = f'{generation_time}_singleband'
        tile_output_directories = []
        for band in band_numbers:
            tile_output_directories.append(
                os.path.join(output, result_directory, str(band))
            )
    extract_tasks = []
    for input, band_number in zip(inputs, band_numbers):
        extract_tasks.append(
            (input, band_number, temp_dir.name)
        )
    extracted_bands = process_map(extract_band, extract_tasks, max_workers=threads, desc='Извлечение каналов')
    warp_tasks = []
    if cutline_filename:
        for extracted_band in extracted_bands:
            input_filename = extracted_band[0]
            band = extracted_band[1]
            output_directory = extracted_band[2]
            warp_tasks.append(
                [input_filename, band, output_directory, output_crs_name, None, None, cutline_filename, cutline_layer]
            )
    else:
        if output_crs_name == 'EPSG:3857':
            extracted_bands = process_map(precut_bands, extracted_bands, max_workers=threads, desc='Предварительное перепроецирование каналов')
        for extracted_band in extracted_bands:
            input_filename = extracted_band[0]
            band = extracted_band[1]
            output_directory = extracted_band[2]
            warp_tasks.append(
                [input_filename, band, output_directory, output_crs_name, output_crs.area_of_use.bounds,
                 'EPSG:4326', None, None]
            )
    warped_bands = process_map(warp_bands, warp_tasks, max_workers=threads, desc='Перепроецирование каналов')
    inrange_tasks = []
    for warped_band in warped_bands:
        input_filename = warped_band[0]
        band = warped_band[1]
        inrange_tasks.append(
            [input_filename, band]
        )
    bands_inranges = process_map(calculate_inrange_bands, inrange_tasks, max_workers=threads, desc='Расчёт мин/макс значений')
    print(bands_inranges)
    temp_dir.cleanup()


if __name__ == '__main__':
    try:
        grib_tiler()
    except:
        temp_dir.cleanup()
        sys.exit()
