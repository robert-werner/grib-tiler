import json
import os
import sys
import tempfile
import time
import warnings
from copy import deepcopy

import mercantile
import rasterio
from click import echo, UsageError, command
from pyproj import CRS
from tqdm.contrib.concurrent import process_map
from utils.click_handlers import bands_handler

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks.executors import extract_band, precut_bands, warp_bands, calculate_inrange_bands, \
    transalte_bands_to_byte, concatenate_bands
from grib_tiler.utils import click_options
from grib_tiler.utils.click_handlers import zooms_handler

warnings.filterwarnings("ignore")

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

temp_dir = tempfile.TemporaryDirectory()

META_INFO = {
    "meta": {
        "common": [
        ]
    }
}

@command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.files_in_arg
@click_options.file_out_arg
@click_options.bands_opt
@click_options.image_format_opt
@click_options.cutline_filename_opt
@click_options.cutline_layer_opt
@click_options.tilesize
@click_options.output_crs_opt
@click_options.threads_opt
@click_options.zooms_list_opt
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
    bands_numbers = bands_handler(band_numbers)
    bands_numbers_quantity = len(bands_numbers)
    inputs = []
    if input_files_quantity > 1:
        if multiband:
            if not bands_numbers:
                for input_file in input:
                    with rasterio.open(input_file) as input_rio_ds:
                        bands_numbers.append(input_rio_ds.indexes)
                        bands_numbers_quantity += len(bands_numbers)
                        inputs.extend([input_file] * len(bands_numbers))
            elif bands_numbers:
                if not all(elem == bands_numbers[0] for elem in bands_numbers):
                    inputs.extend(input)
                    if bands_numbers_quantity != input_files_quantity:
                        raise UsageError('Количество каналов должно соответствовать количеству входных файлов')
                    bands_numbers = [[band_number] for band_number in bands_numbers]
                else:
                    if len(bands_numbers) > 1:
                        inputs.extend(input)
                        bands_numbers = [[band_number] for band_number in bands_numbers]
                    else:
                        inputs.extend(input)
                        bands_numbers = [[bands_numbers[0]]] * input_files_quantity


        else:
            raise UsageError('Создавать одноканальные тайлы из множества (больше одного) входных файлов запрещено')
    else:
        if multiband:
            if bands_numbers_quantity == 1:
                raise UsageError('Мультиканальные тайлы с одним каналом создавать запрещено')
            elif not bands_numbers:
                with rasterio.open(input[0]) as input_rio_ds:
                    bands_numbers.append([input_rio_ds.indexes]) # Оборачиваем в ещё один массив для удобства
                    bands_numbers_quantity += len(bands_numbers)
                    inputs.extend([input[0]] * bands_numbers_quantity)
            elif bands_numbers:
                bands_numbers = [bands_numbers] # Оборачиваем в ещё один массив для удобства
                inputs.extend([input[0]] * bands_numbers_quantity)
        else:
            if not bands_numbers:
                with rasterio.open(input[0]) as input_rio_ds:
                    bands_numbers.append(input_rio_ds.indexes) # Оборачиваем в ещё один массив для удобства
                    bands_numbers_quantity += len(bands_numbers)
                    inputs.extend([input[0]] * bands_numbers_quantity)
            else:
                inputs.extend([input[0]] * bands_numbers_quantity)
                bands_numbers = [[band_number] for band_number in bands_numbers]

    if multiband:
        generation_time = str(int(time.time()))
        result_directory = f'{generation_time}_multiband'
        tile_output_directory = os.path.join(output, result_directory)
        os.makedirs(tile_output_directory, exist_ok=True)
        tile_output_directories = [
            tile_output_directory
        ]
    else:
        generation_time = str(int(time.time()))
        result_directory = f'{generation_time}_singleband'
        tile_output_directories = []
        for band_numbers in bands_numbers:
            for band in band_numbers:
                tile_output_directory = os.path.join(output, result_directory, str(band))
                os.makedirs(tile_output_directory, exist_ok=True)
                tile_output_directories.append(
                    os.path.join(output, result_directory, str(band))
                )

    extract_tasks = []
    for input, band_number in zip(inputs, bands_numbers):
        extract_tasks.append(
                (input, band_number, temp_dir.name)
        )
    extracted_bands = process_map(extract_band, extract_tasks, max_workers=threads, desc='Извлечение каналов')
    if multiband and input_files_quantity > 1:
        input_filenames = [extracted_band[0] for extracted_band in extracted_bands]
        concatenated_bands = concatenate_bands([input_filenames, temp_dir.name])
        extracted_bands = [[concatenated_bands, None, temp_dir.name]]
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
    byte_bands_tasks = []
    for tile_output_directory, band_inranges in zip(tile_output_directories, bands_inranges):
        meta_json_filename = os.path.join(tile_output_directory, 'meta.json')
        meta_info = deepcopy(META_INFO)
        for band_inrange in band_inranges[2]:
            meta_info['meta']['common'].append(
                {'step': (band_inrange[1] - band_inrange[0]) / 255,
                 'min': band_inrange[0]}
            )
        with open(meta_json_filename, 'w') as meta_json:
            json.dump(meta_info, meta_json, indent=4)
    for bands_inrange in bands_inranges:
        byte_bands_tasks.append(
            [bands_inrange[0], bands_inrange[2], temp_dir.name]
        )
    byte_bands = process_map(transalte_bands_to_byte, byte_bands_tasks, max_workers=threads, desc='Рендеринг для тайлирования')
    print(byte_bands)
    temp_dir.cleanup()


if __name__ == '__main__':
    try:
        grib_tiler()
    except:
        temp_dir.cleanup()
        sys.exit()
