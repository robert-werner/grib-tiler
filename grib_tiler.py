import multiprocessing
import os
import sys
import tempfile
import traceback

import mercantile
from click import UsageError, echo, command
from pyproj import CRS

from grib_tiler.tasks.executors import extract_band, warp_band
from grib_tiler.utils import click_options, get_rfc3339nano_time

TEMP_DIR = tempfile.TemporaryDirectory()

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

@command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.input_files_arg
@click_options.output_directory_arg
@click_options.cutline_filename_opt
@click_options.cutline_layer_opt
@click_options.bands_list_opt
@click_options.zooms_list_opt
@click_options.multiband_opt
@click_options.output_crs_opt
@click_options.threads_opt
def grib_tiler(input_files,
               output_directory,
               cutline_filename,
               cutline_layer,
               bands_list,
               zooms_list,
               is_multiband,
               output_crs,
               threads):
    input_pack = None
    bands_list = list(map(int, bands_list.split(',')))
    zooms_list = list(map(int, zooms_list.split(',')))
    output_crs_bounds = CRS.from_user_input(output_crs).area_of_use.bounds

    echo({"level": "info", "time": get_rfc3339nano_time(), "msg": "Генерация номеров тайлов..."})
    tiles = list(mercantile.tiles(*EPSG_4326_BOUNDS, zooms_list)) # TODO: сделать генерацию номеров тайлов либо на Cython+OpenMP, либо на OpenCL

    if is_multiband:
        echo({"level": "info", "time": get_rfc3339nano_time(), "msg": "Тип выходных тайлов: мультиканальный"})
        if (len(input_files) != len(bands_list)) and (len(bands_list) != 1) and (len(input_files) != 1):
            echo({"level": "fatal", "time": get_rfc3339nano_time(), "msg": "Количество каналов и входных файлов должно быть равно, или должен быть указан только один канал"})
            raise UsageError(
                'Количество каналов и входных файлов должно быть равно, или должен быть указан только один канал')
        elif (len(input_files) != len(bands_list)) and (len(bands_list) == 1):
            bands_list = bands_list * len(input_files)
        elif (len(input_files) == 1) and (len(input_files) != len(bands_list)):
            input_files = input_files * len(bands_list)
        input_pack = list(zip(input_files,
                              bands_list,
                              [TEMP_DIR.name] * len(input_files)))
    else:
        if len(input_files) >= 2:
            echo({"level": "fatal", "time": get_rfc3339nano_time(), "msg": "Использование двух и более входных файлов в одноканальном режиме недоступно"})
            raise UsageError('Использование двух и более входных файлов в одноканальном режиме недоступно')
        input_pack = list(zip(input_files,
                              bands_list,
                              [TEMP_DIR.name]))
    band_progress_step = 100 / len(bands_list)
    with multiprocessing.Pool(threads) as extract_pool:
        band_extract_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Извлечение каналов из входных файлов..."})
        extracted_bands = []
        for result in extract_pool.map(extract_band, input_pack):
            band_extract_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Извлечение каналов из входных файлов... {int(band_extract_progress)}%"})
            extracted_bands.append([result, output_crs, output_crs_bounds, cutline_filename, cutline_layer, TEMP_DIR.name])
    with multiprocessing.Pool(threads) as warp_pool:
        warp_extract_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Перепроецирование извлечённых каналов из входных файлов..."})
        warped_extracts = []
        for result in warp_pool.map(warp_band, extracted_bands):
            warp_extract_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Перепроецирование извлечённых каналов из входных файлов... {int(warp_extract_progress)}%"})
            warped_extracts.append(result)
    with multiprocessing.Pool(threads) as inrange_pool:
        in_range_calc_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Вычисление мин/макс каналов..."})
        for result in inrange_pool.map():
            in_range_calc_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Вычисление мин/макс каналов... {int(in_range_calc_progress)}%"})


if __name__ == '__main__':
    try:
        grib_tiler()
    except Exception as e:
        echo({"level": "fatal", "time": get_rfc3339nano_time(),        "msg": traceback.format_exc()})
        TEMP_DIR.cleanup()
        sys.exit()
