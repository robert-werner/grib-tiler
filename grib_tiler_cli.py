import json
import multiprocessing
import os
import sys
import tempfile
import traceback
import warnings
from copy import deepcopy

import mercantile
import numpy as np
from click import UsageError, echo, command
from pyproj import CRS

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks import RenderTileTask
from grib_tiler.tasks.executors import extract_band, warp_band, calculate_band_minmax, transalte_bands_to_byte, \
    concatenate_bands, vrt_to_raster, render_tile, band_isolines
from grib_tiler.utils import click_options, get_rfc3339nano_time

warnings.filterwarnings("ignore")

TEMP_DIR = tempfile.TemporaryDirectory()

EPSG_4326 = CRS.from_epsg(4326)
EPSG_4326_BOUNDS = list(EPSG_4326.area_of_use.bounds)

META_INFO = {
    "meta": {
        "common": [
        ]
    }
}

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
@click_options.tilesize
@click_options.image_format_opt
@click_options.isolines_generate_opt
def grib_tiler(input_files,
               output_directory,
               cutline_filename,
               cutline_layer,
               bands_list,
               zooms_list,
               is_multiband,
               output_crs,
               threads,
               tilesize,
               image_format,
               generate_isolines):
    tms = load_tms(output_crs)

    input_pack = None
    bands_list = list(map(int, bands_list.split(',')))
    zooms_list = list(map(int, zooms_list.split(',')))
    output_crs_bounds = CRS.from_user_input(output_crs).area_of_use.bounds

    echo({"level": "info", "time": get_rfc3339nano_time(), "msg": "Генерация номеров тайлов..."})
    echo({"level": "info", "time": get_rfc3339nano_time(),
          "msg": f"Генерация номеров тайлов... OK"})
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
        echo({"level": "info", "time": get_rfc3339nano_time(), "msg": "Тип выходных тайлов: одноканальный"})
        if len(input_files) >= 2:
            echo({"level": "fatal", "time": get_rfc3339nano_time(), "msg": "Использование двух и более входных файлов в одноканальном режиме недоступно"})
            raise UsageError('Использование двух и более входных файлов в одноканальном режиме недоступно')
        input_pack = list(zip(input_files * len(bands_list),
                              bands_list,
                              [TEMP_DIR.name] * len(bands_list)))
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
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Извлечение каналов из входных файлов... ОК"})
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
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Перепроецирование извлечённых каналов из входных файлов... ОК"})
    with multiprocessing.Pool(threads) as inrange_pool:
        in_range_calc_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Вычисление мин/макс каналов..."})
        warped_minmax = []
        meta_infos = []
        meta_info = deepcopy(META_INFO)
        for result in inrange_pool.map(calculate_band_minmax, warped_extracts):
            in_range_calc_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Вычисление мин/макс каналов... {int(in_range_calc_progress)}%"})
            warped_minmax.append(result)
            meta_info['meta']['common'].append(
                {
                  'step': (result[1] - result[0]) / 255,
                  'min': result[0]
                }
            )
        if not is_multiband:
            for step_min in meta_info['meta']['common']:
                meta_info = deepcopy(META_INFO)
                meta_info['meta']['common'].append({
                    'step': step_min['step'],
                    'min': step_min['min']
                })
                meta_infos.append(
                    meta_info
                )
        else:
            meta_infos = [meta_info]
        echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Вычисление мин/макс каналов... ОК"})
    output_directories = []
    if is_multiband:
        os.makedirs(output_directory, exist_ok=True)
        output_directories.append(output_directory)
        meta_json_filename = os.path.join(output_directory, 'meta.json')
        with open(meta_json_filename, 'w') as meta_json:
            json.dump(meta_infos[0], meta_json, indent=4)
    else:
        for idx, band in enumerate(bands_list):
            band_tiles_output_directory = os.path.join(output_directory, str(band))
            os.makedirs(band_tiles_output_directory, exist_ok=True)
            output_directories.append(band_tiles_output_directory)
            meta_json_filename = os.path.join(band_tiles_output_directory, 'meta.json')
            with open(meta_json_filename, 'w') as meta_json:
                json.dump(meta_infos[idx], meta_json, indent=4)

    if generate_isolines:
        band_isolines_list = []
        isolines_generation_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация изолиний..."})
        isolines_tasks = list(zip(warped_extracts, [TEMP_DIR.name] * len(warped_extracts)))
        for isoline_task in isolines_tasks:
            isoline = band_isolines(isoline_task)
            band_isolines_list.append(isoline)
            isolines_generation_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Генерация изолиний... {isolines_generation_progress}%"})

        if is_multiband:
            for band_isoline, band in zip(band_isolines_list, bands_list):
                with open(os.path.join(output_directories[0], f'{str(band)}_isoline.json'), 'w') as isoline_json:
                    json.dump(band_isoline, isoline_json)
        else:
            for output_directory, band_isoline in zip(output_directories, band_isolines_list):
                with open(os.path.join(output_directory, f'isoline.json'), 'w') as isoline_json:
                    json.dump(band_isoline, isoline_json, indent=4)
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация изолиний... OK"})

    with multiprocessing.Pool(threads) as byte_conv_pool:
        byte_conv_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Конверсия извлечённых каналов в 8-битные изображения..."})
        byte_converted = []
        byte_conv_tasks = list(zip(warped_extracts, [[warped_minmax_elem] for warped_minmax_elem in warped_minmax], [TEMP_DIR.name] * len(bands_list)))
        for result in byte_conv_pool.map(transalte_bands_to_byte, byte_conv_tasks):
            byte_conv_progress += band_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Конверсия извлечённых каналов в 8-битные изображения... {int(byte_conv_progress)}%"})
            byte_converted.append(result)
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Конверсия извлечённых каналов в 8-битные изображения... ОК"})
    tiling_source_files = []
    if is_multiband:
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Объединение и рендеринг 8-битных изображений..."})
        concatenate_args = [byte_converted, TEMP_DIR.name]
        tiling_source_file_vrt = concatenate_bands(concatenate_args)
        tiling_source_file = vrt_to_raster([tiling_source_file_vrt, TEMP_DIR.name])
        tiling_source_files.append(tiling_source_file)
        echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Объединение и рендеринг 8-битных изображений... OK"})
        render_tiles_quantity = len(tiles)
    else:
        with multiprocessing.Pool(threads) as vrt_to_raster_pool:
            vrt_to_raster_progress = 0
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Рендеринг 8-битных изображений..."})
            for result in vrt_to_raster_pool.map(vrt_to_raster, list(zip(byte_converted, [TEMP_DIR.name] * len(bands_list)))):
                vrt_to_raster_progress += band_progress_step
                echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Рендеринг 8-битных изображений... {int(vrt_to_raster_progress)}%"})
                tiling_source_files.append(result)
            echo({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Рендеринг 8-битных изображений... OK"})
        render_tiles_quantity = len(tiles) * len(bands_list)
    render_tiles_progress_step = 100 / render_tiles_quantity
    with multiprocessing.Pool(threads) as render_tile_pool:
        render_tile_tasks = []
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация задач на тайлирование изображений..."}) # TODO: сделать всё это в _ленивом_ итерировании (tiles)
        tiling_task_generation_progress = 0
        for tile in tiles:
            for band, band_output_directory, tiling_source_file in zip(bands_list, output_directories,
                                                                       tiling_source_files):
                tiling_task_generation_progress += render_tiles_progress_step
                echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Генерация задач на тайлирование изображений... {int(tiling_task_generation_progress)}%"})
                nodata_mask = None
                if len(bands_list) > 4 and is_multiband:
                    image_format = 'GTIFF'
                if len(bands_list) == 2 and is_multiband:
                    nodata_mask = np.zeros((tilesize, tilesize), dtype='uint8')
                render_tile_tasks.append(
                    RenderTileTask(
                        input_filename=tiling_source_file,
                        output_directory=band_output_directory,
                        z=tile.z,
                        x=tile.x,
                        y=tile.y,
                        tms=tms,
                        tilesize=tilesize,
                        image_format=image_format,
                        nodata_mask_array=nodata_mask,
                        bands=bands_list
                    )
                )
        tiling_progress = 0
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация задач на тайлирование изображений... OK"})
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Тайлирование изображений..."})
        for result in render_tile_pool.map(render_tile, render_tile_tasks):
            tiling_progress += render_tiles_progress_step
            echo({"level": "info", "time": get_rfc3339nano_time(), "msg": f"Тайлирование изображений... {int(tiling_progress)}%"})
        echo({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Тайлирование изображений... OK"})
        TEMP_DIR.cleanup()






if __name__ == '__main__':
    try:
        grib_tiler()
    except Exception as e:
        echo({"level": "fatal", "time": get_rfc3339nano_time(),        "msg": traceback.format_exc()})
        TEMP_DIR.cleanup()
        sys.exit()
