import glob
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
import warnings
from copy import deepcopy

from shapely import geometry
from shapely.geometry import box
import mercantile
import numpy as np
from click import UsageError, echo, command
from pyproj import CRS

from grib_tiler.data.tms import load_tms
from grib_tiler.tasks import RenderTileTask
from grib_tiler.tasks.executors import extract_band, warp_band, calculate_band_minmax, transalte_bands_to_byte, \
    concatenate_bands, vrt_to_raster, render_tile, band_isolines
from grib_tiler.utils import click_options, get_rfc3339nano_time

from rasterio.cutils.bounds import extent  # TODO: посмотреть код GDALWarp(), выяснить, происходит ли обрезка по пределам СК или входного изображения

os.environ['GDAL_PAM_ENABLED'] = 'NO'

warnings.filterwarnings("ignore")

TEMP_DIR = tempfile.TemporaryDirectory()

EPSG_3857 = CRS.from_epsg(3857)
EPSG_3857_BOUNDS = list(EPSG_3857.area_of_use.bounds)

META_INFO = {
    "common": []
}

input_files_list = None

@command(short_help='Генератор растровых тайлов из GRIB(2)-файлов.')
@click_options.input_files_arg
@click_options.output_directory_arg
@click_options.cutline_filename_opt
@click_options.bands_list_opt
@click_options.zooms_list_opt
@click_options.multiband_opt
@click_options.output_crs_opt
@click_options.threads_opt
@click_options.tilesize
@click_options.image_format_opt
@click_options.isolines_generate_opt
@click_options.isolines_elev_interval_opt
@click_options.isolines_simplify_epsilon_opt
@click_options.equator_opt
@click_options.transparency_opt
@click_options.nodata_opt
def grib_tiler(input_files,
               output_directory,
               cutline_filename,
               bands_list,
               zooms_list,
               is_multiband,
               output_crs,
               threads,
               tilesize,
               image_format,
               generate_isolines,
               isolines_elevation_interval,
               isolines_simplify_epsilon,
               get_equator,
               transparency_percent,
               output_nodata):
    global input_files_list
    input_files_list = input_files

    tms = load_tms(output_crs, tilesize)



    input_pack = None
    bands_list = list(map(int, bands_list.split(',')))
    zooms_list = list(map(int, zooms_list.split(',')))
    input_files_bounds = []

    if transparency_percent:
        image_format = 'PNG'

    if get_equator:
        input_files_bounds = []
        for input_file in input_files:
            input_file_bounds = extent(input_file, True)
            if get_equator == 'northern':
                input_file_bounds[1] = 0.0
                input_file_bounds[3] = float(int(input_file_bounds[3]))
            elif get_equator == 'southern':
                input_file_bounds[1] = float(int(input_file_bounds[1]))
                input_file_bounds[3] = 0.0

            extent_geojson = {}
            extent_geojson["type"] = "FeatureCollection"
            extent_geojson["name"] = "equator"
            extent_geojson["features"] = [{
                "type": "Feature",
                "properties": {},
                "geometry": json.loads(json.dumps(geometry.mapping(box(*input_file_bounds))))
            }]
            extent_fp_fn = os.path.join(TEMP_DIR.name, os.path.basename(input_file.replace(' ', '_')) + '.geojson')
            with open(extent_fp_fn, 'w') as extent_fp:
                json.dump(extent_geojson, extent_fp)
            input_files_bounds.append(extent_fp_fn)

    echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(), "msg": "Генерация номеров тайлов..."}, ensure_ascii=False))
    echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
          "msg": f"Генерация номеров тайлов... OK"}, ensure_ascii=False))
    tiles = list(mercantile.tiles(*EPSG_3857_BOUNDS,
                                  zooms_list))  # TODO: сделать генерацию номеров тайлов либо на Cython+OpenMP, либо на OpenCL
    if is_multiband:
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(), "msg": "Тип выходных тайлов: мультиканальный"}, ensure_ascii=False))
        if (len(input_files) != len(bands_list)) and (len(bands_list) != 1) and (len(input_files) != 1):
            echo(json.dumps({"level": "fatal", "time": get_rfc3339nano_time(),
                  "msg": "Количество каналов и входных файлов должно быть равно, или должен быть указан только один канал"}, ensure_ascii=False))
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
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(), "msg": "Тип выходных тайлов: одноканальный"}, ensure_ascii=False))
        if len(input_files) >= 2:
            echo(json.dumps({"level": "fatal", "time": get_rfc3339nano_time(),
                  "msg": "Использование двух и более входных файлов в одноканальном режиме недоступно"}, ensure_ascii=False))
            raise UsageError('Использование двух и более входных файлов в одноканальном режиме недоступно')
        input_pack = list(zip(input_files * len(bands_list),
                              bands_list,
                              [TEMP_DIR.name] * len(bands_list)))
    band_progress_step = 100 / len(bands_list)
    with multiprocessing.Pool(threads) as extract_pool:
        band_extract_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Извлечение каналов из входных файлов..."}, ensure_ascii=False))
        extracted_cropped_bands = []
        for result in extract_pool.map(extract_band, input_pack):
            band_extract_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Извлечение каналов из входных файлов... {int(band_extract_progress)}%"}, ensure_ascii=False))
            band_bounds = extent(result, True)
            warp_band_args = [result, 'EPSG:4326', band_bounds, 'EPSG:4326', None, None, TEMP_DIR.name, True, output_nodata]
            warped_band = warp_band(warp_band_args)
            if cutline_filename:
                extracted_cropped_bands.append(
                        [warped_band, 'EPSG:4326', None, None, cutline_filename, None, TEMP_DIR.name, True, output_nodata])
            else:
                if get_equator:
                    extracted_cropped_bands.append(
                        [warped_band, 'EPSG:4326', None, None, input_files_bounds[0], None, TEMP_DIR.name, True, output_nodata])
                else:
                    extracted_cropped_bands.append(
                        [warped_band, 'EPSG:4326', band_bounds, 'EPSG:4326', None, None, TEMP_DIR.name, True, output_nodata])


        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Извлечение каналов из входных файлов... ОК"}, ensure_ascii=False))
    with multiprocessing.Pool(threads) as warp_cropped_3857_pool:
        warp_cropped_extract_progress = 0
        echo(json.dumps({
            "level": "info",
            "time": get_rfc3339nano_time(),
            "msg": f"Предварительное перепроецирование в EPSG:4326 извлечённых каналов из входных файлов..."
        }, ensure_ascii=False))
        warped_cropped_3857_extracts = []
        for result in warp_cropped_3857_pool.map(warp_band, extracted_cropped_bands):
            warp_cropped_extract_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Предварительное перепроецирование в EPSG:4326 извлечённых каналов из входных файлов... {int(warp_cropped_extract_progress)}%"}, ensure_ascii=False))
            warped_cropped_3857_extracts.append(result)
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Предварительное перепроецирование в EPSG:4326 извлечённых каналов из входных файлов... ОК"}, ensure_ascii=False))
    with multiprocessing.Pool(threads) as inrange_pool:
        in_range_calc_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Вычисление мин/макс каналов..."}, ensure_ascii=False))
        warped_minmax = []
        meta_infos = []
        meta_info = deepcopy(META_INFO)
        for result in inrange_pool.map(calculate_band_minmax, warped_cropped_3857_extracts):
            in_range_calc_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Вычисление мин/макс каналов... {int(in_range_calc_progress)}%"}, ensure_ascii=False))
            warped_minmax.append(result)
            meta_info['common'].append(
                {
                    'step': (result[1] - result[0]) / 255,
                    'min': result[0]
                }
            )
        if not is_multiband:
            for step_min in meta_info['common']:
                meta_info = deepcopy(META_INFO)
                meta_info['common'].append({
                    'step': step_min['step'],
                    'min': step_min['min']
                })
                meta_infos.append(
                    meta_info
                )
        else:
            meta_infos = [meta_info]
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Вычисление мин/макс каналов... ОК"}, ensure_ascii=False))

    output_directories = []
    if is_multiband:
        os.makedirs(output_directory, exist_ok=True)
        output_directories.append(output_directory)
        meta_json_filename = os.path.join(output_directory, 'meta.json')
        with open(meta_json_filename, 'w') as meta_json:
            json.dump(meta_infos[0], meta_json)
    else:
        for idx, band in enumerate(bands_list):
            band_tiles_output_directory = os.path.join(output_directory, str(band))
            os.makedirs(band_tiles_output_directory, exist_ok=True)
            output_directories.append(band_tiles_output_directory)
            meta_json_filename = os.path.join(band_tiles_output_directory, 'meta.json')
            with open(meta_json_filename, 'w') as meta_json:
                json.dump(meta_infos[idx], meta_json)

    if generate_isolines:
        band_isolines_list = []
        isolines_generation_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация изолиний..."}, ensure_ascii=False))
        isolines_tasks = list(zip(warped_cropped_3857_extracts,
                                  [TEMP_DIR.name] * len(warped_cropped_3857_extracts),
                                  [isolines_elevation_interval] * len(warped_cropped_3857_extracts),
                                  [isolines_simplify_epsilon] * len(warped_cropped_3857_extracts)))
        for isoline_task in isolines_tasks:
            isoline = band_isolines(isoline_task)
            band_isolines_list.append(isoline)
            isolines_generation_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Генерация изолиний... {isolines_generation_progress}%"}, ensure_ascii=False))

        for output_directory, band_isoline in zip(output_directories, band_isolines_list):
            with open(os.path.join(output_directory, f'contours.json'), 'w') as isoline_json:
                json.dump(band_isoline, isoline_json)

        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация изолиний... OK"}, ensure_ascii=False))
    with multiprocessing.Pool(threads) as warp_3857_pool:
        warp_extract_progress = 0
        echo(json.dumps({
            "level": "info",
            "time": get_rfc3339nano_time(),
            "msg": f"Пперепроецирование в EPSG:4326 извлечённых каналов из входных файлов..."
        }, ensure_ascii=False))
        warped_3857_extracts = []
        for result in warp_3857_pool.map(warp_band, extracted_cropped_bands):
            warp_cropped_extract_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Перепроецирование в EPSG:4326 извлечённых каналов из входных файлов... {int(warp_extract_progress)}%"}, ensure_ascii=False))
            warped_3857_extracts.append(result)
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Перепроецирование в EPSG:4326 извлечённых каналов из входных файлов... ОК"}, ensure_ascii=False))

    input_packs = []

    if cutline_filename:
        for input_file in warped_3857_extracts:
            input_packs.append([
                input_file, output_crs, None,
                None,
                None, None, TEMP_DIR.name, False, output_nodata
            ])
    else:
        if get_equator:
            for input_file in warped_3857_extracts:
                input_packs.append([
                    input_file, output_crs, None,
                    None,
                    None, None, TEMP_DIR.name, False, output_nodata
                ])
        else:
            for input_file in warped_3857_extracts:
                input_packs.append([
                    input_file, output_crs, CRS.from_string(output_crs).area_of_use.bounds,
                    'EPSG:4326',
                    None, None, TEMP_DIR.name, False, output_nodata
                ])

    with multiprocessing.Pool(threads) as warp_pool:
        warp_cropped_extract_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Перепроецирование извлечённых каналов из входных файлов..."}, ensure_ascii=False))
        warped_extracts = []
        for result in warp_pool.map(warp_band, input_packs):
            warp_cropped_extract_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Перепроецирование извлечённых каналов из входных файлов... {int(warp_cropped_extract_progress)}%"}, ensure_ascii=False))
            warped_extracts.append(result)
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Перепроецирование извлечённых каналов из входных файлов... ОК"}, ensure_ascii=False))
    tiling_source_files_original_range = []
    if is_multiband:
        concatenate_args = [warped_extracts, TEMP_DIR.name]
        tiling_source_file_vrt = concatenate_bands(concatenate_args)
        tiling_source_files_original_range.append(tiling_source_file_vrt)
    else:
        tiling_source_files_original_range.extend(warped_extracts)
    with multiprocessing.Pool(threads) as byte_conv_pool:
        byte_conv_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Конверсия извлечённых каналов в 8-битные изображения..."}, ensure_ascii=False))
        byte_converted = []
        byte_conv_tasks = list(zip(warped_extracts, [[warped_minmax_elem] for warped_minmax_elem in warped_minmax],
                                   [TEMP_DIR.name] * len(bands_list)))
        for result in byte_conv_pool.map(transalte_bands_to_byte, byte_conv_tasks):
            byte_conv_progress += band_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Конверсия извлечённых каналов в 8-битные изображения... {int(byte_conv_progress)}%"}, ensure_ascii=False))
            byte_converted.append(result)
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Конверсия извлечённых каналов в 8-битные изображения... ОК"}, ensure_ascii=False))
    tiling_source_files = []
    if is_multiband:
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Объединение и рендеринг 8-битных изображений..."}, ensure_ascii=False))
        concatenate_args = [byte_converted, TEMP_DIR.name]
        tiling_source_file_vrt = concatenate_bands(concatenate_args)
        tiling_source_file = vrt_to_raster([tiling_source_file_vrt, TEMP_DIR.name])
        tiling_source_files.append(tiling_source_file)
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Объединение и рендеринг 8-битных изображений... OK"}, ensure_ascii=False))
        render_tiles_quantity = len(tiles)
    else:
        with multiprocessing.Pool(threads) as vrt_to_raster_pool:
            vrt_to_raster_progress = 0
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Рендеринг 8-битных изображений..."}, ensure_ascii=False))
            for result in vrt_to_raster_pool.map(vrt_to_raster,
                                                 list(zip(byte_converted, [TEMP_DIR.name] * len(bands_list)))):
                vrt_to_raster_progress += band_progress_step
                echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                      "msg": f"Рендеринг 8-битных изображений... {int(vrt_to_raster_progress)}%"}, ensure_ascii=False))
                tiling_source_files.append(result)
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Рендеринг 8-битных изображений... OK"}, ensure_ascii=False))
        render_tiles_quantity = len(tiles) * len(bands_list)
    render_tiles_progress_step = 100 / render_tiles_quantity
    with multiprocessing.Pool(threads) as render_tile_pool:
        render_tile_tasks = []
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация задач на тайлирование изображений..."}, ensure_ascii=False))  # TODO: сделать всё это в _ленивом_ итерировании (tiles)
        tiling_task_generation_progress = 0
        for tile in tiles:
            for band, band_output_directory, tiling_source_file, tiling_source_file_original_range in zip(bands_list, output_directories,
                                                                       tiling_source_files, tiling_source_files_original_range):
                tiling_task_generation_progress += render_tiles_progress_step
                echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                      "msg": f"Генерация задач на тайлирование изображений... {int(tiling_task_generation_progress)}%"}, ensure_ascii=False))
                nodata_mask = None
                if len(bands_list) < 3 and is_multiband and image_format != 'PNG':
                    nodata_mask = np.zeros((tilesize, tilesize), dtype='uint8')
                render_tile_tasks.append(
                    RenderTileTask(
                        input_filename=tiling_source_file,
                        output_directory=band_output_directory,
                        z=tile.z,
                        x=tile.x,
                        y=tile.y,
                        tms=tms,
                        nodata=output_nodata,
                        tilesize=tilesize,
                        image_format=image_format,
                        nodata_mask_array=nodata_mask,
                        bands=bands_list,
                        transparency_percent=transparency_percent,
                        original_range_filename=tiling_source_file_original_range
                    )
                )
        tiling_progress = 0
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Генерация задач на тайлирование изображений... OK"}, ensure_ascii=False))
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Тайлирование изображений..."}, ensure_ascii=False))
        for result in render_tile_pool.map(render_tile, render_tile_tasks):
            tiling_progress += render_tiles_progress_step
            echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
                  "msg": f"Тайлирование изображений... {int(tiling_progress)}%"}, ensure_ascii=False))
        echo(json.dumps({"level": "info", "time": get_rfc3339nano_time(),
              "msg": f"Тайлирование изображений... OK"}, ensure_ascii=False))
        TEMP_DIR.cleanup()
        for input_file_dir in input_files:
            for vrtpath in glob.iglob(os.path.join(os.path.dirname(input_file_dir), '*.vrt')):
                os.remove(vrtpath)


if __name__ == '__main__':
    try:
        grib_tiler()
    except Exception as e:
        echo(json.dumps({"level": "fatal", "time": get_rfc3339nano_time(), "msg": traceback.format_exc()}, ensure_ascii=False))
        TEMP_DIR.cleanup()
        for input_file_dir in input_files_list:
            for vrtpath in glob.iglob(os.path.join(os.path.dirname(input_file_dir), '*.vrt')):
                os.remove(vrtpath)
        sys.exit()
