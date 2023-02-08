import os
import os
import tempfile
import warnings

import click
import fiona
import mercantile
import morecantile
import rasterio
import rasterio.mask
from pyproj import CRS
from rasterio.apps.vrt import build_vrt
from tqdm.contrib.concurrent import process_map

from grib_tiler.steps import seek_by_meta_value, extract_band, warp_band, get_in_ranges, grep_meta, render_tile, \
    write_metainfo
from grib_tiler.utils import click_options
from grib_tiler.utils.click_handlers import bands_handler, zooms_handler
from grib_tiler.utils.step_namedtuples import TranslateTask, WarpTask, InRangeTask, VRTask, RenderTileTask, MetaInfoTask

warnings.filterwarnings("ignore")
os.environ['CPL_LOG'] = '/dev/null'


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
               uv):
    tms = morecantile.TileMatrixSet.custom(extent=CRS.from_user_input(output_crs).area_of_use.bounds,
                                           extent_crs=CRS.from_epsg(4326),
                                           crs=CRS.from_user_input(output_crs))
    os.makedirs(output, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory()

    zooms = zooms_handler(zooms)

    if not cutline:
        cutline_layer = None
        with rasterio.open(inputs[0]) as input:
            render_bounds = input.bounds
            tiles_list = list(mercantile.tiles(*render_bounds, zooms))
    else:
        with fiona.open(cutline) as cutline_lyr:
            render_bounds = cutline_lyr.bounds
            tiles_list = list(mercantile.tiles(*render_bounds, zooms))

    if uv:
        uv_in_ranges, uv_pairs_translated = uv_handler(cutline, cutline_layer, inputs, output_crs, temp_dir, threads)
        metainfo_writer(output, uv_in_ranges, uv_pairs_translated)
        render_tile_task = []
        for tile in tiles_list:
            for filename, in_range in zip(uv_pairs_translated, uv_in_ranges):
                render_tile_task.append(
                    RenderTileTask(input_fn=filename, z=tile.z, x=tile.x, y=tile.y, tms=tms, nodata=nodata,
                                   in_range=in_range, tilesize=tilesize,
                                   img_format=img_format, band_name=os.path.splitext(os.path.basename(filename))[0],
                                   output_dir=output)
                )
        process_map(render_tile, render_tile_task, max_workers=threads, desc='Рендеринг тайлов')
    else:
        inputs = inputs[0]
        in_ranges, translated = band_handler(cutline, cutline_layer, inputs, output_crs, temp_dir, threads, bands)
        metainfo_writer(output, in_ranges, translated)
        render_tile_task = []
        for tile in tiles_list:
            for filename, in_range in zip(translated, in_ranges):
                render_tile_task.append(
                    RenderTileTask(input_fn=filename, z=tile.z, x=tile.x, y=tile.y, tms=tms, nodata=nodata,
                                   in_range=in_range, tilesize=tilesize,
                                   img_format=img_format, band_name=os.path.splitext(os.path.basename(filename))[0],
                                   output_dir=output)
                )
        process_map(render_tile, render_tile_task, max_workers=threads, desc='Рендеринг тайлов')




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
                                  [TranslateTask(input_fn=input, output_dir=temp_dir.name,
                                                 band=band, output_format='GRIB', output_format_extension='.grib2')
                                   for band in bands],
                                  max_workers=threads,
                                  desc='Извлечение выбранных каналов')
    in_ranges = process_map(get_in_ranges,
                            [InRangeTask(input_fn=input, band=1) for input in extracted_bands],
                            max_workers=threads,
                            desc='Вычисление мин/макс выбранных каналов')
    warped_bands = process_map(warp_band,
                               [WarpTask(input_fn=u_fn, output_dir=temp_dir.name,
                                         output_crs=output_crs, multithreaded=True,
                                         cutline_fn=cutline,
                                         cutline_layername=cutline_layer, output_format='GTiff',
                                         src_nodata=0, dst_nodata=0,
                                         write_flush=True) for u_fn in extracted_bands],
                               max_workers=threads,
                               desc='Перепроецирование выбранных каналов')
    return in_ranges, warped_bands


def uv_handler(cutline, cutline_layer, inputs, output_crs, temp_dir, threads):
    u_extracts = []
    v_extracts = []
    if len(inputs) > 2 or len(inputs) < 1:
        click.echo('Допускаются на вход только один или два файла, содержащие U- и V- компоненты')
        raise click.Abort()
    if len(inputs) == 2:
        u_extracts.append(inputs[0])
        v_extracts.append(inputs[1])
    elif len(inputs) == 1:
        inputs = inputs[0]
        uv_seek_results = seek_by_meta_value(inputs, GRIB_ELEMENT=['UGRD', 'VGRD'])
        u_list = uv_seek_results['UGRD']
        v_list = uv_seek_results['VGRD']
        if len(u_list) != len(v_list):
            click.echo('Входной GRIB-файл непригоден для генерации UV-тайлов')
            raise click.Abort()

        u_extracts = process_map(extract_band,
                                 [TranslateTask(input_fn=inputs, output_dir=temp_dir.name, band=band,
                                                output_format='GRIB',
                                                output_format_extension='.grib2') for band in u_list],
                                 max_workers=threads,
                                 desc='Извлечение U-компоненты ветра')
        v_extracts = process_map(extract_band,
                                 [TranslateTask(input_fn=inputs, output_dir=temp_dir.name, band=band,
                                                output_format='GRIB',
                                                output_format_extension='.grib2') for band in v_list],
                                 max_workers=threads,
                                 desc='Извлечение V-компоненты ветра')
    u_inranges = process_map(get_in_ranges,
                             [InRangeTask(input_fn=input, band=1) for input in u_extracts],
                             max_workers=threads,
                             desc='Вычисление мин/макс U-компонент')
    v_inranges = process_map(get_in_ranges,
                             [InRangeTask(input_fn=input, band=1) for input in v_extracts],
                             max_workers=threads,
                             desc='Вычисление мин/макс V-компонент')
    uv_inranges = list(zip(u_inranges, v_inranges))
    grib_shortnames = [grep_meta(u_fn, 'GRIB_SHORT_NAME') for u_fn in u_extracts]
    warp_u_filenames = process_map(warp_band,
                                   [WarpTask(input_fn=u_fn, output_dir=temp_dir.name,
                                             output_crs=output_crs, multithreaded=True,
                                             cutline_fn=cutline,
                                             cutline_layername=cutline_layer, output_format='GTiff',
                                             src_nodata=0, dst_nodata=0,
                                             write_flush=True) for u_fn in u_extracts],
                                   max_workers=threads,
                                   desc='Перепроецирование U-компонент ветра')
    warp_v_filenames = process_map(warp_band,
                                   [WarpTask(input_fn=v_fn, output_dir=temp_dir.name,
                                             output_crs=output_crs, multithreaded=True,
                                             cutline_fn=cutline,
                                             cutline_layername=cutline_layer, output_format='GTiff',
                                             src_nodata=0, dst_nodata=0,
                                             write_flush=True) for v_fn in v_extracts],
                                   max_workers=threads,
                                   desc='Перепроецирование V-компонент ветра')
    uv_pair_vrt_filenames = process_map(buildvrt_wrapper,
                                        [VRTask(list(uv_fns),
                                                os.path.join(temp_dir.name, f'{grib_shortname[0]}.vrt'))
                                         for uv_fns, grib_shortname in
                                         zip(list(zip(warp_u_filenames, warp_v_filenames)), grib_shortnames)],
                                        max_workers=threads,
                                        desc='Объединение U- и V-компонент ветра')
    uv_pairs_translated = process_map(extract_band,
                                      [TranslateTask(uv_pair_vrt_filename, temp_dir.name, None, 'GTiff',
                                                     '.tiff') for
                                       uv_pair_vrt_filename in uv_pair_vrt_filenames], max_workers=threads,
                                      desc='Рендеринг U- и V- компонент')
    return uv_inranges, uv_pairs_translated


if __name__ == '__main__':
    grib_tiler()
