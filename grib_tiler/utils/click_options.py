import os

from click import option, argument, Path, Choice
from grib_tiler.utils.click_handlers import cpu_count_handler, crs_handler, zoom_handler, grib_handler, band_handler

multiband_opt = option(
    '--multiband',
    'is_multiband',
    is_flag=True,
    default=False,
    help='Генерировать многоканальные тайлы (если каналов больше 3-ёх, то выходной формат тайлов всегда будет TIFF)'
)

cutline_filename_opt = option(
    '--cutline',
    'cutline_filename',
    default=None,
    help='Путь к файлу обрезки растра'
)

cutline_layer_opt = option(
    '--cl',
    '--cutline-layer',
    'cutline_layer',
    default=None,
    help='Слой внутри файла обрезки растра'
)
files_in_arg = argument(
    'input',
    type=Path(resolve_path=True, file_okay=True, dir_okay=False, exists=True),
    nargs=-1,
    required=True,
    metavar="INPUT...")

input_files_arg = argument(
    'input_files',
    type=Path(resolve_path=True, file_okay=True, dir_okay=False, exists=True),
    nargs=-1,
    required=True,
    metavar="INPUT...")

output_directory_arg = argument(
    'output_directory',
    metavar='OUTPUT',
    required=True,
    type=Path(resolve_path=True, file_okay=False))

file_out_arg = argument(
    'output_directory',
    metavar='OUTPUT',
    required=True,
    type=Path(resolve_path=True, file_okay=False))

image_format_opt = option(
    '--f',
    '--format',
    'image_format',
    default='PNG',
    type=Choice([
        'PNG', 'JPEG', 'GTIFF'
    ]),
    help='Выходной формат тайлов.')

tilesize = option(
    '--tilesize',
    "tilesize",
    nargs=1,
    type=int,
    default=256,
    help='Разрешение тайла (квадрат стороной, заданной флагом).')

output_crs_opt = option('--out-crs',
                        'output_crs',
                        default='EPSG:3857',
                        callback=crs_handler,
                        help="Выходная система координат тайлов.")

threads_opt = option('--threads',
                        'threads',
                        callback=cpu_count_handler,
                        type=int, default=os.cpu_count(),
                        help='Количество потоков обработки тайлов.')

nodata_opt = option(
    '--nodata',
    'output_nodata',
    default=0,
    help='Значение "нет данных".',
)

zooms_list_opt = option(
    '--zooms',
    'zooms_list',
    type=str,
    default='0,1,2,3,4',
    callback=zoom_handler,
    help='Значение (значения) увеличения (zoom) для генерации тайлов.'
)

bands_list_opt = option(
    '--bands',
    'bands_list',
    default='1',
    callback=band_handler,
    type=str,
    help="Каналы входного изображения для генерации тайлов.")

isolines_generate_opt = option(
    '-il',
    '--isolines',
    'generate_isolines',
    default=None,
    help='Генерировать изолинии'
)