from click import option, argument, Path, Choice
from grib_tiler.utils.click_handlers import cpu_count_handler, crs_handler

multiband_opt = option(
    '--multiband',
    'multiband',
    is_flag=True,
    default=False,
    help='Генерировать многоканальные тайлы (если каналов больше 3-ёх, то выходной формат тайлов всегда будет TIFF)'
)

cutline_opt = option(
    '--cutline',
    'cutline_filename',
    default=None,
    help='Путь к файлу обрезки растра'
)

cutline_layer_opt = option(
    '-cl',
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

file_out_arg = argument(
    'OUTPUT',
    required=True,
    type=Path(resolve_path=True, file_okay=False))

img_format_opt = option(
    '-f',
    '--format',
    'image_format',
    default='PNG',
    type=Choice([
        'PNG', 'JPEG', 'TIFF'
    ]),
    help='Выходной формат тайлов.')

tile_dimension_opt = option(
    '--tilesize',
    "tilesize",
    nargs=1,
    type=int,
    default=256,
    help='Разрешение тайла (квадрат стороной, заданной флагом).')

out_crs_opt = option('--out-crs',
                           'output_crs',
                           default='EPSG:3857',
                           callback=crs_handler,
                           help="Выходная система координат тайлов.")

threads_opt = option('--threads',
                           'threads',
                           callback=cpu_count_handler,
                           type=int, default=1,
                           help='Количество потоков обработки тайлов.')

nodata_opt = option(
    '--nodata',
    'output_nodata',
    default=0,
    help='Значение "нет данных".',
)

zooms_opt = option(
    '--zooms',
    type=str,
    default='0-4',
    help='Значение (значения) увеличения (zoom) для генерации тайлов.'
)

bands_opt = option(
    '-b',
    '--band',
    'band_numbers',
    default=None,
    type=str,
    help="Каналы входного изображения для генерации тайлов.")

isolines_generate_opt = option(
    '-il',
    '--isolines',
    'generate_isolines',
    default=None,
    help='Генерировать изолинии'
)