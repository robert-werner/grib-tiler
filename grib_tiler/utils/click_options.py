import click
from rasterio.rio.options import bounds_handler

from grib_tiler.utils.click_handlers import cpu_count_handler, crs_handler, files_in_handler

multiband_opt = click.option(
    '--multiband',
    'multiband',
    is_flag=True,
    default=False,
    help='Генерировать многоканальные тайлы (если каналов больше 3-ёх, то выходной формат тайлов всегда будет TIFF)'
)

wind_opt = click.option(
    '--wind',
    'wind',
    is_flag=True,
    default=False,
    help='Генерировать тайлы ветра'
)

cutline_opt = click.option(
    '--cutline',
    'cutline_filename',
    default=None,
    help='Путь к файлу обрезки растра'
)

cutline_layer_opt = click.option(
    '-cl',
    '--cutline-layer',
    'cutline_layer',
    default=None,
    help='Слой внутри файла обрезки растра'
)

files_in_arg = click.argument(
    'input_filename',
    type=click.Path(resolve_path=True, file_okay=True, dir_okay=False, exists=True),
    required=True,
    metavar="INPUT...")

file_out_arg = click.argument(
    'OUTPUT',
    required=True,
    type=click.Path(resolve_path=True, file_okay=False))

img_format_opt = click.option(
    '-f',
    '--format',
    'image_format',
    default='PNG',
    type=click.Choice([
        'PNG', 'JPEG', 'TIFF'
    ]),
    help='Выходной формат тайлов.')

memory_opt = click.option(
    '--memory',
    'memory_opt',
    is_flag=True,
    help='''
    Обрабатывать входные данные в ОЗУ.
    Без этого флага данные будут обрабатываться частично на жёстком диске.
    '''
)

render_bounds_opt = click.option(
    '--render-bounds',
    'render_bounds',
    default=None,
    callback=bounds_handler,
    help='''
    Границы рендеринга тайла: "верхняя левая правая нижняя" или "[левая, нижняя, правая, верхняя]."
    ''')

tile_dimension_opt = click.option(
    '--tilesize',
    "tilesize",
    nargs=1,
    type=int,
    default=256,
    help='Разрешение тайла (квадрат стороной, заданной флагом).')

out_crs_opt = click.option('--out-crs',
                           'output_crs',
                           default=None,
                           callback=crs_handler,
                           help="Выходная система координат тайлов.")

threads_opt = click.option('--threads',
                           'threads',
                           callback=cpu_count_handler,
                           type=int, default=1,
                           help='Количество потоков обработки тайлов.')

nodata_opt = click.option(
    '--nodata',
    'output_nodata',
    default=0,
    help='Значение "нет данных".',
)

zooms_opt = click.option(
    '--zooms',
    type=str,
    help='Значение (значения) увеличения (zoom) для генерации тайлов.'
)

bands_opt = click.option(
    '-b',
    '--band',
    'band_numbers',
    default=None,
    type=str,
    help="Каналы входного изображения для генерации тайлов.")

metainfo_generate_opt = click.option(
    '-mi',
    '--metainfo',
    'generate_metainfo',
    default=None,
    help='Генерировать похожие на Windy метаданные к изображениям (увеличивает высоту тайла на 8 пикселей)'
)

isolines_generate_opt = click.option(
    '-il',
    '--isolines',
    'generate_isolines',
    default=None,
    help='Генерировать изолинии'
)

coeff_opt = click.option('--coeff',
                         'coeff',
                         default=1,
                         help='Применить коэффицент уменьшения к пределам матричной сетки TMS')
