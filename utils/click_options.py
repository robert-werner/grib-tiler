import click
from rasterio.rio.options import bounds_handler

from utils.click_handlers import cpu_count_handler, crs_handler, bands_handler, zooms_handler

cutline_opt = click.option(
    '-cutline',
    'cutline',
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

file_out_arg = click.argument(
    'OUTPUT',
    type=click.Path(resolve_path=True, file_okay=False))

img_format_opt = click.option(
    '-f',
    '--format',
    'img_format',
    default='PNG',
    type=click.Choice([
        'PNG', 'JPEG'
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
    '--dimension',
    "dimension",
    nargs=1,
    type=int,
    default=256,
    help='Разрешение тайла (квадрат стороной, заданной флагом).')

out_crs_opt = click.option('--out-crs',
                           'output_crs',
                           default='EPSG:4326',
                           callback=crs_handler,
                           help="Выходная система координат тайлов.")

threads_opt = click.option('--threads',
                           'threads',
                           callback=cpu_count_handler,
                           type=int, default=1,
                           help='Количество потоков обработки тайлов.')

nodata_opt = click.option(
    '--nodata',
    'nodata',
    default=0,
    help='Значение "нет данных".',
)

zooms_opt = click.option(
    '--zooms',
    type=str,
    help='Значение (значения) увеличения (zoom) для генерации тайлов.'
)

coeff_opt = click.option(
    '--coeff',
    'coeff',
    default=512,
    help='Значение коэффицента уменьшения экстента.'
)

bidx_magic_opt = click.option(
    '-b',
    '--band',
    'band_idx',
    default=None,
    type=str,
    help="Каналы входного изображения для генерации тайлов.")

uv_generate_opt = click.option(
    '-uv',
    'generate_uv',
    default=None,
    help='Генерация тайлов из u- и -v компонентов ветра'
)

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