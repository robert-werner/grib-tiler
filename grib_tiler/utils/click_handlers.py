import multiprocessing
import os

import click
import rasterio
from rasterio._path import _parse_path, _UnparsedPath
from pyproj import CRS

from grib_tiler.utils import get_rfc3339nano_time


def zooms_handler(value):
    zooms = []
    if '..' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('..'))
        if start is None:
            start = 1
        zooms.extend(list(range(start, stop + 1)))
    elif '-' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('-'))
        if start is None:
            start = 1
        zooms.extend(list(range(start, stop + 1)))
    elif ',' in value:
        zooms.extend(list(map(
            lambda x: int(x) if x else None, value.split(','))))
    else:
        if value:
            zooms.append(int(value))
    return zooms


def bands_handler(value):
    if value:
        band_indexes = []
        if '..' in value:
            start, stop = map(
                lambda x: int(x) if x else None, value.split('..'))
            if start is None:
                start = 1
            band_indexes.extend(list(range(start, stop + 1)))
        elif '-' in value:
            start, stop = map(
                lambda x: int(x) if x else None, value.split('-'))
            if start is None:
                start = 1
            band_indexes.extend(list(range(start, stop + 1)))
        elif ',' in value:
            band_indexes.extend(list(map(
                lambda x: int(x) if x else None, value.split(','))))
        else:
            if value:
                band_indexes.append(int(value))
        return band_indexes

def abspath_forward_slashes(path):
    """Return forward-slashed version of os.path.abspath"""
    return '/'.join(os.path.abspath(path).split(os.path.sep))

def files_in_handler(ctx, param, value):
    """Process and validate input file names"""
    return tuple(file_in_handler(ctx, param, item) for item in value)

def grib_handler(ctx, param, value):
    return value

def file_in_handler(ctx, param, value):
    """Normalize ordinary filesystem and VFS paths"""
    try:
        path = _parse_path(value)

        if isinstance(path, _UnparsedPath):

            if os.path.exists(path.path) and rasterio.shutil.exists(value):
                return abspath_forward_slashes(path.path)
            else:
                return path.name

        elif path.scheme and path.is_remote:
            return path.name

        elif path.archive:
            if os.path.exists(path.archive) and rasterio.shutil.exists(value):
                archive = abspath_forward_slashes(path.archive)
                return "{}://{}!{}".format(path.scheme, archive, path.path)
            else:
                raise OSError(
                    "Input archive {} does not exist".format(path.archive))

        else:
            if os.path.exists(path.path) and rasterio.shutil.exists(value):
                return abspath_forward_slashes(path.path)
            else:
                raise OSError(
                    "Input file {} does not exist".format(path.path))

    except Exception:
        raise click.BadParameter("{} is not a valid input file".format(value))

def cpu_count_handler(ctx, param, value):
    if value > multiprocessing.cpu_count():
        click.echo({"level": "fatal", "time": get_rfc3339nano_time(), "msg": "Входное количество потоков не может превышать количество существующих потоков"})
        raise click.BadParameter('Входное количество потоков не может превышать количество существующих потоков')
    return value


def crs_handler(ctx, param, value):
    crs_def = None # PureC-style костыль, на стороне PyPROJ
    crs_def = CRS.from_user_input(value) # TODO: пулл-реквест в pyproj для возвращения None-значения при отсутствии определения СК
    if not crs_def:
        click.echo({"level": "fatal", "time": get_rfc3339nano_time(), "msg": "Поддерживаются системы координат только из EPSG-базы данных (см. https://epsg.org/)"})
        raise click.BadParameter('Поддерживаются системы координат только из EPSG-базы данных (см. https://epsg.org/)')
    else:
        epsg_auth = crs_def.to_epsg()
        if not epsg_auth:
            click.echo({"level": "fatal", "time": get_rfc3339nano_time(),
                        "msg": "Поддерживаются системы координат только из EPSG-базы данных (см. https://epsg.org/)"})
            raise click.BadParameter(
                'Поддерживаются системы координат только из EPSG-базы данных (см. https://epsg.org/)')
    return ":".join(crs_def.to_authority(auth_name='EPSG'))

def zoom_handler(ctx, param, value):
    zooms = []
    if '..' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('..'))
        if start is None:
            start = 1
        zooms.extend(list(map(str,
                              list(range(start, stop + 1)))))
    elif '-' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('-'))
        if start is None:
            start = 1
        zooms.extend(list(map(str,
                              list(range(start, stop + 1)))))
    elif ',' in value:
        zooms.extend(list(map(
            lambda x: x if x else None, value.split(','))))
    else:
        try:
            int(value)
        except ValueError:
            click.echo({"level": "fatal", "time": get_rfc3339nano_time(),
                        "msg": "Недопустимый формат ввода уровней увеличения"})

            raise click.BadParameter('''
                            Допустимые форматы ввода уровней увеличения:

                            1) 0..4
                            2) 0-4
                            3) 0,1,2,3,4
                            4) 0

                            Остальные форматы недопустимы.
                            ''')
        else:
            if value < 0 or value > 24:
                raise click.BadParameter('Поддерживаются только уровни увеличения с 0 по 24 (включительно).')
    return ",".join(zooms)

def band_handler(ctx, param, value):
    band_indexes = []
    if '..' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('..'))
        if start is None or start == 0:
            start = 1
        band_indexes.extend(list(map(str,
                              list(range(start, stop + 1)))))
    elif '-' in value:
        start, stop = map(
            lambda x: int(x) if x else None, value.split('-'))
        if start is None or start == 0:
            start = 1
        band_indexes.extend(list(map(int,
                              list(range(start, stop + 1)))))
    elif ',' in value:
        band_indexes.extend(list(map(
            lambda x: int(x) if x else None, value.split(','))))
    else:
        try:
            int(value)
        except ValueError:
            click.echo({"level": "fatal", "time": get_rfc3339nano_time(),
                        "msg": "Недопустимый формат ввода каналов"})
            raise click.BadParameter('''
            Допустимые форматы ввода каналов:
            1) 1..5
            2) 1-5
            3) 1,2,3,4,5
            4) 1

            Остальные форматы недопустимы.
            ''')
        else:
            value = int(value)
            if value == 0:
                value += 1
            band_indexes = [value]
    for idx, band_index in enumerate(band_indexes):
        if band_index == 0:
            band_indexes[idx] += 1
    return ",".join(list(map(str, band_indexes)))