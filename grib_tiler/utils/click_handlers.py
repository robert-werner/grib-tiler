import multiprocessing
import os
import re

import click
import rasterio
from rasterio.rio.options import from_like_context
from rasterio._path import _parse_path, _UnparsedPath


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
        raise click.BadParameter('Входное количество потоков не может превышать количество существующих потоков')
    return value


def crs_handler(ctx, param, value):
    return value


def bounds_handler(ctx, param, value):
    """Handle different forms of bounds."""
    retval = from_like_context(ctx, param, value)
    if retval is None and value is not None:
        try:
            value = value.strip(', []')
            retval = tuple(float(x) for x in re.split(r'[,\s]+', value))
            assert len(retval) == 4
            return retval
        except Exception:
            raise click.BadParameter(
                "{0!r} is not a valid bounding box representation".format(
                    value))
    else:  # pragma: no cover
        return retval
