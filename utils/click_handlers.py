import multiprocessing
import re

import click
from rasterio.rio.options import from_like_context

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
                try:
                    value = int(value)
                except:
                    raise click.UsageError('Каналы должны указываться только цифрами')
                band_indexes.append(value)
        return band_indexes
    else:
        return []


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
