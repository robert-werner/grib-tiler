from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional

RenderTileTask = namedtuple('RenderTileTask',
                            ['input_fn', 'z', 'x', 'y', 'tms', 'nodata', 'in_range', 'tilesize', 'img_format',
                             'output_dir', 'band_name'])
SaveTileTask = namedtuple('SaveTask', ['rendered_tile', 'output_dir'])
RenderedTile = namedtuple('RenderedTile', ['tile', 'bands', 'z', 'x', 'y'])
TranslateTask = namedtuple('TranslateTask',
                           ['input_fn', 'output_dir', 'band', 'output_format', 'output_format_extension'])
WarpTask = namedtuple('WarpTask',
                      ['input_fn', 'output_dir', 'output_crs', 'multithreaded', 'cutline_fn', 'cutline_layername',
                       'output_format', 'src_nodata', 'dst_nodata', 'write_flush'])
InRangeTask = namedtuple('InRangeTask', ['input_fn', 'band'])
VRTask = namedtuple('VRTask', ['src_ds_s', 'band'])
MetaInfoTask = namedtuple('MetaInfoTask', ['in_range', 'output_dir'])