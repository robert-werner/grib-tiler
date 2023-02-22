import os

import numpy as np
import rasterio

from grib_tiler.utils import get_driver_extension


class Task:

    def __init__(self, input_filename, output_filename=None, output_directory=None):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.output_directory = output_directory


class WarpTask(Task):

    def __init__(self, input_filename, output_directory, output_crs=None,
                 multithreading=True,
                 write_flush=True,
                 cutline_filename=None,
                 cutline_layer_name=None,
                 target_extent=None,
                 target_extent_crs=None,
                 output_format=None,
                 source_nodata=None,
                 destination_nodata=None,
                 crop_to_cutline=False):
        super().__init__(input_filename=input_filename,
                         output_directory=output_directory)
        self.output_crs = output_crs
        self.multithreading = multithreading
        self.write_flush = write_flush
        self.cutline_filename = cutline_filename
        self.cutline_layer_name = cutline_layer_name
        self.target_extent = target_extent
        self.target_extent_crs = target_extent_crs
        self.output_format = output_format
        self.source_nodata = source_nodata
        self.destination_nodata = destination_nodata
        self.crop_to_cutline = crop_to_cutline

        input_filename_splittext = os.path.splitext(os.path.basename(self.input_filename))
        input_filename_base = input_filename_splittext[0]
        self.output_format_extension = get_driver_extension(output_format)
        self.output_filename = os.path.join(self.output_directory,
                                            f'{input_filename_base}_warped{self.output_format_extension}')


class RenderTileTask(Task):

    def __init__(self, input_filename, output_directory, z, x, y, tms,
                 nodata=None,
                 tilesize=256,
                 dtype='uint8',
                 image_format='PNG',
                 subdirectory_name=None,
                 nodata_mask_array=None):
        super().__init__(input_filename=input_filename, output_directory=output_directory)
        self.z = z
        self.x = x
        self.y = y
        self.tms = tms
        self.nodata = nodata
        self.tilesize = tilesize
        self.dtype = dtype
        self.image_format = image_format
        self.subdirectory_name = subdirectory_name
        if self.subdirectory_name:
            self.output_directory = os.path.join(self.output_directory, self.subdirectory_name)
        self.output_filename = os.path.join(self.output_directory, str(self.z), str(self.x),
                                            f'{self.y}{self.get_raster_extension(self.image_format)}')
        self._nodata_mask = nodata_mask_array


    @staticmethod
    def get_raster_extension(tile_img_format):
        img_to_ext = {
            'PNG': '.png',
            'JPEG': '.jpg',
            'GTIFF': '.tiff',
            'GTiff': '.tiff',
            'VRT': '.vrt'
        }
        return img_to_ext[tile_img_format]

    @property
    def nodata_mask(self):
        if isinstance(self._nodata_mask, np.ndarray):
            return self._nodata_mask

    @property
    def bands(self):
        with rasterio.open(self.input_filename) as input_rio_ds:
            indexes = list(range(1, len(input_rio_ds.indexes) + 1))
            if len(indexes) == 1:
                indexes = indexes[0]
        return indexes


class InRangeTask(Task):

    def __init__(self, input_filename, bands, threads):
        super().__init__(input_filename)
        self.bands = bands
        self.threads = threads


class TranslateTask(Task):

    def __init__(self, input_filename, output_filename, output_directory=None,
                 band=None,
                 output_format='VRT',
                 scale=None,
                 output_dtype='Byte'):
        super().__init__(input_filename=input_filename,
                         output_filename=output_filename,
                         output_directory=output_directory)
        self.bands = band
        self.output_format = output_format
        self.scale = scale
        self.output_dtype = output_dtype


class VirtualTask(Task):

    ...
