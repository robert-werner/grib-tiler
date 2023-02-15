import rasterio


def seek_by_meta_value(input_fn, **meta_term):
    results = {}
    with rasterio.open(input_fn) as input_rio:
        for bidx in input_rio.indexes:
            tags = input_rio.tags(bidx)
            for k, v in meta_term.items():
                if k in tags:
                    for _v in v:
                        if tags[k] == _v:
                            if 'meta' in meta_term:
                                if meta_term['meta']:
                                    if _v in results:
                                        results[_v].append(bidx)
                                    else:
                                        results[_v] = []
                                        results[_v].append(bidx)
                            else:
                                if _v in results:
                                    results[_v].append({bidx: tags})
                                else:
                                    results[_v] = []
                                    results[_v].append({bidx: tags})
    return results


def get_driver_extension(driver):
    driver_to_extension = {
        'GTiff': '.tiff',
        'GRIB': '.grib2',
        'VRT': '.vrt'
    }
    return driver_to_extension[driver]
