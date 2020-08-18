import os
import rasterio
import numpy as np
import aeronet.dataset as ds
from aeronet.converters.split import split

from rasterio.enums import Resampling
from rasterio.windows import Window


class Worker:
    def __init__(self, method,
                 window_size=(2048,2048), bound=0,
                 resampling=Resampling.bilinear):
        self.window_size = window_size
        self.resampling = resampling
        self.bound = bound
        self.method = method()

        
    def process_separate(self, input_dir, output_dir, pan_channel='PAN', mul_channels=None,
                         extensions=('tif', 'tiff', 'TIF', 'TIFF')):

        if mul_channels is None:
            mul_channels = ['RED', 'GRN', 'BLU']

        # We call pansharpened channel with 'P' prefix, like pansharpen RED is PRED
        output_labels = ['P' + channel for channel in mul_channels]

        # A single band collection is necessary for the Predictor
        # reproject_to provides geographic matching of pan and mul images
        pan_band = ds.Band(ds.parse_directory(input_dir, [pan_channel], extensions)[0])
        mul_bands = ds.BandCollection(ds.parse_directory(input_dir, mul_channels, extensions))

        # Or should we do it after reprojection?
        if not self.method.ready:
            self.method.setup(pan_band, mul_bands)

        all_bands = ds.BandCollection([b.reproject_to(pan_band, interpolation=self.resampling) for b in mul_bands] + [pan_band])

        pred = ds.Predictor(mul_channels + [pan_channel], output_labels, self.method.get_sharpen_fn(),
                            sample_size=self.window_size, bound=self.bound, verbose=False, dtype=all_bands[-1].dtype)

        pred.process(all_bands, output_dir)

        
    def process_single(self, pan_file, ms_file, out_file, channels=None, clean=True):
        folder = os.path.dirname(ms_file)
        pan_band = ds.Band(pan_file)

        with rasterio.open(ms_file) as src:
            ms_profile = src.profile
        with rasterio.open(pan_file) as src:
            pan_profile = src.profile

        if channels is None:
            channels = ['B' + str(i+1).zfill(2) for i in range(ms_profile['count'])]
        mul_bands = split(ms_file, folder, channels)
        output_labels = ['P' + channel for channel in channels]

        if not self.method.ready:
            self.method.setup(pan_band, mul_bands)

        all_bands = ds.BandCollection([b.reproject_to(pan_band, interpolation=self.resampling) for b in mul_bands] + [pan_band])

        pred = ds.Predictor(channels +  [pan_band.name], output_labels, self.method.get_sharpen_fn(),
                            sample_size=self.window_size, bound=self.bound, verbose=False, dtype=all_bands[-1].dtype)

        
        out_bc = pred.process(all_bands, folder)
        pan_profile.update(count = ms_profile['count'])
        
        # We want to merge the channels back into one file as it was
        with rasterio.open(out_file, 'w', **pan_profile) as dst:
            dst.write(out_bc.numpy())

        if clean:
            for band, pband in zip(mul_bands, out_bc):
                os.remove(band._band.name)
                os.remove(pband._band.name)
