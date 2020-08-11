import os
import rasterio
import numpy as np
import aeronet.dataset as ds
from aeronet.converters.split import split

from rasterio.enums import Resampling
from rasterio.windows import Window
from methods import Panshrp


class Worker:
    def __init__(self, method:Panshrp,
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

        input_channels = mul_channels + [pan_label]
        # We call pansharpened channel with 'P' prefix, like pansharpen RED is PRED
        output_labels = ['P' + channel for channel in input_channels]
        pan_band = ds.Band(pan_channel)

        # A single band collection is necessary for the Predictor
        # reproject_to provides geographic matching of pan and mul images
        pan_band = ds.Band(ds.parse_directory(input_dir, [pan_channel], extensions)[0])
        mul_bands = ds.BandCollection(ds.parse_directory(input_dir, mul_channels, extensions))

        # Or should we do it after reprojection?
        if not self.method.ready:
            self.method.setup(pan_band, mul_bands)

        all_bands = mul_bands.reproject_to(pan_band, interpolation=self.resampling).append(pan_band)

        pred = ds.Predictor(input_channels, output_labels, self.method.get_sharpen_fn(),
                            sample_size=self.window_size, bound=self.bound, verbose=False)

        pred.process(all_bands, output_dir)

    def process_single(self, pan_file, ms_file, out_file, channels=None, clean=True):
        folder = os.path.dirname(ms_file)
        pan_band = ds.Band(pan_file)

        with rasterio.open(ms_file) as src:
            ms_profile = src.profile

        if channels is None:
            channels = ['B' + str(i).zfill(2) for i in range(ms_profile.count)]
        mul_bands = split(ms_file, folder, channels)

        if not self.method.ready:
            self.method.setup(pan_band, mul_bands)

        all_bands = mul_bands.reproject_to(pan_band, interpolation=self.resampling).append(pan_band)

        pred = ds.Predictor(input_channels, output_labels, self.method.get_sharpen_fn(),
                            sample_size=self.window_size, bound=self.bound, verbose=False)

        out_bc = pred.process(all_bands, output_dir)

        profile = pan_band.profile
        profile.update(count = mul_bands.count)
        # We want to merge the channels back into one file as it was
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(out_bc.numpy())

        if clean:
            for band, pband in zip(mul_bands, out_bc):
                os.remove(band._band.name)
                os.remove(pband._band.name)

    def process(self, pan_file, ms_file, out_file):
        with rasterio.open(pan_file) as pan:
            profile = pan.profile
            pan_w = pan.width
            pan_h = pan.height
            dtype = pan.dtypes[0]

            with rasterio.open(ms_file) as mul:
                mul_w = mul.width
                mul_h = mul.height
                factor = (pan_w / mul_w, pan_h / mul_h)

                # method may need a setup based on the whole image
                self.method.setup(pan, mul)
                windows = self._generate_windows((pan_w, pan_h), self.window_size, factor, self.padding)

                profile.update(count=mul.count)
                with rasterio.open(out_file, 'w', **profile) as dst:
                    dst.colorinterp = mul.colorinterp
                    for mul_window, pan_window in windows:
                        pan_tile = pan.read(1, window=pan_window)
                        # Read with resampling
                        mul_tile = np.zeros((mul.count, pan_tile.shape[0], pan_tile.shape[1]), dtype=dtype)
                        mul.read(out=mul_tile, resampling=self.resampling, window=mul_window)

                        result = self.method.sharpen(pan_tile, mul_tile).astype(dtype)
                        dst.write(result, window=pan_window)

