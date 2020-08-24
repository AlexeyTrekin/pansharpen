import os
import rasterio
import numpy as np
import aeronet.dataset as ds
from numpy.ma import MaskedArray
from typing import List
from aeronet.converters.split import split
from pysharpen.methods import ImgProc


class Worker:
    def __init__(self, methods: List[ImgProc],
                 window_size=(2048, 2048),
                 resampling='bilinear'):

        self.window_size = window_size
        self.resampling = resampling
        # methods have to be initialized
        self.methods = methods

        # bound is specified by the methods
        self.bound = np.max([m.bound for m in self.methods])

        # all the functions of the methods are bound into a single processing function to calculate in a single pass
        def processing_fn(collection):
            if collection.count > 1:
                img = collection.numpy()
                pan = img[-1]
                ms = img[:-1]
            else:
                raise ValueError('The samples inside the collection must be a list '
                                 'with 2 or more channels where PAN channel is the last')
            for m in self.methods:
                pan, ms = m.process(pan, ms)
            return ms
        self.processing_fn = processing_fn

    def setup_methods(self, bc, channels):

        sampler = ds.io.SequentialSampler(bc, channels, sample_size=self.window_size, bound=self.bound)
        # Apply different bound for different calculations to avoid errors
        # At the moment, the bound is maximum of all methods' working bounds
        # Also maybe different bounds for setup and processing

        for sample, block in sampler:
            img = sample.numpy()
            ms = img[:-1]
            pan = img[-1]
            # TODO: apply nodata masks to exclude values
            # if bc.nodata is not None:
            #    pan = MaskedArray(pan, (pan == bc[-1].nodata), fill_value=bc[-1].nodata)
            #    ms = MaskedArray(ms, (ms == bc.nodata), fill_value=bc.nodata)

            for m in self.methods:
                if m.setup_required:
                    m.setup_from_patch(pan, ms)

        for m in self.methods:
            if m.setup_required:
                m.finalize_setup()

    def process(self, bc, channels, output_labels, output_dir):
        return ds.Predictor(channels, output_labels, self.processing_fn,
                            sample_size=self.window_size, bound=self.bound,
                            verbose=True, dtype=bc[-1].dtype)\
            .process(bc, output_dir)

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

        all_bands = ds.BandCollection([b.reproject_to(pan_band, interpolation=self.resampling) for b in mul_bands]
                                      + [pan_band])
        self.setup_methods(all_bands, mul_channels + [pan_channel])
        self.process(all_bands, mul_channels + [pan_channel], output_labels, output_dir)

    def process_single(self, pan_file, ms_file, out_file, channels=None, clean=True):
        """
        TODO: deprecate?
        :param pan_file:
        :param ms_file:
        :param out_file:
        :param channels:
        :param clean:
        :return:
        """
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

        all_bands = ds.BandCollection([b.reproject_to(pan_band, interpolation=self.resampling) for b in mul_bands]
                                      + [pan_band])

        self.setup_methods(all_bands, channels + [pan_band.name])
        out_bc = self.process(all_bands, channels + [pan_band.name], output_labels, folder)

        pan_profile.update(count=ms_profile['count'])
        
        # We want to merge the channels back into one file as it was
        # It requires reading full files, so it's not memory-efficient
        # However, just the read-write operations will not consume more than the image size
        with rasterio.open(out_file, 'w', **pan_profile) as dst:
            for band_num, band in enumerate(out_bc):
                print(band.numpy().shape)
                dst.write(band.numpy(), band_num + 1)

        if clean:
            for band, pband in zip(mul_bands, out_bc):
                os.remove(band._band.name)
                os.remove(pband._band.name)
