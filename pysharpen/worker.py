import os
import rasterio
import numpy as np
import aeronet.dataset as ds

from typing import List
from tqdm import tqdm
from aeronet.converters.split import split
from pysharpen.methods import ImgProc
from rasterio.windows import Window


class Worker:

    def __init__(self, methods: List[ImgProc],
                 window_size=(2048, 2048),
                 resampling='bilinear', out_dtype=None):

        self.out_dtype = out_dtype
        self.window_size = window_size
        self.resampling = resampling
        # methods have to be initialized
        self.methods = methods

        # bound is specified by the methods
        self.setup_bound = np.max([m.setup_bound for m in self.methods])
        self.processing_bound = np.max([m.processing_bound for m in self.methods])
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

    @staticmethod
    def _merge_channels(bc, out_file, window_size=(2048, 2048), labels=None, verbose=False, **kwargs):
        if labels is None:
            labels = [b.name for b in bc]
        sampler = ds.io.SequentialSampler(bc, labels, window_size, bound=0)
        with rasterio.open(out_file, 'w', **kwargs) as dst:
            w = dst.width
            h = dst.height
            for sample, block in tqdm(sampler, disable=not verbose):

                window = Window(block['x'], block['y'],
                                min(block['width'], w - block['x']),
                                min(block['height'], h - block['y']))

                dst.write(sample.sample(0,0,
                                        min(block['height'], h - block['y']),
                                        min(block['width'], w - block['x'])).numpy(), window=window)

    def setup_methods(self, bc, channels, verbose=False):

        sampler = ds.io.SequentialSampler(bc, channels, sample_size=self.window_size, bound=self.setup_bound)
        # Apply different bound for different calculations to avoid errors
        # At the moment, the bound is maximum of all methods' working bounds
        # Also maybe different bounds for setup and processing

        for sample, block in tqdm(sampler, disable=not verbose):
            img = sample.numpy()
            ms = img[:-1]
            pan = img[-1]
            # TODO: apply nodata masks to exclude values
            # if bc.nodata is not None:
            #    pan = MaskedArray(pan, (pan == bc[-1].nodata), fill_value=bc[-1].nodata)
            #    ms = MaskedArray(ms, (ms == bc.nodata), fill_value=bc.nodata)

            for m in self.methods:
                if m.setup_required:
                    # The excessive boundary can affect the statistics of the image, so we cut the input for each method
                    # according to their minimum necessary bound
                    cut = self.setup_bound - m.processing_bound
                    m.setup_from_patch(pan[cut: pan.shape[0]-cut, cut: pan.shape[1]-cut],
                                       ms[:, cut:ms.shape[1]-cut, cut:ms.shape[2]-cut])

        for m in self.methods:
            if m.setup_required:
                m.finalize_setup()

    def process(self, bc, channels, output_labels, output_dir, verbose=False):
        if self.out_dtype is not None:
            dtype = self.out_dtype
        else:
            dtype = bc[-1].dtype
        return ds.Predictor(channels, output_labels, self.processing_fn,
                            sample_size=self.window_size, bound=self.processing_bound,
                            verbose=verbose, dtype=dtype)\
            .process(bc, output_dir)

    def process_separate(self, input_dir,  mul_channels, pan_channel='PAN', output_dir=None, output_labels=None,
                         extensions=('tif', 'tiff', 'TIF', 'TIFF'), verbose=False):
        """
        Processing of the bands represented as a set separate files, one for each band
        Args:
            input_dir:
            mul_channels:
            pan_channel:
            output_dir:
            output_labels:
            extensions:
            verbose:

        Returns:

        """

        # We call pansharpened channel with 'P' prefix, like pansharpen RED is PRED
        # if they are not specified or with errors
        if output_labels is None or len(output_labels) != len(mul_channels):
            if verbose:
                print('Generating names for pansharpened channels')
            output_labels = ['P' + channel for channel in mul_channels]
        # A single band collection is necessary for the Predictor
        # reproject_to provides geographic matching of pan and mul images
        pan_band = ds.Band(ds.parse_directory(input_dir, [pan_channel], extensions)[0])
        mul_bands = ds.BandCollection(ds.parse_directory(input_dir, mul_channels, extensions))

        tmp_names = [os.path.join(input_dir, 'resampled_' + channel + '.tif') for channel in mul_channels]
        if verbose:
            print('Reprojection of the multispectral bands to the size of the panchromatic band')
        tmp_bands = [b.reproject_to(pan_band,
                                    fp=tmp_name,  # We manage the temp files by hand
                                    interpolation=self.resampling) for b, tmp_name in tqdm(zip(mul_bands, tmp_names),
                                                                                           disable=not verbose)]

        all_bands = ds.BandCollection(tmp_bands + [pan_band])
        if verbose:
            print('Methods setup in progress')
        self.setup_methods(all_bands, mul_channels + [pan_channel], verbose=verbose)
        if verbose:
            print('Processing in progress')
        self.process(all_bands, mul_channels + [pan_channel], output_labels, output_dir, verbose)
        # remove temp files
        for band in tmp_bands:
            band._tmp_file = True

    def process_single(self, pan_file, ms_file, out_file, channels=None, clean=True, verbose=False):
        """
        Processing of files where the MS image is in one file and Pan in another.
        It splits the MS file to a set of bands and then as in process_separate.
        The result is then stacked back to a single MS file
        Args:
            pan_file:
            ms_file:
            out_file:
            channels: names for the channels of ms file to split. If None, the names are generated
            clean: if True, all separate band files (both source and pansharpened) are deleted
            verbose: if True, print the
        Returns:

        """

        folder = os.path.dirname(os.path.abspath(ms_file))
        pan_band = ds.Band(pan_file)

        with rasterio.open(ms_file) as src:
            ms_profile = src.profile
        with rasterio.open(pan_file) as src:
            pan_profile = src.profile

        if channels is None:
            channels = ['B' + str(i+1).zfill(2) for i in range(ms_profile['count'])]
        mul_bands = split(ms_file, folder, channels)

        output_labels = ['P' + channel for channel in channels]
        tmp_names = [os.path.join(folder, 'resampled_' + channel + '.tif') for channel in channels]
        if verbose:
            print('Reprojection of the multispectral bands to the size of the panchromatic band')
        tmp_bands = [b.reproject_to(pan_band,
                                    fp=tmp_name, # We manage the temp files by hand
                                     interpolation=self.resampling) for b, tmp_name in tqdm(zip(mul_bands, tmp_names),
                                                                                            disable=not verbose)]

        all_bands = ds.BandCollection(tmp_bands + [pan_band])
        if verbose:
            print('Methods setup in progress')
        self.setup_methods(all_bands, channels + [pan_band.name], verbose=verbose)
        if verbose:
            print('Processing in progress')
        out_bc = self.process(all_bands, channels + [pan_band.name], output_labels, folder, verbose=verbose)

        pan_profile.update(count=ms_profile['count'], dtype=out_bc[0].dtype, BIGTIFF='IF_SAFER')

        # We want to merge the channels back into one file as it was
        # It requires reading full files, so it's not memory-efficient
        # However, just the read-write operations will not consume more than the image size
        if verbose:
            print('Processing is completed, stacking the output channels to a single file')
        self._merge_channels(out_bc, out_file, self.window_size, labels=output_labels, verbose=verbose, **pan_profile)

        if clean:
            if verbose:
                print('Cleaning the intermediate files')
            # Mark the files for removal
            for band, pband, tmp_band in zip(mul_bands, out_bc, tmp_bands):
                band._tmp_file = True
                pband._tmp_file = True
                tmp_band._tmp_file = True
