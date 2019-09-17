import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window


class Worker:
    def __init__(self, method,
                 window_size=(2048,2048), padding=(0,0),
                 resampling=Resampling.bilinear):
        self.window_size = window_size
        self.resampling = resampling
        self.padding = padding
        self.method = method()

    @staticmethod
    def _generate_windows(imsize, window_size, factor, overlap=(0, 0)):
        full_ms = Window(0, 0, imsize[0] / factor[0], imsize[1] / factor[1])
        full_pan = Window(0, 0, imsize[0], imsize[1])
        windows = []
        col_off = 0
        row_off = 0
        m_width = window_size[0] / factor[0]
        m_height = window_size[1] / factor[1]
        while row_off < imsize[1]:
            while col_off < imsize[0]:
                pan_window = Window(col_off, row_off, window_size[0], window_size[1]).intersection(full_pan)
                mul_window = Window(col_off / factor[0], row_off / factor[1], m_width, m_height).intersection(full_ms)
                windows.append((mul_window, pan_window))
                col_off += window_size[0] - overlap[0]
            row_off += window_size[1] - overlap[1]
            col_off = 0
        return windows

    @staticmethod
    def _generate_windows_geo(pan_shape, pan_transform, mul_transform, window_size, overlap=(0, 0)):

        return

    def process(self,  pan_file, ms_file, out_file):
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

