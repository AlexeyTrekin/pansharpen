import cv2
import numpy as np
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import linear_brightness_scale


class IHSPansharpening(ImgProc):
    """
     IHS pansharpening is for RGB image only. It transforms RGB into IHS color space,
     then replaces Intensity with PAN channel and transforms backwards. Based on OpenCV color transforms
    """

    def __init__(self):
        super().__init__()

        self._mins = []
        self._maxs = []
        self._min_value = 0
        self._max_value = 255

        self.setup_required = True

    def setup_from_patch(self, pan, ms, nodata=None):
        if pan.dtype == ms.dtype == np.uint8:
            return
        #self.nodata = nodata
        #if nodata is not None:
        #    pan = MaskedArray(pan, (pan == nodata), fill_value=nodata)
        #    ms = MaskedArray(pan, (ms == nodata), fill_value=nodata)
        self._mins.append(min(pan.min(), ms.min()))
        self._maxs.append(max(pan.max(), ms.max()))

    def finalize_setup(self):
        """
            Claculates total min and max for the pan and ms values from accumulated windowed min and max
        """
        if len(self._mins) == 0 or len(self._maxs) == 0:
            self._min_value = 0
            self._max_value = 1
            return
        self._max_value = np.max(self._maxs)
        self._min_value = np.max(self._mins)
        self._mins = []
        self._maxs = []

    def process(self, pan, ms):

        """

        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker

        Returns:
            pansharpened image, the same size as pan, but with number of channels as in ms, with IHS method
        """

        if pan.shape != ms.shape[1:]:
            raise ValueError('Shapes of multispectral and panchrom are different')
        if ms.shape[0] != 3:
            raise ValueError('IHS pansharpening is restricted to 3 channels, use GIHS for others')
        if ms.dtype != pan.dtype:
            raise ValueError(f'Data types of multispectral {ms.dtype} and panchrom {pan.dtype} are different')

        dtype = pan.dtype
        ms = ms.transpose(1, 2, 0)

        if dtype != np.uint8:
            # Opencv works with uint8 and float32. All other types are converted to float32
            # and translated to 0:1 range.
            # Float32 images are treated the same way in order to manage the value range
            # We assume that the pan and rgb images are in the same brightness range.
            # Histogram equalization can be managed as an additional step, if necessary
            if self._min_value == self._max_value:
                # if one of the images is constant, we do nothing
                return pan, ms
            ms = linear_brightness_scale(ms, self._min_value, self._max_value, dtype=np.float32)
            pan = linear_brightness_scale(pan, self._min_value, self._max_value, dtype=np.float32)
        hsv = cv2.cvtColor(ms, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)
        if not (dtype == np.uint8 or dtype == np.float32):
            RGBpan = linear_brightness_scale(RGBpan, 0, 1, dtype=dtype,
                                             out_range=(self._min_value, self._max_value))
        return pan, RGBpan.transpose(2, 0, 1)
