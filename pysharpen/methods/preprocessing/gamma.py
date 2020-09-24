import numpy as np
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import gamma_correction


class GammaCorrection(ImgProc):
    """
     IHS pansharpening is for RGB image only. It transforms RGB into IHS color space,
     then replaces Intensity with PAN channel and transforms backwards. Based on OpenCV color transforms
    """

    def __init__(self, gamma=2.2, dtype=None,
                 per_channel=False, pan_separate=False,
                 process_pan=True, process_ms=True,
                 **kwargs):
        """

        Args:
            method: minmax, or
            dtype:
            per_channel:
            pan_separate:
            process_pan: can be disabled if not necessary to speed up
            process_ms: can be disabled if not necessary to speed up
        """
        super().__init__()
        self.gamma = gamma
        self.dtype = dtype
        self.process_pan = process_pan
        self.process_ms = process_ms

        self.per_channel = per_channel
        # if ms is scaled per-channel, we cannot scale pan together with all the channels
        self.pan_separate = pan_separate or per_channel

        self._mins = []
        self._maxs = []
        self._min_value = 0
        self._max_value = 255

        self.setup_required = True

    def setup_from_patch(self, pan, ms, nodata = None):
        if pan.dtype == ms.dtype == np.uint8:
            return

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
        if self.process_pan:
            pan = gamma_correction(pan, self.gamma, self._min_value, self._max_value, self.dtype)
        if self.process_ms:
            ms = gamma_correction(ms, self.gamma, self._min_value, self._max_value, self.dtype)
        return pan, ms

