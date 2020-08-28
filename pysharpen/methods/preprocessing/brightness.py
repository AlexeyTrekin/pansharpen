import numpy as np
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import linear_brightness_scale


class LinearBrightnessScale(ImgProc):
    """
     IHS pansharpening is for RGB image only. It transforms RGB into IHS color space,
     then replaces Intensity with PAN channel and transforms backwards. Based on OpenCV color transforms
    """

    def __init__(self, method='minmax', dtype=None,
                 per_channel=False, pan_separate=False,
                 process_pan=True, process_ms=True,
                 **kwargs):
        """

        Args:
            method: minmax
            dtype:
            per_channel:
            pan_separate:
            process_pan: can be disabled if not necessary to speed up
            process_ms: can be disabled if not necessary to speed up
        """
        super().__init__()
        self.method = method
        self.dtype = dtype
        self.process_pan = process_pan
        self.process_ms = process_ms

        self.per_channel = per_channel
        # if ms is scaled per-channel, we cannot scale pan together with all the channels
        self.pan_separate = pan_separate or per_channel

        # separate pan mins and maxs for pan and ms
        self._pan_mins = []
        self._pan_maxs = []
        self._pan_min_value = 0
        self._pan_max_value = 255
        # ms mins and maxs - numeric if not self.per_channel and list else
        self._ms_mins = []
        self._ms_maxs = []
        if self.per_channel:
            self._ms_min_vector = []
            self._ms_max_vector = []
        else:
            self._ms_min_value = 0
            self._ms_max_value = 255

        self.setup_required = True

    def setup_from_patch(self, pan, ms):
        # we do not setup it for unit32 as the range is 0:255 and well defined
        if pan.dtype == ms.dtype == np.uint8:
            return
        if self.method == 'minmax':
            self._minmax_from_patch(pan, ms)

    def finalize_setup(self):
        """
            Claculates total min and max for the pan and ms values from accumulated windowed min and max
        """
        if self.method == 'minmax':
            self._finalize_minmax()

    def process(self, pan, ms):
        """
        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker

        Returns:
            pansharpened image, the same size as pan, but with number of channels as in ms, with IHS method
        """

        dtype = self.dtype if self.dtype is not None else pan.dtype
        if self.process_pan:
            pan = linear_brightness_scale(pan,
                                          self._pan_min_value, self._pan_max_value,
                                          dtype=dtype)
        if self.process_ms:
            if self.per_channel:
                for channel in range(ms.shape[0]):
                    ms[channel] = linear_brightness_scale(ms[channel],
                                                          self._ms_min_vector[channel], self._ms_max_vector[channel],
                                                          dtype=dtype)
            else:
                ms = linear_brightness_scale(ms,
                                             self._ms_min_value, self._ms_max_value,
                                             dtype=dtype)
        return pan, ms

    # =========================== Private functions: methods of setup ======================= #

    # find minimum and maximum value of the whole image
    def _minmax_from_patch(self, pan, ms):

        if not self.pan_separate:
            local_min = min(pan.min(), ms.min())
            local_max = max(pan.max(), ms.max())
            self._pan_mins.append(local_min)
            self._pan_mins.append(local_max)
            self._ms_mins.append(local_min)
            self._ms_maxs.append(local_max)
        else:
            self._pan_mins.append(pan.min())
            self._pan_maxs.append(pan.max())

            if self.per_channel:
                local_min = []
                local_max = []
                for channel in ms:
                    local_min.append(channel.min())
                    local_max.append(channel.max())
                self._ms_mins.append(local_min)
                self._ms_maxs.append(local_max)
            else:
                self._ms_mins.append(ms.min())
                self._ms_maxs.append(ms.max())

    def _finalize_minmax(self):
        if len(self._pan_mins) == 0 or len(self._pan_maxs) == 0 \
                or len(self._ms_mins) == 0 or len(self._ms_maxs) == 0:
            raise ValueError('Setup was unsuccessful: could not calculate min or max values of the image')
            # TODO: manage the default values?
        self._pan_min_value = np.min(self._pan_mins)
        self._pan_max_value = np.max(self._pan_maxs)
        if self.per_channel:
            self._ms_min_vector = np.min(self._ms_mins, axis=0)
            self._ms_min_vector = np.max(self._ms_maxs, axis=0)
        else:
            self._ms_min_value = np.min(self._ms_mins)
            self._ms_max_value = np.max(self._ms_maxs)

    # Find minimum and maximum from histogram
    def _histogram_from_patch(self, pan, ms):
        return

    def _finalize_histogram(self):
        return

    # Find maximum and minimum as mean +- N*std
    def _meanstdme(self, img, axis=None):
        mean = img.mean(axis)
        std = img.std(axis)
        num = img.size
        me = (img - mean).sum(axis) / num

        return mean, std, num, me

    def _meanstd_from_patch(self, pan, ms):
        if not self.pan_separate:
            mean, std, num, me = self._meanstdme(np.concatenate(ms, pan, axis=0))
            self._ms_means.append(mean)
            self._pan_means.append(mean)
            self._ms_stds.append(std)
            self._pan_stds.append(std)
            self._imsizes.append(num)
            self._ms_mes.append(me)
            self._pan_mes.append(me)
        else:
            mean, std, num, me = self._meanstdme(pan)
            self._pan_means.append(mean)
            self._pan_stds.append(std)
            self._pan_mes.append(me)
            self._imsizes.append(num)

            if self.per_channel:
                mean, std, num, me = self._meanstdme(ms)
                self._ms_means.append(mean)
                self._ms_stds.append(std)
                self._ms_mes.append(me)

        return

    def _finalize_meanstd(self):
        return