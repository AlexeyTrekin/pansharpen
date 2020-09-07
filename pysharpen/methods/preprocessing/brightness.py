import numpy as np
from math import sqrt
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import linear_brightness_scale

ALLOWED_METHODS = ['minmax', 'meanstd', 'histogram']


class LinearBrightnessScale(ImgProc):
    """
     IHS pansharpening is for RGB image only. It transforms RGB into IHS color space,
     then replaces Intensity with PAN channel and transforms backwards. Based on OpenCV color transforms
    """

    def __init__(self, method='minmax', dtype=None,
                 per_channel=False, pan_separate=False,
                 process_pan=True, process_ms=True,
                 std_width=3.0, hist_cut=0.05):
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
        self.setup_required = True

        if method not in ALLOWED_METHODS:
            raise ValueError(f'Method {method} is not available. Only {ALLOWED_METHODS} are allowed')
        self._method = method
        self._dtype = dtype
        self._process_pan = process_pan
        self._process_ms = process_ms

        self._per_channel = per_channel
        # if ms is scaled per-channel, we cannot scale pan together with all the channels
        self._pan_separate = pan_separate or per_channel
        # Valid only for method = 'meanstd'
        if type(std_width) not in (int, float) or std_width <=0:
            self._std_width=std_width
            raise ValueError('Only positive std_width is allowed, otherwise the values of min and max are undefined')
        self._std_width = std_width
        # Valid only for method = 'histogram'
        if type(hist_cut) not in (int, float) or hist_cut < 0 or hist_cut >= 1:
            self._hist_cut = hist_cut
            raise ValueError('Hist_cut value must be in range [0:1)')
        self._hist_cut = hist_cut

        # ============ total image statistics used for calculations ===========#
        self._pan_min_value = 0
        self._pan_max_value = 1
        self._ms_min_value = [] if self._per_channel else 0
        self._ms_min_value = [] if self._per_channel else 1

        self._ms_mean_value = [] if self._per_channel else 0
        self._ms_std_value = [] if self._per_channel else 0
        self._pan_mean_value = 0
        self._pan_std_value = 0
        # ================ temp storage for image statistics ================ #
        # separate pan mins and maxs for pan and ms
        self._pan_mins = []
        self._pan_maxs = []

        # ms mins and maxs - numeric if not self.per_channel and list else
        self._ms_mins = []
        self._ms_maxs = []

        # mean, std and mean error
        self._pan_means = []
        self._pan_stds = []
        self._pan_mes = []
        self._pan_nums = []

        self._ms_means = []
        self._ms_stds = []
        self._ms_nums = []
        self._ms_mes = []

        # histogram
        self._pan_hist = []
        self._ms_hist = []

    def __repr__(self):
        repr = f'LinearBrightnessScale(method: {self._method}, ' \
               f'pan_separate: {self._pan_separate}, per_channel: {self._per_channel}, ' \
               f'process_pan: {self._process_pan}, process_ms: {self._process_ms})'
        if self._method == 'meanstd':
            repr = repr + f', width: {self._std_width}'
        elif self._method == 'historgam':
            repr = repr + f', {self._hist_cut}'

        return repr

    def setup_from_patch(self, pan, ms):

        if self._method == 'minmax':
            self._minmax_from_patch(pan, ms)
        elif self._method == 'meanstd':
            self._meanstd_from_patch(pan, ms)
        elif self._method == 'historgam':
            self._histogram_from_patch(pan, ms)
        else:
            raise ValueError(f'Unknown method {self._method}')

    def finalize_setup(self):
        """
            Claculates total min and max for the pan and ms values from accumulated windowed min and max
        """
        if self._method == 'minmax':
            self._finalize_minmax()
        elif self._method == 'meanstd':
            self._finalize_meanstd()
        elif self._method == 'historgam':
            self._finalize_histogram()
        else:
            raise ValueError(f'Unknown method {self._method}')

    def process(self, pan, ms):
        """
        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker

        Returns:
            pansharpened image, the same size as pan, but with number of channels as in ms, with IHS method
        """

        dtype = self._dtype if self._dtype is not None else pan.dtype
        pan_res = None
        ms_res = None

        if self._process_pan:
            pan_res = linear_brightness_scale(pan,
                                          self._pan_min_value, self._pan_max_value,
                                          dtype=dtype)
        if self._process_ms:
            if self._per_channel:
                ms_res = np.zeros(shape=ms.shape, dtype=dtype)
                for channel in range(ms.shape[0]):
                    ms_res[channel] = linear_brightness_scale(ms[channel],
                                                          self._ms_min_value[channel], self._ms_max_value[channel],
                                                          dtype=dtype)
            else:
                ms_res = linear_brightness_scale(ms,
                                             self._ms_min_value, self._ms_max_value,
                                             dtype=dtype)
        return pan_res, ms_res

    # =========================== Private functions: methods of setup ======================= #

    # find minimum and maximum value of the whole image
    def _minmax_from_patch(self, pan, ms):

        if not self._pan_separate:
            local_min = min(pan.min(), ms.min())
            local_max = max(pan.max(), ms.max())
            self._pan_mins.append(local_min)
            self._pan_maxs.append(local_max)
            self._ms_mins.append(local_min)
            self._ms_maxs.append(local_max)
        else:
            self._pan_mins.append(pan.min())
            self._pan_maxs.append(pan.max())

            if self._per_channel:
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
        if self._per_channel:
            self._ms_min_value = np.min(self._ms_mins, axis=0)
            self._ms_max_value = np.max(self._ms_maxs, axis=0)
        else:
            self._ms_min_value = np.min(self._ms_mins)
            self._ms_max_value = np.max(self._ms_maxs)

        # Clear the statistics accumulators
        # separate pan mins and maxs for pan and ms
        self._pan_mins = []
        self._pan_maxs = []

        # ms mins and maxs - numeric if not self.per_channel and list else
        self._ms_mins = []
        self._ms_maxs = []


    # ================== Histogram method ========================= #
    # Find minimum and maximum from histogram
    def _histogram_from_patch(self, pan, ms):
        return

    def _finalize_histogram(self):
        return

    # ======================= Meanstd method ======================== #
    # Find maximum and minimum as mean +- N*std
    @staticmethod
    def _meanstdme(img):
        mean = img.mean()
        std = img.std()
        num = img.size
        me = (img - mean).sum() / num

        return mean, std, num, me

    @staticmethod
    def _totalmeanstd(means, stds, nums, mes):
        mean = np.sum([m*n for m,n in zip(means, nums)])/np.sum(nums)
        # Calculation of the stddev from the window statistics
        std = sqrt(np.sum(
              [(nums[i]*(stds[i]**2 + (mean - means[i])**2 + 2*(means[i] - mean)*mes[i]))
               for i in range(len(means))])/np.sum(nums))
        return mean, std

    def _meanstd_from_patch(self, pan, ms):

        # We also need min and max values if we use meanstd
        self._minmax_from_patch(pan, ms)

        if not self._pan_separate:
            mean, std, num, me = self._meanstdme(np.concatenate([ms, np.expand_dims(pan, 0)], axis=0))
            self._pan_means.append(mean)
            self._pan_stds.append(std)
            self._pan_nums.append(num)
            self._pan_mes.append(me)

            self._ms_means.append(mean)
            self._ms_stds.append(std)
            self._ms_nums.append(num)
            self._ms_mes.append(me)
        else:
            mean, std, num, me = self._meanstdme(pan)
            self._pan_means.append(mean)
            self._pan_stds.append(std)
            self._pan_mes.append(me)
            self._pan_nums.append(num)

            if self._per_channel:
                means = []
                stds = []
                nums = []
                mes = []
                for ch in ms:
                    mean, std, num, me = self._meanstdme(ch)
                    means.append(mean)
                    stds.append(std)
                    nums.append(num)
                    mes.append(me)
                self._ms_means.append(np.array(means))
                self._ms_stds.append(np.array(stds))
                self._ms_mes.append(np.array(mes))
                self._ms_nums.append(np.array(nums))
            else:
                mean, std, num, me = self._meanstdme(ms)
                self._ms_means.append(mean)
                self._ms_stds.append(std)
                self._ms_mes.append(me)
                self._ms_nums.append(num)

    def _finalize_meanstd(self):
        # We need all the statistics to be calculated for every patch in the same order, so we check the length
        patches_num = set(len(stat) for stat in [self._ms_means, self._ms_stds, self._ms_nums, self._ms_mes,
                      self._pan_means, self._pan_stds, self._pan_nums, self._pan_mes])
        if len(patches_num) != 1:
            raise ValueError('Some of the patches staticstics were not calculated correcty, '
                             'and the length of the patch statistics arrays are different')
        elif 0 in patches_num:
            raise ValueError('No data received during the setup')
        # We find the actual min and max of images
        self._finalize_minmax()
        if self._per_channel:
            ms_mean = np.zeros_like(self._ms_means[0])
            ms_std = np.zeros_like(self._ms_stds[0])
            for ch in range(ms_mean.shape[0]):
                ms_mean[ch], ms_std[ch] = self._totalmeanstd(np.array(self._ms_means)[:,ch],
                                                             np.array(self._ms_stds)[:,ch],
                                                             np.array(self._ms_nums)[:,ch],
                                                             np.array(self._ms_mes)[:,ch])

        else:
            ms_mean, ms_std = self._totalmeanstd(self._ms_means, self._ms_stds, self._ms_nums, self._ms_mes)
        pan_mean, pan_std = self._totalmeanstd(self._pan_means, self._pan_stds, self._pan_nums, self._pan_mes)

        self._pan_mean_value = pan_mean
        self._pan_std_value = pan_std

        self._ms_mean_value = ms_mean
        self._ms_std_value = ms_std
        # Then we make the min higher and the max lower if it is necessary
        # Maybe not use minmax, and just clip to the value range?

        axis = 0 if self._per_channel else None
        self._pan_min_value = max(pan_mean - self._std_width * pan_std, self._pan_min_value)
        self._ms_min_value = np.max([ms_mean - self._std_width * ms_std, self._ms_min_value], axis=axis)
        self._pan_max_value = min(pan_mean + self._std_width * pan_std, self._pan_max_value)
        self._ms_max_value = np.min([ms_mean + self._std_width * ms_std, self._ms_max_value], axis=axis)

        # Clear the statistics accumulators
        # separate pan mins and maxs for pan and ms
        self._pan_mins = []
        self._pan_maxs = []

        # ms mins and maxs - numeric if not self.per_channel and list else
        self._ms_mins = []
        self._ms_maxs = []

        # mean, std and mean error
        self._pan_means = []
        self._pan_stds = []
        self._pan_mes = []
        self._pan_nums = []

        self._ms_means = []
        self._ms_stds = []
        self._ms_nums = []
        self._ms_mes = []


