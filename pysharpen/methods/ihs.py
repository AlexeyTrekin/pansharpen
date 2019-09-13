import numpy as np
import cv2
from .pansharp import Pansharp
from preprocessing.type_conversion import saturate_cast, wider_type


class IHS(Pansharp):

    def __init__(self, interp=cv2.INTER_LINEAR):
        Pansharp.__init__(self, interp)

    def sharpen(self, pan, ms):
        """
        :param pan:
        :param ms:
        :return:
        """
        ms = cv2.resize(ms, (pan.shape[1], pan.shape[0]), interpolation=self.interp)
        if ms.shape[-1] == 3:
            return self._sharpen_3(pan, ms)
        else:
            return self._sharpen_gihs(pan, ms)

    def _sharpen_3(self, pan, ms):

        # in opencv it works only with float32 and uint8, so in order to prevent overflow we will convert to uint8 anyway
        pan = pan.astype(np.float32)
        ms = ms.astype(np.float32)

        hsv = cv2.cvtColor(ms, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)
        return saturate_cast(RGBpan, ms.dtype)

    def _sharpen_gihs(self, pan, ms):
        """
        :param pan:
        :param ms:
        :return:
        """
        ms_pansharpened = ms.astype(wider_type(pan.dtype))
        mean = np.expand_dims(np.sum(ms_pansharpened, axis=2) / ms_pansharpened.shape[0], 2)
        ms_pansharpened = ms_pansharpened + pan.astype(wider_type(ms.dtype)) - mean
        # really there can be a problem if pan and mean have different scale, especially if mean is less. Maybe scaling?
        return saturate_cast(ms_pansharpened, ms.dtype)