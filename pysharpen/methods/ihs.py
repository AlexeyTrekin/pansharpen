import cv2
import numpy as np
from .pansharp import Pansharp
from pysharpen.preprocessing.type_conversion import saturate_cast, wider_type


class IHS(Pansharp):

    def __init__(self):
        Pansharp.__init__(self)

    def sharpen(self, pan, ms):
        """
        :param pan:
        :param ms:
        :return:
        """
        # The methods expect already resized images

        assert pan.shape == ms.shape[1:], 'Shapes of multispectral and panchrom are different'
        assert ms.shape[0] == 3, 'IHS pansharpening is restricted to 3 channels, use GIHS for others'
        ms = ms.transpose(1, 2, 0)

        pan = pan.astype(np.float32)
        ms = ms.astype(np.float32)

        hsv = cv2.cvtColor(ms, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)

        return RGBpan.transpose(2, 0, 1)