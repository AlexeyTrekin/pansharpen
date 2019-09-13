import numpy as np
import cv2
from .pansharp import Pansharp
from preprocessing.type_conversion import to8bit_clip


class Brovey(Pansharp):

    def __init__(self, interp=cv2.INTER_LINEAR):
        Pansharp.__init__(self,  interp)

    def __init__(self, count, weights=None, dtype=np.uint8, interp=cv2.INTER_LINEAR, to8bit=to8bit_clip):
        """

        :param count:
        :param weights:
        :param dtype:
        :param interp:
        :param to8bit:
        """

        self.count = None
        self.weights = None

        if weights is not None:
            self.count = len(weights)
            weights = np.array([[weights]]).transpose(2, 1, 0)
            self.weights = weights / sum(weights)

        Pansharp.__init__(self, dtype, interp, to8bit)

    def _calculate_ratio(self, pan, ms):
        return pan / (np.sum(ms * self.weights, 0))

    def sharpen(self, pan, ms):
        ratio = self._calculate_ratio(pan, ms)
        sharp = np.clip(ratio * ms, 0, np.iinfo(pan.dtype).max)
        return sharp.astype(pan.dtype)