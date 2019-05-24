import numpy as np
import cv2
from .method import Method


class IHS(Method):

    def __init__(self):
        super.__init__(self)
        #self.count = count

    def sharpen(self, pan, ms):
        if ms.shape[0] == 3:
            return self._sharpen_3(pan, ms)
        #elif ms.shape[0] == 4:
        #    return self._sharpen_4(pan, ms)
        else:
            raise ValueError("Only 3-channel transform is supported")

    def _sharpen_3(self, pan, ms):
        if ms.dtype != np.uint8:
            pan = pan.astype(np.float32)
            ms = ms.astype(np.float32)
        rgb = ms.transpose(1, 2, 0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)
        return RGBpan.transpose(2, 0, 1).astype(ms.dtype)

    '''
    Not implementing until find something better than Brovey
    def _sharpen_4(self, pan, ms):
        # All images must be already adjusted to the same size
        if ms.shape[0] != self.count:
            raise ValueError("The sharpener requires {} channels, got {} instead".format(self.count, ms.shape[0]))
        # The opencv color space transforms ca nwork with 8bit integers and 32-bit floats only
        if ms.dtype != np.uint8:
            pan = pan.astype(np.float32)
            ms = ms.astype(np.float32)
        RGB = ms[:3].transpose(1, 2, 0)
        HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
        H = HSV[:, :, 0]
        S = HSV[:, :, 1]

        HSpan = np.array([H, S, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)

        # Now making the same for NIRGB
        NIRGB = np.array([ms[3], ms[1], ms[2]]).transpose(1, 2, 0)
        HSV = cv2.cvtColor(NIRGB, cv2.COLOR_RGB2HSV)
        H = HSV[:, :, 0]
        S = HSV[:, :, 1]
        HSpan = np.array([H, S, pan]).transpose(1, 2, 0)

        NIRGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)
        nir_pan = NIRGBpan[:, :, 0]

        # Now assemble RGBNIR image in the right order
        return np.array([RGBpan[:, :, 0], RGBpan[:, :, 1], RGBpan[:, :, 2], nir_pan])
        '''