import cv2
import numpy as np
from .pansharp import Pansharp
from pysharpen.preprocessing.type_conversion import saturate_cast, wider_type, scale


class IHS(Pansharp):
    """
     IHS pansharpening is for RGB image only. It transforms RGB into IHS color space,
     then replaces Intensity with PAN channel and transforms backwards. Based on OpenCV color transforms
    """

    def __init__(self):
        Pansharp.__init__(self)

    def sharpen(self, pan, ms):

        """

        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker

        Returns:
            pansharpened image, the same size as pan, but with number of channels as in ms, with IHS method
        """

        assert pan.shape == ms.shape[1:], 'Shapes of multispectral and panchrom are different'
        assert ms.shape[0] == 3, 'IHS pansharpening is restricted to 3 channels, use GIHS for others'
        assert ms.dtype == pan.dtype, f'Data types of multispectral {ms.dtype} and panchrom {pan.dtype} are different'
        dtype = pan.dtype
        ms = ms.transpose(1, 2, 0)
        
        #default coefficient for no scaling
        m = 0
        M = 1
        
        if dtype != np.uint8:
        # Opencv works with uint8 and float32. All other types are converted to float32 
        # and translated to 0:1 range. 
        # Float32 images are treated the same way in order to manage the value range
        # We assume that the pan and rgb images are in the same brightness range.
        # Histogram equalization can be managed as an additional step, if necessary
            m = min(pan.min(), ms.min())
            M = max(pan.max(), ms.max())
            if m == M:
                ms = np.zeros(ms.shape, np.float32)
                pan = np.zeros(pan.shape, np.float32)
            else:
                ms = (ms.astype(np.float32) - m)/(M-m)
                pan = (pan.astype(np.float32) - m)/(M-m)
                
        hsv = cv2.cvtColor(ms, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
        RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)

        #if not (dtype == np.uint8 or dtype == np.float32):
        RGBpan = (RGBpan*(M-m) + m).astype(dtype)
          
        return RGBpan.transpose(2, 0, 1)