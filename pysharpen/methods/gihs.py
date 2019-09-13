import numpy as np
from .pansharp import Pansharp
from pysharpen.preprocessing.type_conversion import saturate_cast, wider_type


class GIHS(Pansharp):

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
        ms = ms.transpose(1, 2, 0)

        ms_pansharpened = ms.astype(wider_type(pan.dtype))
        mean = np.expand_dims(np.sum(ms_pansharpened, axis=2) / ms_pansharpened.shape[0], 2)
        ms_pansharpened = ms_pansharpened + np.expand_dims(pan.astype(wider_type(ms.dtype)),2) - mean

        # really there can be a problem if pan and mean have different scale, especially if mean is less. Maybe scaling?
        ms_pansharpened = saturate_cast(ms_pansharpened, ms.dtype)
        return ms_pansharpened.transpose(2, 0, 1)