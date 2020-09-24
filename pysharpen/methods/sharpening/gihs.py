import numpy as np
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import saturate_cast, wider_type


class GIHSPansharpening(ImgProc):
    """
    Generalized IHS-like transform for any number of channels.
    Formula for the output is: MSpan[i] = MS[i] + pan - mean (MS)
    Similar to Brovey, but additive rather than multiplicative
    """

    def __init__(self):
        super().__init__()

    def process(self, pan, ms):
        """
        :param pan:
        :param ms:
        :return:
        """
        # The methods expect already resized images
        if pan.shape != ms.shape[1:]:
            raise ValueError('Shapes of multispectral and panchrom are different')
        ms = ms.transpose(1, 2, 0)

        ms_pansharpened = ms.astype(wider_type(pan.dtype))
        mean = np.expand_dims(np.sum(ms_pansharpened, axis=2) / ms_pansharpened.shape[0], 2)
        ms_pansharpened = ms_pansharpened + np.expand_dims(pan.astype(wider_type(ms.dtype)), 2) - mean

        ms_pansharpened = saturate_cast(ms_pansharpened, ms.dtype)
        return pan, ms_pansharpened.transpose(2, 0, 1)