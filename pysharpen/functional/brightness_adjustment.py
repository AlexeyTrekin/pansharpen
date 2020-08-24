import numpy as np
import warnings
from .dtype_utils import value_range, saturate_cast

"""
The purpose of these methods is to scale the image values to enhance the image brightness.

The image type remains unchanged.
If not specified, the range of the output image is 0:1 for floating-point and full allowed image range for integer

We do not assume the input value range becase... why, exactly?
"""


def linear_brightness_scale(img, input_min, input_max,
                            dtype=None, out_range=None):
    """
    linearly rescales values of the images from [input_min, input_max] to [out_min, out_max] range
     img:
    :param input_min:
    :param input_max:
    :param dtype
    :param out_range:
    :return:
    """

    if out_range is None:
        out_range = value_range(img.dtype)

    if input_max < input_min or out_range[0] > out_range[1]:
        raise ValueError('Maximum value of must not be less than minimum')
    if input_max == input_min or out_range[0] == out_range[1]:
        warnings.warn('Minimum and maximum value of the initial range are equal, scaling is undetermined, '
                      'returning array of minimum value of out range')
        return np.ones_like(img)*img.dtype(out_range[0])
    #  We use float32 inside to avoid loss of data for the integer values
    if np.issubdtype(img.dtype, np.integer):
        img_float = (img.astype('float32')-input_min)/(input_max-input_min)
    else:
        img_float = (img - input_min) / (input_max - input_min)
    if dtype is None:
        dtype = img.dtype
    return saturate_cast(img_float*(out_range[1] - out_range[0]) + out_range[0], dtype)


def gamma_correction(img, gamma):
    return img


def retinex(img, sigma):
    return img
