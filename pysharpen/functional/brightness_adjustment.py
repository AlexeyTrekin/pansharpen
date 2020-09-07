import numpy as np
import warnings
from .dtype_utils import value_range, saturate_cast

"""
The purpose of these methods is to scale the image values to enhance the image brightness.

The image type remains unchanged.
If not specified, the range of the output image is 0:1 for floating-point and full allowed image range for integer

We do not assume the input value range because... why, exactly?
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
    if dtype is None:
        dtype = img.dtype
    if out_range is None:
        out_range = value_range(dtype)
    elif value_range(dtype)[0] > out_range[0] \
        or value_range(dtype)[1] < out_range[1]:
        raise ValueError('Out range must fit the out data type')

    if input_max < input_min or out_range[0] > out_range[1]:
        raise ValueError('Maximum value of must not be less than minimum')
    if input_max == input_min or out_range[0] == out_range[1]:
        warnings.warn('Minimum and maximum value of the initial range are equal, scaling is undetermined, '
                      'returning array of minimum value of out range')
        return np.ones_like(img)*img.dtype(out_range[0])

    # Here we preserve dtype - only multiplication
    if np.issubdtype(dtype, np.integer):
        out_img = np.floor_divide(
            np.multiply((img - input_min), (out_range[1] - out_range[0]),
                        dtype=np.float32),
            (input_max - input_min)) + out_range[0]
    else:
        out_img = np.divide(
            np.multiply((img - input_min), (out_range[1] - out_range[0]),
                        dtype=np.float32),
            (input_max - input_min)) + out_range[0]
    return saturate_cast(out_img , dtype)


def gamma_correction(img, gamma, input_min=0, input_max=None, dtype=None):
    """
    Gamma-correction, non-linear brightness shift
    Args:
        img: input image
        gamma: the 1/power of the transform
        input_min: this value remains constant, and all lower values are clipped
        input_max: this value remains constant, and all higher values are clipped
        dtype: output data type. The same as input by default
    Returns:

    """
    if dtype is None:
        dtype = img.dtype
    if input_max is None:
        input_max = img.max()

    # gamma is ill-defined out of the range
    img = img.clip(input_min, input_max)

    return saturate_cast((img / input_max-input_min) ^ (1 / gamma) * input_max + input_min, dtype)


