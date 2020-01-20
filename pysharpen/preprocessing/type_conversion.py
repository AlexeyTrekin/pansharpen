import numpy as np
import warnings

def wider_type(dtype):
    if dtype == np.int8 or dtype == np.uint8:
        return np.int16
    if dtype == np.int16 or dtype == np.uint16:
        return np.int32
    else:
        # actually, we may assume that other types are unlikely and wide enough to fit all the calculations
        return dtype


def saturate_cast(img, dtype):
    """
    Implementation of opencv saturete_cast, which chenges datatype and clips the data to the output type range
    :param img: input image
    :param dtype: data type of output image
    :return: new image of dtype
    """
    if np.issubdtype(dtype, np.integer):
        return np.clip(img, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        return img.astype(dtype)


def scale(img, dtype,
          in_min=None, in_max=None,
          out_min=None, out_max=None):
    """
    Scales linearly the brightness of the input image to the full range of the output data type and casts the image
    to that type.
    If the range m - M of the initial image is not specified, it is set to image actual min-max range
    :param img: image to be scaled
    :param dtype: output data type
    :param in_min: the value of the initial image brightness to be cast to the minimum of output
    :param in_max: the value of the initial image brightness to be cast to the maximum of output
    :param out_min: the minimum value of output
    :param out_max: the maximum value of output
    :return:
    """
    # assuming range of the input image if it is not specified
    if in_min is None:
        in_min = img.min()
    if in_max is None:
        in_max = img.max()
    # special case with constant value is undetermined
    if in_min == in_max:
        warnings.warn('Minimum and maximum value of the initial range are equal, scaling is undetermined, '
                      'returning zero array')
        return np.zeros(img.shape, dtype)

    # default output range for int is full data type range
    if np.issubdtype(dtype, np.integer):
        if out_min is None:
            out_min = np.iinfo(dtype).min
        if out_max is None:
            out_max = np.iinfo(dtype).max
    # default range for float is 0:1
    else:
        if out_min is None:
            out_min = 0.
        if in_max is None:
            out_max = 1.
    # how to choose the data type to preserve data from overflow?
    scaled = (img - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return saturate_cast(scaled, dtype)


def cast_meanstd(img, dtype, width=3.0,
          out_min=None, out_max=None):
    mean = np.mean(img)
    std = np.std(img)

    in_min = max(0., mean - width * std)
    in_max = min(img.max(), mean + width * std)

    res = scale(img, dtype, in_min, in_max, out_min, out_max)
    return res



def to8bit_meanstd(img16, WIDTH=3):
    """

    :param img16:
    :param WIDTH:
    :return:
    """
    mean = np.mean(img16)
    std = np.std(img16)

    m = max(0, mean - WIDTH * std)
    M = min(img16.max(), mean + WIDTH * std)

    img8 = ((img16 - m) * 255.0 / (M - m)).clip(1, 255).astype(np.uint8)
    return img8


def to8bit_clip(img16, percent=0.1):
    """

    :param img16:
    :param percent:
    :return:
    """
    channels = min(img16.shape)
    data = img16.flatten()
    V = data.shape[0] / channels
    dv = V * percent / 100

    hist = np.histogram(data, bins=data.max() - data.min())[0]

    tail = 0
    m = 0
    while tail < dv:
        tail += hist[m]
        m += 1

    tail = 0
    M = data.max() - data.min() - 1
    while tail < dv:
        tail += hist[M]
        M -= 1

    # print (m, M)
    img8 = ((img16.astype(np.int16) - m) * 255.0 / (M - m)).clip(1, 255).astype(np.uint8)
    return img8