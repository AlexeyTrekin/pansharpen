import numpy as np
"""
The purpose of this module is to change the data types of the images in convenient way.
These are not the default numpy-like methods of casting but the ways to convert image type without loss of quality
or even with 

Also includes utils to handle numpy dtypes better
"""


def value_range(dtype):
    """
    returns a tuple (min value, max value) for integer dtype
    or (0,1) for float dtype
    :param dtype: numpy data type or its string representation
    :return:
    """
    if np.issubdtype(dtype, np.integer):
        out_range = (np.iinfo(dtype).min, np.iinfo(dtype).max)
    else:
        out_range = (0., 1.)
    return out_range


def wider_type(dtype):
    """
    Returns the dtype which can handle arithmetic with the specified dtype without overflow.
    However, for the multiplication/division it is more convenient to cast to float32 in any case

    # TODO: decide whether we need or np.result_type is enough
    :param dtype: numpy data type or its string representation
    :return:
    """
    if dtype == np.int8 or dtype == np.uint8:
        return np.int16
    if dtype == np.int16 or dtype == np.uint16:
        return np.int32
    else:
        # actually, we may assume that other types are unlikely and wide enough to fit all the calculations
        return dtype


def saturate_cast(img, dtype):
    """
    Implementation of opencv saturate_cast, which changes datatype and clips the data to the output type range
    :param img: input image
    :param dtype: data type of output image
    :return: new image of dtype
    """
    if np.issubdtype(dtype, np.integer):
        return np.clip(np.around(img,0), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        return img.astype(dtype)
