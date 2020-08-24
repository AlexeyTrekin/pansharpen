import numpy as np
from .brightness_adjustment import linear_brightness_scale
from .dtype_utils import value_range
"""
The purpose of this module is to change the data types of the images in convenient way.
These are not the default numpy-like methods of casting but the ways to convert image type without loss of quality
or even with 

Also includes utils to handle numpy dtypes better
"""


def linear_cast_meanstd(img, dtype, width=3.0,
                        out_min=None, out_max=None):
    mean = np.mean(img)
    std = np.std(img)

    in_min = max(0., mean - width * std)
    in_max = min(img.max(), mean + width * std)
    if out_min is not None and out_max is not None:
        out_range = (out_min, out_max)
    else:
        out_range = value_range(dtype)
    return linear_brightness_scale(img, in_min, in_max, dtype, out_range)


def linear_cast_clip(img, dtype, percent=0.1,
                     out_min=None, out_max=None):

    channels = min(img.shape)
    data = img.flatten()
    V = data.shape[0] / channels
    dv = V * percent / 100

    hist = np.histogram(data, bins=data.max() - data.min())[0]

    tail = 0
    in_min = 0
    while tail < dv:
        tail += hist[in_min]
        in_min += 1

    tail = 0
    in_max = data.max() - data.min() - 1
    while tail < dv:
        tail += hist[in_max]
        in_max -= 1

    if out_min is not None and out_max is not None:
        out_range = (out_min, out_max)
    else:
        out_range = value_range(dtype)
    return linear_brightness_scale(img, in_min, in_max, dtype, out_range)
