import random

import json
import pytest
import numpy as np
from pysharpen.methods import LinearBrightnessScale
SIZE = 7
# =================== utils for generation of the test and gt data ========================#


# Get the random test data. Must work with any data
def generate_data(dtype=np.uint8, channels=3, m=10, M=200):
    if np.issubdtype(dtype, np.integer):
        m = int(m)
        M = int(M)
        return np.random.randint(m, M, (10*SIZE, 10*SIZE), dtype),\
           np.random.randint(m, M, (channels, 10*SIZE, 10*SIZE), dtype)
    else:
        return (np.random.sample((10*SIZE, 10*SIZE))*(M-m) + m).astype(dtype),\
           (np.random.sample((channels, 10*SIZE, 10*SIZE))*(M-m) + m).astype(dtype)


def generate_batch():
    return [generate_data(np.uint8, channels=1),
     generate_data(np.uint8, channels=5, m=50, M=60),
     generate_data(np.uint16, channels=3, m=0, M=4500),
     generate_data(np.int32, channels=3, m=-50, M=4500),
     generate_data(np.float32, channels=4, m=0, M=1),
     generate_data(np.float32, channels=2, m=10., M=1000.)]


# Get the image statistics from the whole image to compare with tile-based variant
def get_setup_parameters(pan, ms, pan_separate=False, per_channel=False):

    if not pan_separate and not per_channel:
        all_img = np.concatenate([ms, np.expand_dims(pan, 0)], axis=0)
        total_min = all_img.min()
        total_max = all_img.max()
        total_mean = all_img.mean()
        total_std = all_img.std()
        return total_min, total_max, total_mean, total_std, total_min, total_max, total_mean, total_std
    #
    else:
        pan_min = pan.min()
        pan_max = pan.max()
        pan_std = pan.std()
        pan_mean = pan.mean()
        if not per_channel:
            ms_min = ms.min()
            ms_max = ms.max()
            ms_std = ms.std()
            ms_mean = ms.mean()
        else:
            ms_min = ms.min(axis=(1, 2))
            ms_max = ms.max(axis=(1, 2))
            ms_std = ms.std(axis=(1, 2))
            ms_mean = ms.mean(axis=(1, 2))
        return pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std


# Test of the patch-based functions
def test_meanstd_functional():

    test_data = generate_batch()

    for data in test_data:
        for img in data:
            if img.shape[-2:] != (10 * SIZE, 10 * SIZE):
                raise ValueError('Wrong size')
            gt_mean = img.mean()
            gt_std = img.std()
            means = []
            stds = []
            nums = []
            mes = []
            for i in range(10):
                for j in range(10):
                    if img.ndim == 2:
                        mean, std, num, me = LinearBrightnessScale._meanstdme(img[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])
                    else:
                        mean, std, num, me = LinearBrightnessScale._meanstdme(img[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])
                    means.append(mean)
                    stds.append(std)
                    nums.append(num)
                    mes.append(me)

            patch_mean, patch_std = LinearBrightnessScale._totalmeanstd(means, stds, nums, mes)

            assert np.allclose(patch_mean, gt_mean) and np.allclose(patch_std, gt_std)


# Processing of the image as a whole to compare with the tile-based approach
def linear_brightness_scale_whole(pan, ms, method,
                 per_channel=False, pan_separate=False, width=3, hist_cut=0.01, dtype=np.uint8):
    pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan, ms,
                                                                                                pan_separate, per_channel)
    #print(pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std)

    if method == 'meanstd':
        axis=0 if per_channel else None
        pan_min = max(pan_min, pan_mean - width*pan_std)
        ms_min = np.max([ms_min, ms_mean - width * ms_std], axis=axis)
        pan_max = min(pan_max, pan_mean + width*pan_std)
        ms_max = np.min([ms_max, ms_mean + width * ms_std], axis=axis)

    elif method == 'histogram':
        # TODO: add test
        return pan, ms

    # MINMAX does not require any pre-setup so it goes right from here
    if np.issubdtype(dtype, np.integer):
        pan_res = (np.floor_divide(np.multiply((pan - pan_min),
                                               255,
                                               dtype=np.float64),
                                   (pan_max - pan_min))).clip(0, 255)
        if per_channel:
            ms_res = np.zeros_like(ms)
            for channel in range(ms.shape[0]):
                ms_res[channel] = (np.floor_divide(np.multiply((ms[channel] - ms_min[channel]),
                                                               255,
                                                               dtype=np.float64)
                                                   , (ms_max[channel] - ms_min[channel]))).clip(0, 255)
        else:
            ms_res = (np.floor_divide(np.multiply((ms - ms_min),
                                                  255,
                                                  dtype=np.float64),
                                      (ms_max - ms_min))).clip(0, 255)

    else:
        pan_res = (np.divide(np.multiply((pan - pan_min),
                                         255,
                                         dtype=np.float64),
                             (pan_max - pan_min))).clip(0, 255)
        if per_channel:
            ms_res = np.zeros_like(ms)
            for channel in range(ms.shape[0]):
                ms_res[channel] = (np.divide(np.multiply((ms[channel] - ms_min[channel]),
                                                         255,
                                                         dtype=np.float64),
                                             (ms_max[channel] - ms_min[channel]))).clip(0, 255)
        else:
            ms_res = (np.divide((ms - ms_min) * 255, (ms_max - ms_min))).clip(0, 255)

    return pan_res.astype(dtype), ms_res.astype(dtype)


# ============================ TEST CASES ===========================#
# Test that the initialization fails on incorrect params

def test_initialization_params_are_checked():

    # Test that it initializes with a set of correct parameters
    for method in ('minmax', 'meanstd', 'histogram'):
        for per_channel in (True, False):
            for pan_separate in (True, False):
                for width in (0.1, 1, 1.4, 3, 10):
                    for hist_cut in (0, 0.0001, 0.5, 0.99):
                        LinearBrightnessScale(method=method, per_channel=per_channel, pan_separate=pan_separate,
                                              std_width=width, hist_cut=hist_cut)

    # Check that exception is raised on incorrect parameters
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(method='WAT')
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(method=1)

    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(std_width=-1)
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(std_width=0)
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(std_width='WAT')

    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(hist_cut=1)
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(hist_cut=1.4)
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(hist_cut=-0.4)
    with pytest.raises(ValueError) as e_info:
        LinearBrightnessScale(hist_cut='WAT')

def test_properties():
    assert LinearBrightnessScale().processing_bound == 0
    assert LinearBrightnessScale().setup_bound == 0

# =========== Test of preparation function =================#
# Here we ensure that the tile-based preparation functions within our class work the same as a whole-image ones #

def test_preparation_minmax():
    for (pan, ms) in generate_batch():
        for pan_separate in [True, False]:
            for per_channel in [True, False]:
                pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan,
                                                                                                            ms,
                                                                                                            pan_separate,
                                                                                                            per_channel)

                method = LinearBrightnessScale('minmax', per_channel=per_channel, pan_separate=pan_separate)
                for i in range(10):
                    for j in range(10):
                        method.setup_from_patch(pan[i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE],
                                                        ms[:, i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE])
                #print(method._ms_maxs, method._ms_mins, method._pan_maxs, method._pan_mins)
                method.finalize_setup()
                assert pan_min == pytest.approx(method._pan_min_value)
                assert pan_max == pytest.approx(method._pan_max_value)
                assert np.allclose(ms_min, method._ms_min_value)
                assert np.allclose(ms_max, method._ms_max_value)


def test_preparation_meanstd():
    pan, ms = generate_data()
    print('Pan: ', pan.shape, pan.dtype)
    print('MS: ', ms.shape, ms.dtype)
    for pan_separate in [True, False]:
        for per_channel in [True, False]:
            # different widths to check. Actually,
            for width in (0.5, 1, 2, 10):
                pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan,
                                                                                                            ms,
                                                                                                            pan_separate,
                                                                                                            per_channel)

                method = LinearBrightnessScale('meanstd',
                                               per_channel=per_channel, pan_separate=pan_separate,
                                               std_width=width)
                for i in range(10):
                    for j in range(10):
                        method.setup_from_patch(pan[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE],
                                                ms[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])

                #print(method._ms_means, method._ms_stds)
                #print(method._pan_means, method._pan_stds)
                #print(pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std)
                method.finalize_setup()

                assert np.max([pan_mean - width*pan_std, pan_min]) == pytest.approx(method._pan_min_value)
                assert np.min([pan_mean + width*pan_std, pan_max]) == pytest.approx(method._pan_max_value)
                assert np.allclose(np.max([ms_mean - width*ms_std, ms_min], axis = 0 if per_channel else None), (method._ms_min_value))
                assert np.allclose(np.min([ms_mean + width*ms_std, ms_max], axis = 0 if per_channel else None), (method._ms_max_value))


def test_preparation_historgram():
    pass

# ==================== Test of processing ==================== #


# Test all the methods at different data
def test_processing_minmax():
    random.seed(42)

    test_data = generate_batch()

    for img_num, data in enumerate(test_data):
        pan = data[0]
        ms = data[1]

        # MINMAX
        for pan_separate in [True, False]:
            for per_channel in [True, False]:
                pan_gt, ms_gt = linear_brightness_scale_whole(pan, ms, method='minmax', pan_separate=pan_separate,
                                                              per_channel=per_channel)
                pan_gt_float, ms_gt_float = linear_brightness_scale_whole(pan, ms, 'minmax', per_channel, pan_separate, dtype=np.float64)
                method = LinearBrightnessScale('minmax', dtype=np.uint8,
                                               per_channel=per_channel, pan_separate=pan_separate)
                for i in range(10):
                    for j in range(10):
                        method.setup_from_patch(pan[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE],
                                                ms[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])
                method.finalize_setup()
                for i in range(10):
                    for j in range(10):
                        pan_res, ms_res = method.process(pan[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE],
                                                ms[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])

                        # We test every output patch to be equal to the according patch of the processed as whole image
                        assert np.allclose(pan_res, pan_gt[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE], atol=1)
                        """
                        res = True
                        for col in range(SIZE):
                            for row in range(SIZE):
                                if pan_res[row, col] != pan_gt[row+i*SIZE, col + j*SIZE]:
                                    print(row, col, pan[row+i*SIZE, col + j*SIZE], pan_res[row, col],
                                          pan_gt[row+i*SIZE, col + j*SIZE], pan_gt_float[row+i*SIZE, col + j*SIZE])
                                    print(json.dumps(float(method._pan_min_value)), json.dumps(float(method._pan_max_value)))
                                    print(json.dumps(float((pan.min()))), json.dumps(float(pan.max())))
                                    print(((pan[row+i*SIZE, col + j*SIZE] - pan.min())*255/(pan.max()-pan.min())))
                                    alt = np.floor_divide(np.multiply((pan[row+i*SIZE, col + j*SIZE] - pan.min()),
                                                          255,
                                                          dtype=np.float64),
                                        (pan.max() - pan.min()))
                                    alt = np.clip(alt, 0,255).astype(np.uint8)
                                    print(alt)
                                    res = False
                        assert res
                        """
                        assert np.allclose(ms_res, ms_gt[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE], atol=1)


def test_processing_meanstd():
    test_data = [generate_data(np.uint8, channels=1),
                 generate_data(np.uint8, channels=5, m=50, M=60),
                 generate_data(np.uint16, channels=3, m=0, M=4500),
                 generate_data(np.int32, channels=3, m=-50, M=4500),
                 generate_data(np.float32, channels=4, m=0, M=1),
                 generate_data(np.float32, channels=2, m=10., M=1000.)]

    for data in test_data:
        pan = data[0]
        ms = data[1]
        print('Pan: ', pan.shape, pan.dtype)
        print('MS: ', ms.shape, ms.dtype)

        # MEANSTD
        for pan_separate in [True, False]:
            for per_channel in [True, False]:
                # different widths to check. Actually,
                for width in (0.5, 1, 2, 10):
                    pan_gt, ms_gt = linear_brightness_scale_whole(pan, ms, method='meanstd', pan_separate=pan_separate,
                                                                  per_channel=per_channel, width=width)

                    method = LinearBrightnessScale('meanstd', dtype=np.uint8,
                                                   per_channel=per_channel, pan_separate=pan_separate,
                                                   std_width=width)
                    for i in range(10):
                        for j in range(10):
                            method.setup_from_patch(pan[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE],
                                                    ms[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])
                    method.finalize_setup()

                    for i in range(10):
                        for j in range(10):
                            pan_res, ms_res = method.process(pan[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE],
                                                             ms[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE])

                            # We test every output patch to be equal to the according patch of the processed as whole image
                            assert np.allclose(pan_res, pan_gt[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE], atol=1)
                            assert np.allclose(ms_res, ms_gt[:, i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE], atol=1)

