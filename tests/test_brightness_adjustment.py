import pytest
import numpy as np
from pysharpen.methods import LinearBrightnessScale

# =================== utils for generation of the test and gt data ========================#


# Get the random test data. Must work with any data
def generate_data(dtype=np.uint8, channels=3, m=0, M=255):
    if np.issubdtype(dtype, np.integer):
        m = int(m)
        M = int(M)
        return np.random.randint(m, M, (1000, 1000), dtype),\
           np.random.randint(m, M, (channels, 1000, 1000), dtype)
    else:
        return (np.random.sample((1000, 1000))*(M-m) + m).astype(dtype),\
           (np.random.sample((channels, 1000, 1000))*(M-m) + m).astype(dtype)


# Get the image statistics from the whole image to compare with tile-based variant
def get_setup_parameters(pan, ms, pan_separate=False, per_channel=False):

    if not pan_separate:
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


# Processing of the image as a whole to compare with the tile-based approach
def linear_brightness_scale_whole(pan, ms, method,
                 per_channel=False, pan_separate=False, width=3, hist_cut=0.01):
    pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan, ms,
                                                                                                pan_separate, per_channel)

    if method == 'minmax':
        pan = (pan.astype(np.float32) - pan_min)/(pan_max-pan_min)*255
        ms = (ms.astype(np.float32) - ms_min)/(ms_max-ms_min)*255
        return pan.astype(np.uint8), ms.astype(np.uint8)

    elif method == 'meanstd':
        pan_min = max(pan_min, pan_mean - width*pan_std)
        ms_min = max(ms_min, ms_mean - width * ms_std)
        pan_max = min(pan_max, pan_mean + width*pan_std)
        ms_max = min(ms_max, ms_mean + width * ms_std)
        pan = (pan.astype(np.float32) - pan_min)/(pan_max-pan_min)*255
        ms = (ms.astype(np.float32) - ms_min)/(ms_max-ms_min)*255
        return pan.astype(np.uint8), ms.astype(np.uint8)

    elif method == 'histogram':
        # TODO: add test
        return pan, ms

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
        LinearBrightnessScale(method='incorrect')
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


# =========== Test of preparation function =================#
# Here we ensure that the tile-based preparation functions within our class work the same as a whole-image ones #

def test_preparation_minmax():
    pan, ms = generate_data()
    for pan_separate in [True, False]:
        for per_channel in [True, False]:
            pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan,
                                                                                                        ms,
                                                                                                        pan_separate,
                                                                                                        per_channel)

            method = LinearBrightnessScale('minmax', per_channel=per_channel, pan_separate=pan_separate)
            for i in range(pan.shape[0]/100):
                for j in range(pan.shape[1]/100):
                    method.setup_from_patch(pan[i*100:(i+1)*100, j*100:(j+1)*100],
                                                    ms[:, i*100:(i+1)*100, j*100:(j+1)*100])
            method.finalize_setup()

            assert pan_min == method._pan_min_value
            assert pan_max == method._pan_max_value
            assert ms_min == method._ms_min_value
            assert ms_max == method._ms_max_value


def test_preparation_meanstd():
    pan, ms = generate_data()
    print('Pan: ', pan.shape, pan.dtype)
    print('MS: ', ms.shape, ms.dtype)
    for pan_separate in [True, False]:
        for per_channel in [True, False]:
            # different widths to check. Actually,
            for width in (0, 0.5, 1, 2, 10):
                pan_min, pan_max, pan_mean, pan_std, ms_min, ms_max, ms_mean, ms_std = get_setup_parameters(pan,
                                                                                                            ms,
                                                                                                            pan_separate,
                                                                                                            per_channel)

                method = LinearBrightnessScale('meanstd',
                                               per_channel=per_channel, pan_separate=pan_separate,
                                               std_width=width)
                for i in range(pan.shape[0] / 100):
                    for j in range(pan.shape[1] / 100):
                        method.setup_from_patch(pan[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                ms[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100])
                method.finalize_setup()

                assert np.max([pan_mean - width*pan_std, pan_min]) == method._pan_min_value
                assert np.min([pan_mean + width*pan_std, pan_max]) == method._pan_max_value
                assert np.max([ms_mean - width*ms_std, ms_min], axis = 0 if per_channel else None) == method._ms_min_value
                assert np.min([ms_mean + width*ms_std, ms_max], axis = 0 if per_channel else None) == method._ms_max_value


def test_preparation_historgram():
    pass

# ==================== Test of processing ==================== #


# Test all the methods at different data
def test_processing():
    test_data = [generate_data(np.uint8, channels=1),
                 generate_data(np.uint8, channels=5, m=50, M=60),
                 generate_data(np.uint16, channels=3, m=0, M=4500),
                 generate_data(np.in32, channels=3, m=-50, M=4500),
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
                for width in (0, 0.5, 1, 2, 10):
                    pan_gt, ms_gt = linear_brightness_scale_whole(pan, ms, method='meanstd', pan_separate=pan_separate,
                                                                  per_channel=per_channel, width=width)

                    method = LinearBrightnessScale('meanstd', dtype=np.uint8,
                                                   per_channel=per_channel, pan_separate=pan_separate,
                                                   std_width=width)
                    for i in range(pan.shape[0] / 100):
                        for j in range(pan.shape[1] / 100):
                            method.setup_from_patch(pan[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                    ms[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100])
                    method.finalize_setup()

                    for i in range(pan.shape[0] / 100):
                        for j in range(pan.shape[1] / 100):
                            pan_res, ms_res = method.process(pan[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                    ms[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100])
                            
                            # We test every output patch to be equal to the according patch of the processed as whole image
                            assert pan_res == pan_gt[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]
                            assert ms_res == ms_gt[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]

        # MINMAX
        for pan_separate in [True, False]:
            for per_channel in [True, False]:
                pan_gt, ms_gt = linear_brightness_scale_whole(pan, ms, method='minmax', pan_separate=pan_separate,
                                                              per_channel=per_channel, width=width)

                method = LinearBrightnessScale('minmax', dtype=np.uint8,
                                               per_channel=per_channel, pan_separate=pan_separate,
                                               std_width=width)
                for i in range(pan.shape[0] / 100):
                    for j in range(pan.shape[1] / 100):
                        method.setup_from_patch(pan[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                ms[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100])
                method.finalize_setup()

                for i in range(pan.shape[0] / 100):
                    for j in range(pan.shape[1] / 100):
                        pan_res, ms_res = method.process(pan[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                ms[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100])

                        # We test every output patch to be equal to the according patch of the processed as whole image
                        assert pan_res == pan_gt[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]
                        assert ms_res == ms_gt[:, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]