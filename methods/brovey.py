import numpy as np
from .method import Method

class Brovey(Method):

    def __init__(self, count, weights=None):
        if weights is None:
            weights = [1] * count
        elif len(weights) != count:
            raise ValueError("Weights number must be equal to channels number")

        weights = np.array([[weights]]).transpose(2, 1, 0)
        self.weights = weights / sum(weights)
        self.count = count

    def _calculate_ratio(self, pan, ms):
        return pan / (np.sum(ms * self.weights, 0))

    def sharpen(self, pan, ms):
        ratio = self._calculate_ratio(pan, ms)
        sharp = np.clip(ratio * ms, 0, np.iinfo(pan.dtype).max)
        return sharp.astype(pan.dtype)