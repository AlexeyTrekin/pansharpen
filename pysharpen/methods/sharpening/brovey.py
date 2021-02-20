import numpy as np
from pysharpen.methods.img_proc import ImgProc
from pysharpen.functional import saturate_cast


class BroveyPansharpening(ImgProc):
    """
    Brovey transform for pansharpening.
    Formula for the output is: MSpan[i] = MS[i] * pan / mean (MS)
    Mean is weighted, default weights are equal.
    """

    def __init__(self, weights=None):
        """
        :param weights:
        """
        super().__init__()
        self.weights = None
        self.count = None

        if weights is not None:
            self.count = len(weights)
            # shape of weights must be (self.count, 1, 1) to be broadcastable with the MS image
            self.weights = np.array([[weights]]).transpose(2, 1, 0)
            # we normalize the weights sum to be equal to len(weights) to be the same as default case
            self.weights = self.weights * len(weights) / sum(weights)
        else:
            self.weights = 1
        self.setup_required = False

    def _calculate_ratio(self, pan, ms):
        return pan / (np.mean(ms * self.weights, 0))

    def process(self, pan, ms):
        if self.count is not None and self.count != ms.shape[0]:
            raise ValueError(f'The number of ms channels ({ms.shape[0]}) '
                             f'must be equal to the number of weights specified ({self.count})')
        ratio = self._calculate_ratio(pan, ms)
        res = ratio * ms
        return pan, saturate_cast(res, ms.dtype)
