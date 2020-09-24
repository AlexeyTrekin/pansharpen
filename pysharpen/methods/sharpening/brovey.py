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

        if weights is not None:
            self.count = len(weights)
            weights = np.array([[weights]]).transpose(2, 1, 0)
            self.weights = weights / sum(weights)
        else:
            self.weights = 1
        self.setup_required = False

    def _calculate_ratio(self, pan, ms):
        return pan / (np.sum(ms * self.weights, 0))

    def process(self, pan, ms):
        ratio = self._calculate_ratio(pan, ms)
        res = ratio * ms
        return pan, saturate_cast(res, ms.dtype)
