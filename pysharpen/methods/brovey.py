
import numpy as np
from .pansharp import Pansharp
from pysharpen.preprocessing.type_conversion import saturate_cast


class Brovey(Pansharp):
    """
    Brovey transform for pansharpening.
    Formula for the output is: MSpan[i] = MS[i] * pan / mean (MS)
    Mean is weighted, default weights are equal.
    """

    def __init__(self, weights=None):
        """
        :param weights:
        """
        Pansharp.__init__(self)
        self.weights = None

        if weights is not None:
            self.count = len(weights)
            weights = np.array([[weights]]).transpose(2, 1, 0)
            self.weights = weights / sum(weights)
        else:
            self.weights = 1
        #The method does not require setup, so we set it ready at once


    def _calculate_ratio(self, pan, ms):
        return pan / (np.sum(ms * self.weights, 0))

    def sharpen(self, pan, ms):
        ratio = self._calculate_ratio(pan, ms)
        res = ratio * ms
        return saturate_cast(res, ms.dtype)
