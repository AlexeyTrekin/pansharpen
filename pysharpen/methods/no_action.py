from pansharp import Pansharp


class Noaction(Pansharp):

    def __init__(self):
        self.count = None
        Pansharp.__init__(self)

    def sharpen(self, pan, ms):
        """
        Does nothing
        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-last, the same territory with pan;

        Returns: original ms image

        """
        return ms

