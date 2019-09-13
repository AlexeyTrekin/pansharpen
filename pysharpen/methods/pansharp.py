

class Pansharp:

    def __init__(self):
        self.count = None

    def setup(self, pan, ms):
        pass

    def sharpen(self, pan, ms):
        """

        :param pan: panchromatic image, 2-dimensional numpy array
        :param ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker
        :return: pansharpened image, the same size as pan, but with number of channels as in ms
        """
        raise NotImplementedError