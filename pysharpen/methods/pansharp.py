

class Pansharp:

    def __init__(self, interp):
        self.count = None
        self.interp=interp

    def setup(self):
        pass

    def sharpen(self, pan, ms):
        """

        :param pan: panchromatic image, 2-dimensional numpy array
        :param ms: multispectral image, 3-dimensional numpy array, channels-last, the same territory with pan;
        :return: pansharpened image, the same size as pan, but with number of channels as in ms
        """
        raise NotImplementedError