from pysharpen.methods.pansharp import Pansharp
import cv2

class Noaction(Pansharp):

    def __init__(self, interp=cv2.INTER_LINEAR):
        self.count = None
        self.interp = interp

    def setup(self, pan, ms):
        pass

    def sharpen(self, pan, ms):
        """

        :param pan: panchromatic image, 2-dimensional numpy array
        :param ms: multispectral image, 3-dimensional numpy array, channels-last, the same territory with pan;
        :return: pansharpened image, the same size as pan, but with number of channels as in ms
        """
        return ms