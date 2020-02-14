from aeronet.dataset import BandCollectionSample, BandCollection, Band

class Pansharp:

    def __init__(self):
        self.count = None
        self.ready = False

    def setup(self, pan, ms):
        if isinstance(pan, Band) and isinstance(ms, BandCollection):
            return self.setup_from_band_collection(pan, ms)
        elif isinstance(pan, str) and isinstance(ms, str):
            return self.setup_from_files(pan, ms)

    def setup_from_files(self, pan: str, ms: str):
        self.ready = True

    def setup_from_band_collection(self, pan: Band, ms: BandCollection):
        self.ready = True

    def sharpen(self, pan, ms):
        """

        :param pan: panchromatic image, 2-dimensional numpy array
        :param ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker
        :return: pansharpened image, the same size as pan, but with number of channels as in ms
        """
        raise NotImplementedError

    def get_sharpen_fn(self):
        """
        Method setup must have been already carried out
        """
        if not self.ready:
            # However, a method that was set up for one image, could be suboptimal for another one,
            # yet it will be shown as `ready`. What should we do?
            raise ValueError('The panharpening method requires set up to work. '
                             'Please call method.setup() for the input image')

        def fn(collection: BandCollectionSample):
            if isinstance(collection._samples, list) and len(collection) <= 1:
                pan = collection[0].numpy()
                ms = [sample.numpy() for sample in collection[1:]]
            else:
                raise ValueError('The samples inside the collection must be a list with 2 or more channels (PAN + MS)')

            return self.sharpen(pan, ms)

        return fn
