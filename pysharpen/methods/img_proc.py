from aeronet.dataset import BandCollectionSample


class ImgProc:
    def __init__(self):
        self.setup_required = False
        self.nodata = None

    def setup_from_patch(self, pan, ms):
        return

    def finalize_setup(self):
        return

    def process(self, pan, ms):
        """

        Args:
            pan: panchromatic image, 2-dimensional numpy array
            ms: multispectral image, 3-dimensional numpy array, channels-first (rasterio format), as read in worker

        Returns:
            pansharpened image, the same size as pan, but with number of channels as in ms
        """

        raise NotImplementedError

    def get_sharpen_fn(self):
        """
        """

        def fn(collection: BandCollectionSample):
            if collection.count > 1:
                img = collection.numpy()
                pan = img[-1]
                ms = img[:-1]
            else:
                raise ValueError('The samples inside the collection must be a list '
                                 'with 2 or more channels where PAN channel is the last')
            return self.process(pan, ms)

        return fn

    @property
    def processing_bound(self):
        """
        required bound for this processing
        :return:
        """
        return 0

    @property
    def setup_bound(self):
        """
        required bound for setup of this function
        :return:
        """
        return 0
