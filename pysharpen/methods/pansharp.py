from aeronet.dataset import BandCollectionSample, BandCollection, Band

class Pansharp:

    def __init__(self):
        self.count = None
        # By default the method is treated as always ready. The methods that require the setup for the whole image,
        # should set `ready=False` at creation
        self.ready = True

    def setup(self, pan, ms):
        """
        Prototype for the setup function. It takes either Band for PAN and BandCollection for MS
        or filenames for multispectral and panchromatic bands.

        Inherited classes must implement the setup_from_files and setup_from_band_collection if
        they are needed or leave them as is
        """

        if isinstance(pan, Band) and isinstance(ms, BandCollection):
            return self.setup_from_band_collection(pan, ms)
        elif isinstance(pan, str) and isinstance(ms, str):
            return self.setup_from_files(pan, ms)

    def setup_from_files(self, pan: str, ms: str):
        """

        Args:
            pan: path to PAN file
            ms: path to single MS file

        Returns:

        """
        pass

    def setup_from_band_collection(self, pan: Band, ms: BandCollection):
        """

        Args:
            pan: Band referring to the PAN file
            ms:  BandCollecion referring to the MS bands

        Returns:

        """
        pass

    def sharpen(self, pan, ms):
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
        Method setup must have been already carried out
        """
        if not self.ready:
            # However, a method that was set up for one image, could be suboptimal for another one,
            # yet it will be shown as `ready`. What should we do?
            raise ValueError('The panharpening method requires set up to work. '
                             'Please call method.setup() for the input image')

        def fn(collection: BandCollectionSample):
            if collection.count > 1:
                pan = collection[-1].numpy()
                ms = [sample.numpy() for sample in collection[:-1]]
            else:
                raise ValueError('The samples inside the collection must be a list '
                                 'with 2 or more channels where PAN channel is the last')
            return self.sharpen(pan, ms)

        return fn
