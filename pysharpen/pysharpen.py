import click
import numpy as np
import rasterio
import warnings

from .worker import Worker
from pysharpen.methods import BroveyPansharpening, IHSPansharpening, GIHSPansharpening, LinearBrightnessScale
from pysharpen.functional import saturate_cast

METHODS= {'ihs' : IHSPansharpening,
          'gihs': GIHSPansharpening,
          'brovey': BroveyPansharpening}
PREPROCESSINGS = ['minmax', 'meanstd', 'none']


def process_nogeo(pan_file, ms_file, out_file,
                   methods, out_dtype, resampling='bilinear'):
    """
    A function to process the images without georeference (or without regard to existing georeference)
    The multispectral file is simply resampled to the panchrom image size and read as whole.

    Args:
        pan_file:
        ms_file:
        out_file:
        methods:
        out_dtype:
        resampling:

    Returns:

    """
    with rasterio.open(pan_file) as pan:
        profile = pan.profile
        pan_img = pan.read(1)
    dtype = out_dtype or profile['dtype']
    with rasterio.open(ms_file) as ms:
        ms_img = np.zeros((ms.count, pan_img.shape[0], pan_img.shape[1]), dtype=ms.profile['dtype'])
        ms.read(out=ms_img, resampling=getattr(rasterio.enums.Resampling, resampling))
        profile.update(count=ms.count, dtype=dtype)
        colorinterp = ms.colorinterp

    for method in methods:
        method.setup_from_patch(pan_img, ms_img)
        method.finalize_setup()
    for method in methods:
        pan_img, ms_img = method.process(pan_img, ms_img)

    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.colorinterp = colorinterp
        dst.write(saturate_cast(ms_img, dtype))


@click.command()
@click.argument('pan_path', type=click.Path(exists=True))
@click.argument('ms_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path(exists=False, writable=True))
@click.option('--method', type=str, required=False, default='ihs')
@click.option('--resampling', type=str, required=False, default='bilinear')
@click.option('--preprocessing', type=click.Choice(PREPROCESSINGS), required=False, default='none')
@click.option('--nogeo', is_flag=True, help='Ignores georeference of the image, and resamples MS to PAN size')
@click.option('--noclean', is_flag=True, help='Leave the intermediate files containing separate spectral bands')
@click.option("-v", is_flag=True, help='Enables verbose output')
#@click.option("--separate", is_flag=True, help='Used the channels are in separate files')
def command(pan_path, ms_path, out_path,
            method='ihs',
            resampling='bilinear',
            preprocessing='none',
            nogeo=False, noclean=False, v=False):
    """
    CLI: pysharpen panchrom_name.tif multispectral_name.tif out_name.tif method <preprocessing>
    method = <ihs|brovey>
    """
    if not v:
        warnings.filterwarnings("ignore")
    methods = []
    out_dtype=None
    # We provide an example of the most basic preprocessing functionality: panchrom is scaled separately,
    # all the spectral channels together
    if preprocessing == 'minmax':
        out_dtype='uint8'
        methods.append(LinearBrightnessScale('minmax', dtype='uint8', pan_separate=True, per_channel=False))
    elif preprocessing == 'meanstd':
        out_dtype='uint8'
        methods.append(LinearBrightnessScale('meanstd', dtype='uint8', pan_separate=True, per_channel=False))

    methods.append(METHODS[method]())

    if nogeo:
        if v:
            print('Start processing of not georeferenced images')
        process_nogeo(pan_path, ms_path, out_path,
                      methods=methods, resampling=resampling, out_dtype=out_dtype)

    else:
        if v:
            print('Start tile-based processing of georeferenced images')
        # todo: detect broken georeference and use nogeo option in this case?
        w = Worker(methods=methods, resampling=resampling, out_dtype=out_dtype)
        w.process_single(pan_path, ms_path, out_path, clean=not noclean, verbose=v)
    if v:
        print('Finished')

if __name__ == "__main__":
    command()