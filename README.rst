
A library for remote sensing image pansharpening and enhancement
================================================================

**The goal is to assemble all the known good methods within one useful python library**

The data processing is based on rasterio bindings to GDAL binaries.
Big files are supported via windowed IO.


Use of library:
---------------


Worker class is initialized with a list of methods that are applied to the pair of images (pan, ms) consequently.
Some of the methods may need the setup stage where they gather the image statistics before processing,
use `worker.setup_methods()` before `worker.process()` to do it. The setup is also carried out with windowed reading.

`worker.process_single` and `worker.process_separate()` applies to the set of files
 and includes methods' setup and application.


Pansharpening methods:
----------------------

By the moment, only the very basic methods are supported:
   1. IHS
   2. Brovey
   3. Generalized IHS for any number of channels


Preprocessing methods:
----------------------


   1. Linear brightness scaling allowes to stretch the brightness to the whole range of the data format or fit to 8bit. The initial values range can be defined in the following variants:

      - min - max,
      - mean +- WIDTH*std


Adding your own method:
-----------------------

You can inherit ImgProc class, implementing process() function to add any preprocessing, pansharpening or postprocessing
function you need.


Command line interface:
-----------------------

CLI contains a subset of the methods that can be used off the shelf,
but for full functionality it is better to use the library

``pysharpen [--method METHOD] [--preprocessing PREPROCESSING] [--resampling RESAMPLING] [--nogeo] [--noclean] pan_file ms_file out_file``



Options:
- method : the main pansharpening method, allowed options are `ihs`, `brovey`, `gihs`, default `ihs`
- preprocessing: the optional step of image preparation before pansharpening allowed options are `minmax` and `meanstd`
(the latter with parameter WIDTH=3), default `none`
- resampling: a method for multispectral image resampling to the resolution of panchrom image.
Allowed options are `bilinear`, `nearest`, `cubic` etc., default `bilinear`
- nogeo: if checked, the images are opened and resampled without regard to the georeference. It is necessary when the
data is not georeferenced at all, or can be used if the image is small and windowed IO is not necessary
- noclean: if checked, the intermediate files are not removed. This can be useful if you need the separated channel-by-channel
to separate files multispectral data

Known issues
------------
The windowed georeferenced IO gives not the same result as whole image IO without georeference,
this is caused by the subpixel misplacement and will be addressed in the next updates.

Contributing
------------
You can describe any problem with the package via issues at github.
Your contributions are always welcome, especially new widely-used pansharpening methods.