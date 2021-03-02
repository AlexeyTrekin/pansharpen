import warnings
from pathlib import  Path

default_ext = '.tif'

def extension_driver_consistency(cls, file_path, driver):
    file_path = Path(file_path)


    if file_path.suffix not in ['.tif', '.tiff']:
        warnings.warn(f"Extension '{file_path.suffix}' is not supported. \
                            Out file will be written with '{default_ext}' extension")

        file_path = file_path.with_suffix(default_ext)


    if driver not in ['GTiff']:
        driver = 'GTiff'

    return file_path, driver