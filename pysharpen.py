import sys
import click
from pathlib import Path
from pysharpen.worker import Worker
from pysharpen.methods import IHS, Brovey

if __name__ == "__main__":
    """
    CLI: python worker.py panchrom_name.tif multispectral_name.tif out_name.tif method
    method = <ihs|brovey>
    """
    if len(sys.argv) < 5:
        print ("Usage: python worker.py panchrom_name.tif multispectral_name.tif out_name.tif <ihs|brovey>")
        exit(0)
    pan = Path(sys.argv[1])
    ms = Path(sys.argv[2])
    out = Path(sys.argv[3])
    if sys.argv[4].lower() == 'ihs':
        method = IHS
    else:
        method = Brovey

    if not pan.exists():
        print("Panchrom file does not exist")
        exit(0)
    if not ms.exists():
        print("MS file does not exist")
        exit(0)

    try:
        w = Worker(method=method)
    except Exception as e:
        print('Error in pansharpening')
        print(str(e))
        exit(-1)