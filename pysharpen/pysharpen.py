import sys
from pathlib import Path
from .worker import Worker
from .methods import IHS, Brovey

def run_cli():
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
    elif sys.argv[4].lower() == 'brovey':
        method = Brovey
    else:
        print("Method is incorrect")
        exit(0)

    if not pan.exists():
        print("Panchrom file does not exist")
        exit(0)
    if not ms.exists():
        print("MS file does not exist")
        exit(0)

    #try:
    w = Worker(method=method)
    w.process(pan, ms, out)
    #except Exception as e:
     #   print('Error in pansharpening')
    #    print(str(e))
     #   exit(-1)

if __name__ == "__main__":
    run_cli(sys.argv)