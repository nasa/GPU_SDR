'''
This program install and check python modules needed for pyUSRP to fully work.
To make this work, run it with sudo.
'''
import os
def run():
    try:
        import os,sys,subprocess,importlib
        if os.getuid()!= 0:
            print "Script must have admin privilege. Try running with sudo."
            exit(-1)
    except ImportError:
        print("\033[1;31mERROR\033[0m: Before running this script you must install the os, sys, subprocess and importlib packages!")
        exit()

    modulelist = [
        "numpy",
        "scipy",
        "h5py",
        "Queue",
        "multiprocessing",
        "joblib",
        "subprocess",
        "plotly",
        "colorlover",
        "matplotlib",
        "progressbar2",
        "PyInquirer",
        "yattag",
        "peakutils",
    ]
    err = 0
    try:
        ret = os.system("pip install -U pip")
    except SystemExit as e:
        err+=1
        print("\033[1;31mERROR\033[0m: Cannot upgrade pip.")
        return
    for module in modulelist:
        try:
            ret = subprocess.call([sys.executable, "-m", "pip", "install", module])
            if ret !=0:
                try:
                    importlib.import_module(module)
                except ImportError:
                    err+=1
                    print("\033[1;31mERROR\033[0m: Cannot install or find module %s. pyUSRP will not work." % module)
        except SystemExit as e:
            err+=1
            print("\033[1;31mERROR\033[0m: Cannot install module %s. pyUSRP will not work." % module)

    print("\nProcess complete with %d errors"%err)

if __name__ == "__main__":
    run()
