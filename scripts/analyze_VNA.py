
import sys,os,glob

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

import argparse

def run(backend, files, decim):
    for f in files:
        u.VNA_analysis(f)
    u.plot_VNA(files, backend = backend, plot_decim = decim)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--plot_decimate', '-d', help='deciamte data in the plot to get lighter files', type=int, default= None)


    args = parser.parse_args()
    os.chdir(args.folder)

    files = glob.glob("USRP_VNA*.h5")

    run(backend = args.backend, files = files, decim = args.plot_decimate)
