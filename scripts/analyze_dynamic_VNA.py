
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

def run(backend, f, decim):
    print f
    u.VNA_timestream_analysis(filename = f, usrp_number = 0)
    u.VNA_timestream_plot(f, backend = backend, mode = 'magnitude')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--plot_decimate', '-d', help='deciamte data in the plot to get lighter files', type=int, default= None)


    args = parser.parse_args()
    os.chdir(args.folder)

    list_of_files = glob.glob("USRP_VNA*.h5")
    latest_file = [x.split(".")[0] for x in (sorted(list_of_files, key=os.path.getctime))][0]
    run(backend = args.backend, f = latest_file, decim = args.plot_decimate)
