
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
def run(backend, files):
    u.plot_frequency_timestreams(files, decimation=None, low_pass=None, backend=backend, output_filename=None,
              channel_list=None, auto_open=True, displayed_samples = 10000)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--mode', '-m', help='Plotting mode to use. Usually PM or IQ', type=str, default = "data")


    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")

    run(backend = args.backend, files = files)
