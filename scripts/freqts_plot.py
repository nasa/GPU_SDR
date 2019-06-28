
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
def run(backend, files, decimation, low_pass, channel_list, displayed_samples):
    u.plot_frequency_timestreams(files, decimation=decimation, low_pass=None, backend=backend, output_filename=None,
              channel_list=ch_list, auto_open=True, displayed_samples = displayed_samples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--decimation', '-d', help='decimation factor for plotting', type=float)
    parser.add_argument('--displayed_samples', '-ds', help='Total number of samples per channel to plot. override decimation', type=int)
    parser.add_argument('--channel_list', '-ch', help='Channel number to plot, single value', type=int)
    parser.add_argument('--lowpass', '-lp', help='Low pass filter order', type=int)


    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    if args.channel_list is not None:
        ch_list = [args.channel_list]
    else:
        ch_list = None

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")

    run(backend = args.backend, files = files, decimation = args.decimation, displayed_samples = args.displayed_samples, channel_list = ch_list, low_pass = args.lowpass)
