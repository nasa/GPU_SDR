
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

def run(end_time, mode, backend, files, channel_list, decimation, displayed_samples, lowpass):
    u.plot_raw_data(files, decimation=decimation, low_pass=lowpass, backend=backend, output_filename=None,
              channel_list=channel_list, mode=mode, auto_open=True, displayed_samples = displayed_samples, end_time = end_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--mode', '-m', help='Plotting mode to use. Usually PM or IQ', type=str, default = "IQ")
    parser.add_argument('--decimation', '-d', help='decimation factor for plotting', type=float)
    parser.add_argument('--displayed_samples', '-ds', help='Total number of samples per channel to plot. override decimation', type=int)
    parser.add_argument('--channel_list', '-ch', help='Channel number to plot', type=int)
    parser.add_argument('--lowpass', '-lp', help='Low pass filter order', type=int)
    parser.add_argument('--end_time', '-e', help='how many seconds to plot from the beginning. Default is All', type=float)

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")
    if args.channel_list is not None:
        ch_list = [args.channel_list]
    else:
        ch_list = None

    run(end_time = args.end_time, mode = args.mode, backend = args.backend, files = files, decimation = args.decimation, displayed_samples = args.displayed_samples, channel_list = ch_list, lowpass = args.lowpass)
