
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

def run(backend, files, welch, dbc):
    for f in files:
        u.calculate_noise(f, verbose = True, welch = max(welch,1), dbc = dbc, clip = 0.1)

    print u.plot_noise_spec(files, channel_list=None, max_frequency=None, title_info=None, backend=backend,
                    cryostat_attenuation=0, auto_open=True, output_filename=None, add_info = ["decimation: 100x fs: 100Msps","loopback decimation 100x","decimation: OFF fs: 1Msps"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--welch', '-w', help='Whelch factor relative to timestream length so that welch factor is len(timestream)/this_arg', type=int, default= 5)
    parser.add_argument('--dbc', '-dbc', help='Analyze and plot in dBc or not', action="store_true")



    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")

    run(backend = args.backend, files = files, welch = args.welch, dbc = args.dbc)
