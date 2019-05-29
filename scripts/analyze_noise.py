
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
    for f in files:
        u.calculate_noise(f, verbose = True, welch = 7, dbc = False, clip = 0.1)

    print u.plot_noise_spec(files, channel_list=None, max_frequency=1e3, title_info=None, backend=backend,
                    cryostat_attenuation=0, auto_open=True, output_filename=None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")


    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")

    run(backend = args.backend, files = files)
