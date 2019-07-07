
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--VNA', '-vna', help='Ignore the resoator group in the noise file and source the resnators from a vna file', type=str)

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")
    latest_file = [x.split(".")[0] for x in (sorted(files, key=os.path.getctime))][0]

    u.diagnostic_VNA_noise(latest_file, noise_points = None, VNA_file = None, ant = "A_RX2", backend = 'matplotlib')
