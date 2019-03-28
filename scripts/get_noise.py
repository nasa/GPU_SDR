
import sys,os
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

import argparse

def run(rate,freq,front_end, tones, lapse, decimation):


    noise_filename = u.Get_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = None,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = None, pf_average = 4)

    return noise_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--tones','-T', nargs='+', help='Tones in MHz as a list i.e. -T 1 2 3', required=True)
    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=10)

    args = parser.parse_args()

    try:
        tones = [float(x) for x in args.tones]
    except ValueError:
        u.print_error("Cannot convert tone argument.")

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    # Data acquisition

    cmd = ""
    filenames = []
    while cmd != "S":
        f = run(rate = args.rate*1e6, freq = args.freq*1e6, front_end = args.frontend,
                tones = np.asarray(tones)*1e6, lapse = args.time, decimation = args.decimation)
        cmd = raw_input("Press enter to measure again or type S to stop.")
        filenames.append(f)

    # Data analysis and plotting will be in an other python script