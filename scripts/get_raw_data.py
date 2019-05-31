
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

def run(rate,freq,front_end, tones, lapse, gain):


    noise_filename = u.get_NODSP_tones(tones, measure_t = lapse, rate = rate, amplitudes = None,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = 0, pf_average = 4, tx_gain = gain)

    return noise_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Put some tones in output and store unprocessed data')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--gain', '-g', help='TX noise', type=int, default= 0)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--tones','-T', nargs='+', help='Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=2)
    parser.add_argument('--VNA', '-vna', help='VNA file containing the resonators. Relative to the specified folder above.', type=str)


    args = parser.parse_args()
    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if args.VNA is not None:
        rf_freq, tones = u.get_tones(args.VNA)
        u.print_debug("getting %d tones from %s" % (len(tones),args.VNA))
    else:
        try:
            tones = [float(x) for x in args.tones]
            tones = np.asarray(tones)*1e6
        except ValueError:
            u.print_error("Cannot convert tone arfreqgument.")

        rf_freq = args.freq*1e6

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    # Data acquisition

    f = run(rate = args.rate*1e6, freq = rf_freq, front_end = args.frontend,
            tones = np.asarray(tones), lapse = args.time, gain = args.gain)

    # Data analysis and plotting will be in an other python script
