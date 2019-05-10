#########################################################
# DEVELOPED TO MAKE SINGLE FILE CONTAINING A BEAM MAP   #
# OF OPTICALLY COUPLED KIDS                             #
#########################################################

import numpy as np
import sys,os
import time
import velmex as v
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"


import argparse

def run(rate,freq,front_end, tones, lapse, decimation, gain, nx, ny, integ_time):

    v.start_thread(nx,ny,integ_time,15,"beam_map")
    v.wait_for_positioning()
    v.start_scan()
    noise_filename = u.Get_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = None,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = 0, pf_average = 4, tx_gain = gain)
    v.stop()
    return noise_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "../data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--gain', '-g', help='TX noise', type=int, default= 0)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--tones','-T', nargs='+', help='Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--VNA', '-vna', help='VNA file containing the resonators. Relative to the specified folder above.', type=str)
    parser.add_argument('--nx', '-nx', help='Steps in the x direction', type=int, required = True)
    parser.add_argument('--ny', '-ny', help='Steps in the y directions', type=int, required = True)
    parser.add_argument('--integ_time', '-i', help='Duration of each xy step', type=int, required = True)

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

    motor_movement = 60 # one minute per row
    accell_time = 1./3
    calculated_duration = int(args.nx * args.ny * (args.integ_time  + accell_time) + motor_movement*(1+args.nx)) + 100 #just to be sure
    print "Duration will be %.1f minutes"%(calculated_duration/60.)
    resp = ""
    while resp != "n" and resp != "y":
        resp = raw_input("procede? (y/n): ")
        if resp == "n":
            print "aborted"
            exit()
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    # Data acquisition

    f = run(rate = args.rate*1e6, freq = rf_freq, front_end = args.frontend,
            tones = np.asarray(tones), lapse = calculated_duration, decimation = args.decimation,
             nx = args.nx, ny = args.ny, integ_time = args.integ_time, gain = args.gain)
