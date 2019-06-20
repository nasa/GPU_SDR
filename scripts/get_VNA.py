
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

def run(gain,iter,rate,freq,front_end, f0,f1, lapse, points, ntones, delay_duration, delay_over):

    try:
        if u.LINE_DELAY[str(int(rate/1e6))]: pass
    except KeyError:

        if delay_over is None:
            print "Cannot find line delay. Measuring line delay before VNA:"

            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = delay_duration)

            delay = u.analyze_line_delay(filename, True)

            u.write_delay_to_file(filename, delay)

            u.load_delay_from_file(filename)

        else:

            u.set_line_delay(rate, delay_over)

        if ntones ==1:
            ntones = None

    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = gain, Rate=rate, decimation=True, RF=freq, Front_end=front_end,
               Device=None, output_filename=None, Multitone_compensation=ntones, Iterations=iter, verbose=False)

    return vna_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default 300 MHz)', nargs='+')
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--f0', '-f0', help='Baseband start frequrency in MHz', type=float, default=-45)
    parser.add_argument('--f1', '-f1', help='Baseband end frequrency in MHz', type=float, default=+45)
    parser.add_argument('--points', '-p', help='Number of points used in the scan', type=float, default=50e3)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds per iteration', type=float, default=10)
    parser.add_argument('--iter', '-i', help='How many iterations to perform', type=float, default=1)
    parser.add_argument('--gain', '-g', help='set the transmission gain. Multiple gains will result in multiple scans (per frequency). Default 0 dB',  nargs='+')
    parser.add_argument('--tones', '-tones', help='expected number of resonators',  type=int)
    parser.add_argument('--delay_duration', '-dd', help='Duration of the delay measurement',  type=float, default=0.01)
    parser.add_argument('--delay_over', '-do', help='Manually set line delay in nanoseconds. Skip the line delay measure.',  type=float)

    args = parser.parse_args()

    if args.tones is None:
        ntones = 1
    else:
        ntones = args.tones

    if args.freq is None:
        frequencies = [300,]
    else:
        frequencies = [float(a) for a in args.freq]

    if args.gain is None:
        gains = [0,]
    else:
        gains = [int(float(a)) for a in args.gain]

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    if np.abs(args.f0)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f0,args.rate))
        f0 = args.rate/2 * (np.abs(args.f0)/args.f0)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(f0))
    else:
        f0 = args.f0


    if np.abs(args.f1)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f1,args.rate))
        f1 = args.rate/2 * (np.abs(args.f1)/args.f1)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(f1))
    else:
        f1 = args.f1
    # Data acquisition
    for g in gains:
        for f in frequencies:
            x = run(
                    gain = g,
                    iter = int(args.iter),
                    rate = args.rate*1e6,
                    freq = f*1e6,
                    front_end = args.frontend,
                    f0 = f0*1e6,
                    f1 = f1*1e6,
                    lapse = args.time,
                    points = args.points,
                    ntones = ntones,
                    delay_duration = args.delay_duration,
                    delay_over = args.delay_over
                )

    u.Disconnect()
    # Data analysis and plotting will be in an other python script
