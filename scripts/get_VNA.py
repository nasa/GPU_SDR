
import sys,os

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

import argparse

def run(gain,iter,rate,freq,front_end, f0,f1, lapse, points):

    try:
        if u.LINE_DELAY[str(int(rate/1e6))]: pass
    except KeyError:
        print "Cannot find line delay. Measuring line delay before VNA:"

        filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True)

        delay = u.analyze_line_delay(filename, True)

        u.write_delay_to_file(filename, delay)

        u.load_delay_from_file(filename)

    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = gain, Rate=None, decimation=True, RF=freq, Front_end=None,
               Device=None, output_filename=None, Multitone_compensation=None, Iterations=iter, verbose=False)

    return vna_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--f0', '-f0', help='Baseband start frequrency in MHz', type=float, default=-45)
    parser.add_argument('--f1', '-f1', help='Baseband end frequrency in MHz', type=float, default=+45)
    parser.add_argument('--points', '-p', help='Number of points used in the scan', type=float, default=50e3)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds per iteration', type=float, default=10)
    parser.add_argument('--iter', '-i', help='How many iterations to perform', type=float, default=1)
    parser.add_argument('--gain', '-g', help='set the transmission gain', type=float, default=1)

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    rr = raw_input("press to start")

    # Data acquisition

    cmd = ""
    filenames = []
    while cmd != "S":
        f = run(
                gain = int(args.gain),
                iter = int(args.iter),
                rate = args.rate*1e6,
                freq = args.freq*1e6,
                front_end = args.frontend,
                f0 = args.f0*1e6,
                f1 = args.f1*1e6,
                lapse = args.time,
                points = args.points
            )
        cmd = raw_input("Press enter to measure again or type S to stop.")
        filenames.append(f)

    # Data analysis and plotting will be in an other python script
