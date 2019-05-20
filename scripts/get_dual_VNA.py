
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

def run(rate,freq_a,freq_b, f0a,f1a,f0b,f1b, lapse, points, gain_a, gain_b):


    try:
        if u.LINE_DELAY[str(int(rate/1e6))]: pass
    except KeyError:
        print "Cannot find line delay. Measuring line delay before VNA:"

        front_end = 'A'

        filename = u.measure_line_delay(rate, freq_a, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True)

        delay = u.analyze_line_delay(filename, True)

        u.write_delay_to_file(filename, delay)

        u.load_delay_from_file(filename)

        vna_filename = u.Dual_VNA(
            start_f_A = f0a,
            last_f_A = f1a,
            start_f_B = f0b,
            last_f_B = f1b,
            measure_t = lapse,
            n_points = points,
            tx_gain_A = gain_a,
            tx_gain_B = gain_b,
            Rate = None,
            decimation = True,
            RF_A = freq_a,
            RF_B = freq_b,
            Device = None,
            output_filename = None,
            Multitone_compensation_A = None,
            Multitone_compensation_B = None,
            Iterations = 1,
            verbose = False
        )

    return vna_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq_a', '-fa', help='LO frequency in MHz (frontend A)', type=float, default= 300)
    parser.add_argument('--freq_b', '-fb', help='LO frequency in MHz (frontend B)', type=float, default= 390)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--f0a', '-f0a', help='Baseband start frequrency in MHz (frontend A)', type=float, default=-45)
    parser.add_argument('--f1a', '-f1a', help='Baseband end frequrency in MHz (frontend A)', type=float, default=+45)
    parser.add_argument('--f0b', '-f0b', help='Baseband start frequrency in MHz (frontend B)', type=float, default=-45)
    parser.add_argument('--f1b', '-f1b', help='Baseband end frequrency in MHz (frontend B)', type=float, default=+45)
    parser.add_argument('--points', '-p', help='Number of points used in the scan', type=float, default=50e3)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=10)
    parser.add_argument('--gain_a', '-ga', help='Gain for frontend A', type=int, default=0)
    parser.add_argument('--gain_b', '-gb', help='Gain for frontend B', type=int, default=0)



    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    # Data acquisition
    f = run(rate = args.rate*1e6,
        freq_a = args.freq_a*1e6,
        f0a = args.f0a*1e6,
        f1a = args.f1a*1e6,
        freq_b = args.freq_b*1e6,
        f0b = args.f0b*1e6,
        f1b = args.f1b*1e6,
        lapse = args.time,
        points = args.points,
        gain_a = args.gain_a,
        gain_b = args.gain_b
    )

    # Data analysis and plotting will be in an other python script
