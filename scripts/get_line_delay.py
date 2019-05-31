
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

def run(rate,freq,front_end,duration):
    '''
    Basic test of the line delay functionality.
    :param rate: USRP sampling rate.
    :param freq: LO frequncy.
    :param front_end: A or B front end.
    :return: None
    '''

    filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = duration)

    delay = u.analyze_line_delay(filename, True)

    try:
        delay += u.LINE_DELAY[str(int(rate/1e6))]*1e-9
    except KeyError:
        pass

    u.write_delay_to_file(filename, delay)

    u.load_delay_from_file(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic line delay functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--time', '-t', help='Duration of the deley measurement in seconds', type=float, default = 0.01)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    while True:
        run(rate = args.rate*1e6, freq = args.freq*1e6, front_end = args.frontend, duration = args.time)
        raw_input("Press to measure again...")
