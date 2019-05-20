
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--freq_A', '-fa', help='Frontend A) LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--freq_B', '-fb', help='Frontend B) LO frequency in MHz', type=float, default= 300)

    parser.add_argument('--gain_A', '-ga', help='Frontend A) TX noise', type=int, default= 0)
    parser.add_argument('--gain_B', '-gb', help='Frontend B) TX noise', type=int, default= 0)
    parser.add_argument('--power_target', '-ppt', help='Instead of setting the gains and amplitude of each tone,define the power per tone in dbm; the gain specified in other argument becomes additional and can be negative', type=float)


    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)

    parser.add_argument('--tones_A','-Ta', nargs='+', help='Frontend A) Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--tones_B','-Tb', nargs='+', help='Frontend B) Tones in MHz as a list i.e. -T 1 2 3')

    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=10)
    parser.add_argument('--VNA', '-vna', help='VNA file containing the resonators. Relative to the specified folder above.', type=str)


    args = parser.parse_args()
    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if args.VNA is not None:
        rf_freq_A, tones_A = u.get_tones(args.VNA,frontends = 'A_RX2')
        u.print_debug("getting %d tones from %s frontend A" % (len(tones_A),args.VNA))
        rf_freq_B, tones_B = u.get_tones(args.VNA,frontends = 'B_RX2')
        u.print_debug("getting %d tones from %s frontend B" % (len(tones_B),args.VNA))

    else:
        try:
            tones_A = [float(x) for x in args.tones_A]
            tones_A = np.asarray(tones_A)*1e6
            tones_B = [float(x) for x in args.tones_B]
            tones_B = np.asarray(tones_B)*1e6
        except ValueError:
            u.print_error("Cannot convert tone arfreqgument.")

        rf_freq_A = args.freq_A*1e6
        rf_freq_B = args.freq_A*1e6



    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    if args.power_target is not None:
        ppt_A = - 20* np.log10(len(tones_A))
        ppt_B = - 20* np.log10(len(tones_B))
        u.print_debug("Power splitting per tone:")
        u.print_debug("Frontend A:\t%.2f dB"%ppt_A)
        u.print_debug("Frontend B:\t%.2f dB"%ppt_B)
        u.print_debug("Target power per tone is: %.2f dBm"%args.power_target)
        gain_A = int(args.power_target + 6 - ppt_A)
        gain_B = int(args.power_target + 6 - ppt_B)
        u.print_debug("A->%ddb + args.gain_A (%d db)"%(int(gain_A),args.gain_A))
        u.print_debug("B->%ddb + args.gain_B (%d db)"%(int(gain_B),args.gain_B))
        gain_A+=args.gain_A
        gain_B+=args.gain_B


    filename = u.dual_get_noise(
        tones_A = tones_A,
        tones_B = tones_B,
        measure_t = args.time,
        rate = args.rate*1e6,
        decimation = args.decimation,
        amplitudes_A = None,
        amplitudes_B = None,
        RF_A = rf_freq_A,
        RF_B = rf_freq_B,
        tx_gain_A = args.gain_A,
        tx_gain_B = args.gain_B,
        output_filename = None,
        Device = None,
        delay = None,
        pf_average = 4
    )
