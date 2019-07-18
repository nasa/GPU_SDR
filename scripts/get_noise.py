
import sys,os,random
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

def run(rate,freq,front_end, tones, lapse, decimation, gain, vna, mode, pf, trigger):

    if trigger is not None:
        try:
            trigger = eval('u.'+trigger+'()')
        except SyntaxError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""
        except AttributeError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""


    trigger = u.trigger_template(rate = rate/decimation)
    noise_filename = u.Get_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = None,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = 0,
                              pf_average = pf, tx_gain = gain, mode = mode, trigger = trigger)
    if vna is not None:
        u.copy_resonator_group(vna, noise_filename)

    return noise_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--gain', '-g', help='TX noise', type=int, default= 0)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--tones','-T', nargs='+', help='Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--guard_tones','-gt', nargs='+', help='Add guard Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=10)
    parser.add_argument('--pf', '-pf', help='Polyphase averaging factor for PF mode and taps multiplier for DIRECT mode FIR filter', type=int, default=4)
    parser.add_argument('--VNA', '-vna', help='VNA file containing the resonators. Relative to the specified folder above.', type=str)
    parser.add_argument('--mode', '-m', help='Noise acquisition kernels. DIRECT uses direct demodulation PFB use the polyphase filter bank technique.', type=str, default= "DIRECT")
    parser.add_argument('--random', '-R', help='Generate random tones for benchmark and test reason', type=int)
    parser.add_argument('--trigger', '-tr', help='String describing the trigger to use. Default is no trigger. Use the name of the trigger classes defined in the trigger module with no parentesis', type=str)

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
        if args.random is not None:
            tones = [random.randint(-args.rate/2.,-args.rate/2.) for c in range(args.random)]
        else:
            try:
                if args.tones is not None:
                    tones = [float(x) for x in args.tones]
                    tones = np.asarray(tones)*1e6
                else:
                    tones = []
            except ValueError:
                u.print_error("Cannot convert tone argument.")

    rf_freq = args.freq*1e6

    if args.random is not None:
        tones = [random.uniform(-args.rate*1e6/2, args.rate*1e6/2) for ui in range(args.random)]

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    if args.guard_tones is not None:
        guard_tones = [float(x) for x in args.guard_tones]
        guard_tones = np.asarray(guard_tones)*1e6
        # it's important that guard tones are at the end of the tone array !!!
        tones = np.concatenate((tones,guard_tones))

    # Data acquisition

    f = run(rate = args.rate*1e6, freq = rf_freq, front_end = args.frontend,
            tones = np.asarray(tones), lapse = args.time, decimation = args.decimation,
            gain = args.gain, vna= args.VNA, mode = args.mode, pf = args.pf, trigger = args.trigger)

    # Data analysis and plotting will be in an other python script
