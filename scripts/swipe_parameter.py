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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--gain', '-g', help='list of TX gains, default is 0', nargs='+')
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--guard_tones','-gt', nargs='+', help='Add guard Tones in MHz (offset from freq) as a list i.e. -T 1 2 3')
    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--time', '-t', help='Duration of the noise acquisition in seconds', type=float, default=10)
    parser.add_argument('--VNA', '-vna', help='Initial VNA, resonators will be initialized with the parameter of this file (has to be fitted)', type=str, required = True)
    parser.add_argument('--seed_init', '-si', help='Initialize every fit with the seed VNA instead of using last VNA', action="store_true")
    parser.add_argument('--peak_width', '-w', help='Frequency span for fitting.', type=float, default=80e3)
    parser.add_argument('--mode', '-m', help='Noise acquisition kernels. DIRECT uses direct demodulation PFB use the polyphase filter bank technique.', type=str, default= "DIRECT")
    parser.add_argument('--trigger', '-tr', help='String describing the trigger to use. Default is no trigger. Use the name of the trigger classes defined in the trigger module with no parentesis', type=str)



    args = parser.parse_args()
    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if args.gain is None:
        gains = [0,]
    else:
        gains = [int(float(a)) for a in args.gain]

    if args.guard_tones is None:
        guard_tones = []
    else:
        guard_tones = np.asarray([int(float(a)*1e6) for a in args.guard_tones])

    if args.frontend == 'A':
        ant = "A_RX2"
    elif args.frontend == 'B':
        ant = "B_RX2"
    else:
        err_msg = "Frontend %s unknown" % args.frontend
        u.print_warning(err_msg)
        ant = None

    if trigger is not None:
        try:
            trigger = eval('u.'+trigger+'()')
        except SyntaxError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""
        except AttributeError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""

    #replicate the VNA measure (WARNING: DOES NOT SUPPORT ITERATIONS)
    u.print_debug("Replicating seed VNA measure to ensure phase coherency...")
    VNA_seed_info = u.get_rx_info(args.VNA, ant=ant)
    seed_rf, seed_tones = u.get_tones(args.VNA)
    seed_start_f = VNA_seed_info['freq'][0]
    seed_end_f = VNA_seed_info['chirp_f'][0]
    seed_measure_t = VNA_seed_info['chirp_t'][0]
    seed_points = VNA_seed_info['swipe_s'][0]
    seed_gain = VNA_seed_info['gain']
    seed_rate = VNA_seed_info['rate']
    seed_ntones = len(seed_tones) + len(guard_tones)
    u.print_debug("Adjusting power for %d tone readout..."%seed_ntones)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    #measure line delay

    filename = u.measure_line_delay(seed_rate, seed_rf, str(args.frontend), USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = 0.1)
    delay = u.analyze_line_delay(filename, True)
    u.write_delay_to_file(filename, delay)
    u.load_delay_from_file(filename)

    vna_seed_filename = u.Single_VNA(
        start_f = seed_start_f,
        last_f = seed_end_f,
        measure_t = seed_measure_t,
        n_points = seed_points,
        tx_gain = seed_gain,
        Rate=seed_rate,
        decimation=True,
        RF=seed_rf,
        Front_end=str(args.frontend),
        Device=None,
        output_filename=None,
        Multitone_compensation=seed_ntones,
        Iterations=1,
        verbose=False
    )
    u.VNA_analysis(vna_seed_filename)
    u.initialize_from_VNA(args.VNA, vna_seed_filename)
    u.vna_fit(vna_seed_filename, p0=None, fit_range = args.peak_width, verbose = False)
    u.plot_VNA(vna_seed_filename, backend = "plotly", plot_decim = None)
    u.plot_resonators(vna_seed_filename, reso_freq = None, backend = 'plotly')


    #start swiping the gain parameter.
    # Add here other for loops on different params and possibly create/change folders
    #ack = raw_input("Press any key to continue (plotting of the first VNA should be in the browser)")
    for i in range(len(gains)):
        vna_filename = u.Single_VNA(
            start_f = seed_start_f,
            last_f = seed_end_f,
            measure_t = seed_measure_t,
            n_points = seed_points,

            tx_gain = gains[i],

            Rate=seed_rate,
            decimation=True,
            RF=seed_rf,
            Front_end=args.frontend,
            Device=None,
            output_filename=None,
            Multitone_compensation=seed_ntones,
            Iterations=1,
            verbose=False
        )
        u.VNA_analysis(vna_filename)
        #initialize resonators from last VNA scan or from seed
        if args.seed_init or (i == 0):
            u.initialize_from_VNA(vna_seed_filename, vna_filename)
        else:
            u.initialize_from_VNA(last_vna_filename, vna_filename)

        # WARNING: If folder is changed this line has to change accordingly!
        last_vna_filename = vna_filename

        #fit resonators
        u.vna_fit(vna_filename, p0=None, fit_range = args.peak_width, verbose = False)
        #u.plot_VNA(vna_filename, backend = "plotly", plot_decim = None)
        #u.plot_resonators(vna_filename, reso_freq = None, backend = 'plotly')

        #gather tones and acquire noise
        rf_freq, tones = u.get_tones(vna_filename)
        tones = np.asarray(tones)
        tones = np.concatenate((tones,guard_tones))

        noise_filename = u.Get_noise(
            tones,
            measure_t = args.time,
            rate = seed_rate,
            decimation = args.decimation,
            amplitudes = None,
            RF = seed_rf,
            output_filename = None,
            Front_end = args.frontend,
            Device = None,
            delay = None,
            pf_average = 4,
            tx_gain = gains[i],
            mode = args.mode,
            trigger = trigger
        )

        #copy the resonator group
        u.copy_resonator_group(vna_filename, noise_filename)

        u.diagnostic_VNA_noise(noise_filename, noise_points = None, VNA_file = None, ant = ant, backend = 'matplotlib')
