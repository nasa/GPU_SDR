'''
This program will fit a VNA scans and plot the results.
'''

import sys,os,glob

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        raise ImportError("Cannot find the pyUSRP package")

import argparse

def run(file_list, backend, attenuation, N_peaks, smoothing, a_cutoff, threshold, peak_width, Mag_depth_cutoff):
    for i in range(len(file_list)):
        if threshold is not None:
            u.extimate_peak_number(file_list[i], threshold = threshold, smoothing = smoothing, peak_width = peak_width, verbose = False, exclude_center = True, diagnostic_plots = True)
        else:
            u.initialize_peaks(file_list[i], N_peaks = N_peaks[i], a_cutoff = a_cutoff, smoothing = smoothing, peak_width = peak_width, Qr_cutoff=4e3, verbose = True, exclude_center = True, diagnostic_plots = True,  Mag_depth_cutoff = Mag_depth_cutoff)
        u.vna_fit(file_list[i], p0=None, fit_range = peak_width, verbose = False)

        # all resonator plotted on a single static png can be overwelming
        if backend == 'matplotlib':
            single_plots = True
        else:
            single_plots = False
        u.plot_resonators(file_list[i], reso_freq = None, backend = 'plotly', title_info = None, verbose = False, output_filename = None, auto_open = True, attenuation = None,single_plots = single_plots)
        u.plot_VNA(file_list[i], backend = 'plotly')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--att', '-a', help='Attenuation?', type=float, default= None)
    parser.add_argument('--ac', '-ac', help='Asymmetry cutoff', type=float, default= 10)
    parser.add_argument('--smoothing', '-s', help='deciamtion factor', type=int, default= None)
    parser.add_argument('--N_peaks', '-p', help='List of numbers containing the number of peaks expected in each VNA in the form -p 1 2 3', nargs='+')
    parser.add_argument('--threshold', '-t', help='threshold for peakfinder algotyrhm. If this argument is given there are no contraint on the number of peaks and the N_peaks argument will be ignored', type=float, default= None)
    parser.add_argument('--peak_width', '-w', help='minimum peak distance and fit init range in Hz. Default behaviour is different if the number of peaks is contrained of if a threshold is provided', type=float, default= 20e3)
    parser.add_argument('--mag', '-m', help='Magnitude cut-off', type=float, default= 1)
    args = parser.parse_args()
    os.chdir(args.folder)
    peaks = []
    files = glob.glob("USRP_VNA*.h5")
    if args.threshold is None and args.N_peaks is None:
        u.print_error("Provide number of peaks or threshold.")
    if args.N_peaks is not None:
        if len(files)!=len(args.N_peaks):
            err_msg = "number of peaks excpected different from number of files found"
            u.print_error(err_msg)
            raise ValueError(err_msg)

        print("Fitting vna scan from files:")
        for i in range(len(files)):
            p = int(args.N_peaks[i])
            peaks.append(p)
            print "\'%s\' expected peaks: %d" % (files[i], p)

    run(files, args.backend, args.att, peaks,smoothing = args.smoothing, a_cutoff = args.ac, threshold = args.threshold, peak_width = args.peak_width, Mag_depth_cutoff = args.mag)
