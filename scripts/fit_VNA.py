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

def run(file_list, backend, attenuation, N_peaks):
    for i in range(len(file_list)):
        u.initialize_peaks(file_list[i], N_peaks = N_peaks[i], smoothing = 10, peak_width = 90e3, Qr_cutoff=5e3, verbose = True, exclude_center = True, diagnostic_plots = True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--att', '-a', help='Line attenuation in dB', type=float, default= None)
    parser.add_argument('--N_peaks', '-p', help='Listo of numbers containing the number of peaks expected in each VNA in the form -p 1 2 3', nargs='+', required=True)

    args = parser.parse_args()
    os.chdir(args.folder)

    files = glob.glob("USRP_VNA*.h5")

    if len(files)!=len(args.N_peaks):
        err_msg = "number of peaks excpected different from number of files found"
        print_error(err_msg)
        raise ValueError(err_msg)

    print("Fitting vna scan from files:")
    peaks = []
    for i in range(len(files)):
        p = int(args.N_peaks[i])
        peaks.append(p)
        print "\'%s\' expected peaks: %d" % (files[i], p)

    run(files, args.backend, args.att, peaks)
