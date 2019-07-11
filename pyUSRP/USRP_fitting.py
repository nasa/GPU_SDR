########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, lfilter, freqz
import signal as Signal
import h5py
import sys
import struct
import json
import os
import socket
import Queue
from Queue import Empty
from threading import Thread, Condition
import multiprocessing
from joblib import Parallel, delayed
from subprocess import call
import time
import gc
import datetime

# plotly stuff
from plotly.graph_objs import Scatter, Layout
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl

# matplotlib stuff
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

# needed to print the data acquisition process
import progressbar

# import submodules
from USRP_low_level import *
from USRP_files import *
from scipy import optimize
from USRP_plotting import *
from USRP_VNA import linear_phase


def real_of_complex(z):
    '''
    Flatten n-dim complex vector to 2n-dim real vector for fitting.
    :param z: array of complex numbers.
    :return: an array composed by real and imaginary part of the number.
    '''
    r = np.hstack((z.real, z.imag))
    return r


def complex_of_real(r):
    '''
    Does the inverse of real_of_complex() function.
    :param r: real + imaginary data
    :return: array of complex numbers
    '''
    assert len(r.shape) == 1
    nt = r.size
    assert nt % 2 == 0
    no = nt / 2
    z = r[:no] + 1j * r[no:]
    return z

def nonlinear_model(f, f0, A, phi, D, dQr, dQe_re, dQe_im, a):
    '''
    Non-linear model for fitting resonators developed by Albert and Bryan.

    :param f: Array containing frequency in MHz.
    :param f0: Resonant frequency in MHz.
    :param A: Amplitude of the resonator circle
    :param phi: phase of the resonator center
    :param D: Line delay of the line in ns?
    :param dQr: Inverse of Qr.
    :param dQe_re: Inverse of the real part of coupling quality factor.
    :param dQe_im: Inverse of the imaginary part of the coupling quality factor.
    :param a: Non-linear parameter.
    :return:
    '''
    f0 = f0 * 1e6
    cable_phase = np.exp(2.j * np.pi * (1e-6 * D * (f - f0) + phi))
    dQe = dQe_re + 1.j * dQe_im

    x0 = (f - f0) / f0
    y0 = x0 / dQr
    k2 = np.sqrt((y0 ** 3 / 27. + y0 / 12. + a / 8.) ** 2 - (y0 ** 2 / 9. - 1 / 12.) ** 3, dtype=np.complex128)
    k1 = np.power(a / 8. + y0 / 12. + k2 + y0 ** 3 / 27., 1. / 3)
    eps = (-1. + 3 ** 0.5 * 1j) / 2.

    y1 = y0 / 3. + (y0 ** 2 / 9. - 1 / 12.) / k1 + k1
    y2 = y0 / 3. + (y0 ** 2 / 9. - 1 / 12.) / eps / k1 + eps * k1
    y3 = y0 / 3. + (y0 ** 2 / 9. - 1 / 12.) / eps ** 2 / k1 + eps ** 2 * k1

    y1[np.abs(k1) == 0.0] = y0[np.abs(k1) == 0.0] / 3.
    y2[np.abs(k1) == 0.0] = y0[np.abs(k1) == 0.0] / 3.
    y3[np.abs(k1) == 0.0] = y0[np.abs(k1) == 0.0] / 3.

    # Out of the three roots we need to pick the right branch of the bifurcation
    thresh = 1e-4
    low_to_high = np.all(np.diff(f) > 0)
    if low_to_high:
        y = y2.real
        mask = (np.abs(y2.imag) >= thresh)
        y[mask] = y1.real[mask]
    else:
        y = y1.real
        mask = (np.abs(y1.imag) >= thresh)
        y[mask] = y2.real[mask]

    x = y * dQr

    s21 = A * cable_phase * (1. - (dQe) / (dQr + 2.j * x))

    return real_of_complex(s21)

def S21_func(f, f0, A, phi, D, dQr, dQe_re, dQe_im, a):
    '''
    Given a frequency range (f) and the resonator paramters return the S21 complex function.
    d<param name> is intended as 1./<param name>
    '''
    return complex_of_real(nonlinear_model(f, f0, A, phi, D, dQr, dQe_re, dQe_im, a))

def FWMH(freq, magnitude):
    magnitude = np.abs(magnitude)
    min_point = freq[np.argmax(magnitude)]
    MH = (np.max(magnitude) - np.mean([magnitude[0], magnitude[-1]])) / 2.
    sel_freq = freq[magnitude > MH]
    return np.abs(min(sel_freq) - max(sel_freq))


def do_fit(freq, re, im, p0=None):
    '''
    Function internally used to fit the resonators.
    This is not the function to call to fit a VNA scan, to do that, try vna_fit().
    Notes:
    * f0 in p0 is in MHz
    '''
    model = nonlinear_model
    nt = len(freq)
    mag = np.sqrt(re * re + im * im)
    phase = np.unwrap(np.arctan2(im, re))
    # initialization helper
    # phase_,m,q = good_phase(phase,freq,True)
    i_m = np.mean([im[0], im[-1]])
    r_m = np.mean([re[0], re[-1]])
    p_m = np.arctan2(i_m, r_m)
    if p0 == None:
        f0 = freq[np.argmin(mag)] / 1.e6
        scale = np.max(mag)
        phi = p_m / (2 * np.pi)  # q/(2*np.pi)
        A = scale  # *np.cos(phi)
        B = scale * np.sin(phi)
        D = 0  # m/(2.*np.pi)

        fwmh = FWMH(freq, phase) / 1e6
        Qr = 10 * f0 / fwmh
        Qe_re = Qr * 2
        Qe_im = 0
        dQe = 1. / (1.j * Qe_im + Qe_re)
        a = 0.0
        p0 = (f0, A, phi, D, 1. / Qr, dQe.real, dQe.imag, a)

    ydata = np.hstack((re, im))

    popt, pcov = optimize.curve_fit(model, freq, ydata, p0=p0)  # ,bounds = (0,np.inf)

    f0, A, phi, D, dQr, dQe_re, dQe_im, a = popt

    yfit = model(freq, *popt)
    zfit = complex_of_real(yfit)

    zm = re + 1.j * im
    resid = zfit - zm
    Qr = 1 / dQr
    Qi = 1.0 / (dQr - dQe_re)

    dQe = dQe_re + 1.j * dQe_im
    Qe = 1. / dQe

    modelwise = (f0, A, phi, D, Qi, Qr, Qe.real, Qe.imag, a)

    return f0, Qi, Qr, zfit, modelwise

import peakutils

def extimate_peak_number(filename, threshold = 0.2, smoothing = None, peak_width = 200e3, verbose = False, exclude_center = True, diagnostic_plots = False):
    """
    This function uses peakutils module to initialize the peaks i the file.
    Stores in the H5 file the result. This does not count as a fit as only the initialization is stored.

    Arguments:
        - Filename: filename of the h5 vna file.
        - threshold: a number between 0 an 1 determaning the peak finding. The threshold is applied to the absolute value of the derivative of s21.
        - smoothing: if the vna is too noise one may consider decimation and fir filtering. dmoothing is the decimation factor.
        - peak_width: minimum distance between each peak.
        - verbose: output some diagnostic information.
        - exclude_center: exclude the center (DC) from the fitting.
        - diagnostic_plots: creates a folder with diagnostic png plots.

    :return: the number of extimated peaks.
    """

    filename = format_filename(filename)
    print("Looking for peaks in file: \'%s\' ..."%filename)

    if verbose:
        print_debug("Peaks finder algorithm based on peakutil module. Search parameters:")
        print_debug( "\tResonator minimum width: %.2f kHz"%(peak_width/1e3))
        print_debug( "\tPeaks threshold: %.2f"%threshold)

    info = get_rx_info(filename, ant="A_RX2")
    info_B = get_rx_info(filename, ant="B_RX2")

    if (info_B['mode'] == 'RX') and (info['mode'] == 'RX'):
        center = [info['rf'], info_B['rf']]
        single_frontend = False
        resolution_A = np.abs(info['freq'][0] - info['chirp_f'][0])/float(info['swipe_s'][0])
        resolution_B = np.abs(info_B['freq'][0] - info_B['chirp_f'][0])/float(info_B['swipe_s'][0])
        if resolution_A!=resolution_B:
            print_warning("Resolution between frontends is different. The shallower resolution will be displayed.")
            resolution = max(resolution_A,resolution_B)
        else:
            resolution = resolution_A
    elif (info_B['mode'] == 'RX') and not (info['mode'] == 'RX'):
        center = info_B['rf']
        resolution = np.abs(info_B['freq'][0] - info_B['chirp_f'][0])/float(info_B['swipe_s'][0])
        single_frontend = True
    elif not (info_B['mode'] == 'RX') and  (info['mode'] == 'RX'):
        center = info['rf']
        resolution = np.abs(info['freq'][0] - info['chirp_f'][0])/float(info['swipe_s'][0])
        single_frontend = True

    freq, S21 = get_VNA_data(filename, calibrated = True, usrp_number = 0)

    phase = np.angle(S21)
    magnitude = np.abs(S21)
    magnitudedb = vrms2dbm(magnitude)


    # Remove the AA filter for usrp x300 ubx-160 100 Msps
    arbitrary_cut = int(len(magnitudedb)/95)
    freq=freq[arbitrary_cut:-arbitrary_cut]
    phase=phase[arbitrary_cut:-arbitrary_cut]
    magnitudedb=magnitudedb[arbitrary_cut:-arbitrary_cut]
    magnitude = magnitude[arbitrary_cut:-arbitrary_cut]
    phase = linear_phase(phase)


    if smoothing is not None:
        if verbose:
            print_debug( "Decimating signal before looking for peaks...")
        smoothing = int(smoothing)
        freq = signal.decimate(freq,smoothing,ftype="fir")[20:-20]
        magnitudedb = signal.decimate(magnitudedb,smoothing,ftype="fir")[20:-20]
        phase = signal.decimate(phase,smoothing,ftype="fir")[20:-20]
        magnitude = signal.decimate(magnitude,smoothing,ftype="fir")[20:-20]
        resolution *= smoothing

    S21_val = np.exp(1.j*phase)*magnitude

    if diagnostic_plots:
        diagnostic_folder = "Init_peaks_diagnostic_"+os.path.splitext(filename)[0]
        try:
            os.mkdir(diagnostic_folder)
        except OSError:
            print_warning("Overwriting initialization diagnostic plots")
        print_debug("Generating diagnostic plots in folder \'%s\'..."%diagnostic_folder)

    #supposed width of each peak in index unit
    peak_width /= resolution
    peak_width = int(peak_width)
    max_diag = []
    q_diag = []
    f0s = []

    # Optimizing on the magnitude of derivative of S21
    gradS21 = np.gradient(S21_val)

    #zero the negative part
    gradS21 = np.asarray([np.abs(vv) if np.abs(vv) > 0 else 0 for vv in gradS21])

    try:
        # exclude conjunction point
        if len(center)>1:
            f_prof = np.gradient(freq)
            freq_point = np.argmax(np.abs(f_prof - np.mean(f_prof)))
            fp_min = max(0,freq_point-10)
            fp_max = min(freq_point+10,len(gradS21))
            gradS21 = np.delete(gradS21,range(fp_min,fp_max))
            freq = np.delete(freq,range(fp_min,fp_max))
    except:
        center = [center,]

    mask = np.zeros(len(freq), dtype=bool)

    # We could use there three to determine if there are effectively resonators
    # the pekutils also peaks up noise from the flat S21.
    #print max(gradS21)
    #print np.std(gradS21)
    #print np.mean(gradS21)



    # exclude center frequency
    center_excl = [[] for X in range(len(center))]
    center_min = [[] for X in range(len(center))]
    center_max = [[] for X in range(len(center))]
    print_debug("Using resolution of %.1f Hz" % resolution)
    for j in range(len(center)):
        if exclude_center:
            for ii in range(len(mask)):
                if np.abs(freq[ii] - center[j]) < (int(500000e3/resolution)):
                    mask[ii] = False
                    gradS21[ii] = gradS21[ii-1]
                    center_excl[j].append(ii)
                    print "excluding: %.2f"%(freq[ii]/1e6)
        if len(center_excl[j])>1:
            center_min[j] = min(center_excl[j])
            center_max[j] = max(center_excl[j])
        else:
            center_min[j] = None
            center_max[j] = None

    if(diagnostic_plots):fig, ax = pl.subplots()

    indexes = peakutils.indexes(gradS21, thres=threshold, min_dist=peak_width)
    for ii in range(len(freq)):
        if ii in indexes:
            mask[ii] = True


    max_diag = freq[mask]
    if(diagnostic_plots):
        print_debug("Plotting diagostic...")

        ax.plot(gradS21, label = "grad")
        ax.scatter(np.arange(len(gradS21))[mask],gradS21[mask], color = 'r', label = "%d peaks"%len(max_diag))
        for k in range(len(center_min)):
            if center_min[k] is not None:
                if k == 0:
                    ax.axvspan(center_min[k], center_max[k], alpha=0.4, color='red',label = "center excl")
                else:
                    ax.axvspan(center_min[k], center_max[k], alpha=0.4, color='red')
        ax.set_xlabel("Index of saved $S_{21}$")
        ax.set_ylabel(r"|$ \frac{ \partial S_{21} }{ \partial i }|$ ")
        pl.legend()
        pl.grid()
        pl.savefig("diagnostic_peaks_init_%s.png"%filename.split('.')[0])


    # Write stuff on file
    if len(max_diag)>0:
        fv = h5py.File(filename,'r+')

        try:
            reso_grp = fv.create_group("Resonators")
        except ValueError:
            print_warning("Overwriting resonator initialization attribute")
            del fv["Resonators"]
            reso_grp = fv.create_group("Resonators")

        reso_grp.attrs.__setitem__("tones_init", max_diag)

        fv.close()

    print("Initialize_peaks() found " +str(len(max_diag))+ " resonators.")


def initialize_peaks(filename, N_peaks = 1, smoothing = None, peak_width = 90e3, Qr_cutoff=5e3, a_cutoff = 10, Mag_depth_cutoff = 0.15, verbose = False, exclude_center = True, diagnostic_plots = False):
    """
    This function uses a filter on quality factor estimated using the nonlinear resonator model. This function considers the resonator around the maximum of the unwraped phase trace, tries to fit the resonator and make a decision if it's a resonator or not by parsing the quality factor of the fit with the Qt_vutoff argument. Before iterating excludes a zone of peak_width Hz around the previously considered point.
    Stores in the H5 file the result. This does not count as a fit as only the initialization is stored.

    Arguments:
        - Filename: filename of the h5 vna file.
        - smoothing: if the vna is too noise one may consider decimation and fir filtering. dmoothing is the decimation factor.
        - peak_width: minimum distance between each peak.
        - Qr_cutoff: thrashold above wich a fit will result in a resonaton being stored.
        - a_cutoff: cutoff on asymmetry (if the fit returns a>a_cutoff is discarded)
        - Mag_depth_cutoff: cutoff on magnitude depth of the resonance in dB.
        - N_peaks: how may peaks to expect. If this number is bigger than the actual number of resonator, the search process will end after all the frequency chunks are masked.
        - verbose: output some diagnostic information.
        - exclude_center: exclude the center (DC) from the fitting.
        - diagnostic_plots: creates a folder with diagnostic png plots.
    Returns:
        - boolean: False if the number of requested peaks does not corresond to the number of peaks found.
    """

    filename = format_filename(filename)
    print("Inintializing peaks in file: \'%s\' ..."%filename)

    if verbose:
        print_debug("Peaks finder algorithm based on non-linear resonator model is being used. Search parameters:")
        print_debug( "\tResonator minimum width: %.2f kHz"%(peak_width/1e3))
        print_debug( "\tResonator minimum Qr: %.2fk"%(Qr_cutoff/1e3))
        print_debug( "\tExpected peaks: %d"%int(N_peaks))

    info = get_rx_info(filename, ant=None)
    freq, S21 = get_VNA_data(filename, calibrated = True, usrp_number = 0)

    resolution = np.abs(info['freq'][0] - info['chirp_f'][0])/float(len(S21))
    center = info['rf']

    phase = np.angle(S21)
    magnitude = np.abs(S21)
    magnitudedb = vrms2dbm(magnitude)


    # Remove the AA filter for usrp x300 ubx-160 100 Msps
    arbitrary_cut = int(len(magnitudedb)/90)
    freq=freq[arbitrary_cut:-arbitrary_cut]
    phase=phase[arbitrary_cut:-arbitrary_cut]
    magnitudedb=magnitudedb[arbitrary_cut:-arbitrary_cut]
    magnitude = magnitude[arbitrary_cut:-arbitrary_cut]


    if smoothing is not None:
        if verbose:
            print_debug( "Decimating signal before looking for peaks...")
        smoothing = int(smoothing)
        freq = signal.decimate(freq,smoothing,ftype="fir")[20:-20]
        magnitudedb = signal.decimate(magnitudedb,smoothing,ftype="fir")[20:-20]
        phase = signal.decimate(phase,smoothing,ftype="fir")[20:-20]
        magnitude = signal.decimate(magnitude,smoothing,ftype="fir")[20:-20]
        resolution *= smoothing

    S21_val = np.exp(1.j*phase)*magnitude

    if diagnostic_plots:
        diagnostic_folder = "Init_peaks_diagnostic_"+os.path.splitext(filename)[0]
        try:
            os.mkdir(diagnostic_folder)
        except OSError:
            print_warning("Overwriting initialization diagnostic plots")
        print_debug("Generating diagnostic plots in folder \'%s\'..."%diagnostic_folder)

    #supposed width of each peak
    peak_width /= resolution
    peak_width = int(peak_width)

    max_diag = []
    q_diag = []
    f0s = []
    mask = np.ones(len(magnitude), dtype=bool)

    # exclude center frequency
    if exclude_center:
        for ii in range(len(mask)):
            if np.abs(freq[ii] - center) < (50000):
                mask[ii] = False

    # Optimizing on the magnitude of derivative of S21
    gradS21 = np.abs(np.gradient(S21_val))
    freq_ = freq #mock frequency axis
    iteration_number = 0

    #hardcoded maximum quality decimation_factor
    Qr_max = 500e3

    while(sum(mask)>0):

        maximum = np.max(gradS21[mask])
        maximum = np.where(gradS21 == maximum)[0]

        low_index = int(max(maximum-peak_width,0))
        low_index = max(0,low_index)

        high_index = int(min(maximum+peak_width, len(freq_)))
        high_index = min(len(gradS21)-1,high_index)

        #used in peack rejections
        half_low_index = int(max(maximum-peak_width/1.2,0))
        half_high_index = int(min(maximum+peak_width/1.2, len(freq_)-1))


        try:
            #with nostdout():
            f0,Qi,Qr,zfit,modelwise = do_fit(
                freq_[low_index:high_index],
                S21_val.real[low_index:high_index],
                S21_val.imag[low_index:high_index],
                p0=None
            )

            # Asymmetry is usually limited to ~10
            a = modelwise[8]

            #depth filter
            depth = np.abs(min(vrms2dbm(zfit))-max(vrms2dbm(zfit)))

        except RuntimeError:
            Qr = 0
            depth = 0
            a = np.inf

        #####################################
        # CONDITIONS FOR ACCEPTING THE INIT #
        #####################################

        if (Qr>Qr_cutoff) and (Qr<Qr_max) and (f0>freq_[half_low_index]/1e6) and (f0<freq_[half_high_index]/1e6) and (a<a_cutoff) and (depth> Mag_depth_cutoff):
            print_debug("%d) Resonator found at %.2f MHz"%(len(f0s), freq_[maximum]/1.e6))
            max_diag.append(maximum)
            q_diag.append(Qr)
            f0s.append(f0)
            label_set = "Accepted init:\nQr: %.2fk range: [%.2fk - %.2fk]"%(Qr/1e3, Qr_cutoff/1e3, Qr_max/1e3)
            label_set += "\nf0 = %.2f MHz range [%.2f - %.2f] MHz" % (f0,freq_[low_index]/1e6,freq_[high_index]/1e6)
            label_set += "\nAsymmetry: %.2f / %.2f" % (a, a_cutoff)
            label_set += "\nMagnitude depth = %.2f dB / %.2f dB" % (depth, Mag_depth_cutoff)
            col = 'green'
        else:
            label_set = "Refused init:\nQr: %.2fk range: [%.2fk - %.2fk]"%(Qr/1e3, Qr_cutoff/1e3, Qr_max/1e3)
            label_set += "\nf0 = %.2f MHz range [%.2f - %.2f] MHz" % (f0,freq_[low_index]/1e6,freq_[high_index]/1e6)
            label_set += "\nAsymmetry: %.2f / %.2f" % (a, a_cutoff)
            label_set += "\nMagnitude depth = %.2f dB / %.2f dB" % (depth, Mag_depth_cutoff)
            col = 'red'

        if diagnostic_plots:
            os.chdir(diagnostic_folder)
            fig, ax = pl.subplots(nrows=1, ncols=1)
            fig.suptitle("Peak initialization diagnosic #%d"%iteration_number)
            ax.plot(
                freq_[low_index:high_index],
                magnitudedb[low_index:high_index],
                label = label_set,
                color = col
            )
            ax.plot(
                freq_[low_index:high_index],
                vrms2dbm(np.abs(zfit)),
                label = "fit",
                color = "k"
            )
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("S21 Magnitude [dB]")
            ax.grid()
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.savefig("init_diag_%d.png"%iteration_number, bbox_inches="tight")
            pl.close(fig)
            os.chdir("..")

        # If the number of peaks expected is reached, stop
        if len(max_diag) >= N_peaks:
            break

        # Remove points from mask
        #mask = mask[mask]
        for i in range(len(gradS21)):
            if i >maximum-peak_width and i<maximum+peak_width:
                mask[i] = False

        iteration_number+=1


    # Write stuff on file
    if len(max_diag)>0:
        fv = h5py.File(filename,'r+')

        try:
            reso_grp = fv.create_group("Resonators")
        except ValueError:
            print_warning("Overwriting resonator initialization attribute")
            del fv["Resonators"]
            reso_grp = fv.create_group("Resonators")

        results = [freq[j] for j in max_diag]
        reso_grp.attrs.__setitem__("tones_init", results)

        fv.close()

    print("Initialize_peaks() found " +str(len(max_diag))+ " resonators.")

    if N_peaks!=len(results):
        return False
    else:
        return True


def initialize_from_VNA(original_VNA, new_VNA, verbose = False):
    '''
    Initialize the peaks in a new VNA file using fitted parameters from an other VNA file that has already been fitted.
    This function overwrites the Resonato group of the new_VNA file.
    :param original_VNA: name of the VNA file containing the fits.
    :param new_VNA: name of the VNA file to initialize.
    :param verbose: print debug strings.

    '''
    original_VNA = format_filename(original_VNA)
    new_VNA = format_filename(new_VNA)
    original_fits_param = get_fit_param(original_VNA, verbose = verbose)
    if len(original_fits_param)>0:
        fv = h5py.File(new_VNA,'r+')
        try:
            reso_grp = fv.create_group("Resonators")
        except ValueError:
            print_warning("Overwriting resonator initialization attribute")
            del fv["Resonators"]
            reso_grp = fv.create_group("Resonators")

        results = [reso['f0']*1e6 for reso in original_fits_param]
        reso_grp.attrs.__setitem__("tones_init", results)

        fv.close()
    else:
        err_msg = "Cannot find any resonator in the original file! check that the original VNA has been fitted."
        print_error(err_msg)
        raise ValueError(err_msg)

def vna_fit(filename, p0=None, fit_range = 10e4, verbose = False):
    """
    Open a pre analyzed, pre plotted (with tagged resonator inside) .h5 VNA file and fit the resonances in it. Creates a new group in the ".h5" file called "resonators" and save fitted curve and attributes in it.

    Arguments:
    	- filename: string representing the name of the target .h5 file without the ".h5" extension
    	- p0 : initial parameters for the fit. if None (default: None: the function tries to generate initial parameters)
    	- fit_range: half size in Hz of a the interval around the resonator to consider for the fit
        - verbose: print some diagnostic information

    Returns:
    	Returns boolean: True if the number of succesfull fit corresponds to the number of initialized fit, False otherwise.

    Known bugs:
    	- initialization of the parameters is not complete, the quality factor is just guessed as a constant

    Notes:
    	- ported from Bryan's code: subfunctions for effective fit.
    """

    filename = format_filename(filename)

    print("Fitting resonators in file \'%s\' ..."%filename)

    peaks_init = get_init_peaks(filename)
    frequency, S21 = get_VNA_data(filename, calibrated = True, usrp_number = 0)
    info = get_rx_info(filename)
    resolution = np.abs(info['freq'][0] - info['chirp_f'][0])/float(len(S21))
    model = nonlinear_model

    if len(peaks_init) == 0:
        err_msg = "Cannot find any initialized peak"
        print_error(err_msg)
        raise ValueError(err_msg)

    fv = h5py.File(filename,'r+')

    # there is no try catch on this experssion because it's preceded by get_init_peaks()
    reso_grp = fv['Resonators']

    # WARNING: this number is coherent only in a single file: it may be NOT consistent arcoss multiple files!
    fit_number = 0

    overwriting_warning = True

    for tone in peaks_init:
        if verbose: print_debug("Fitting resonator initialized at %.2f MHz ..."%(tone/1e6))

        # Select a range atound the initialized tone.
        selection = np.abs(frequency - tone) < fit_range
        base_fit_re = S21.real[selection]
        base_fit_im = S21.imag[selection]
        base_fit_freq = frequency[selection]

        try:
    		f0,Qi,Qr,zfit,modelwise = do_fit(base_fit_freq,base_fit_re,base_fit_im,p0=p0)
    	except:
    		print_warning("Something went wrong with the fit of resonator at %.2f MHz"%(tone/1e6))
    	else:
            if verbose: print_debug("Fitted succesfully.")
            try:
                single_reso_grp = reso_grp.create_group("reso_%d"%fit_number)
            except ValueError:
                del reso_grp["reso_%d"%fit_number]
                single_reso_grp = reso_grp.create_group("reso_%d"%fit_number)
                if(overwriting_warning):
                    print_warning("Overwriting resonator group")
                    overwriting_warning = False

            # Write fitting data
            single_reso_grp.create_dataset("freq", data = base_fit_freq)
            single_reso_grp.create_dataset("base_S21", data = S21[selection])
            single_reso_grp.create_dataset("fitted_S21", data = zfit)

            # Write fit parameters as attributes
            (f0, A, phi, D, Qi, Qr, Qe_r, Qe_i, a) = modelwise
            Qe = Qe_r + 1.j * Qe_i
            single_reso_grp.attrs.__setitem__("f0", f0)
            single_reso_grp.attrs.__setitem__("A", A)
            single_reso_grp.attrs.__setitem__("phi", phi)
            single_reso_grp.attrs.__setitem__("D", D)
            single_reso_grp.attrs.__setitem__("Qr", Qr)
            single_reso_grp.attrs.__setitem__("Qe", Qe)
            single_reso_grp.attrs.__setitem__("a", a)

            fit_number += 1

    if len(peaks_init)!=fit_number:
        print_warning("%d fit(s) went wrong" % (len(peaks_init) - fit_number))

    print("Resonator fitted")
    fv.close()

    if fit_number!=len(peaks_init):
        return False
    else:
        return True

def get_fit_data(filename, verbose = False):
    '''
    Retrive fit data from a file. For fit data is intended the fitted S21.

    Arguments:
        - filename the name of the h5 file containing the data.
        - verbose: print some debug information.

     Return:
        - List of dictionaries with keys "frequency", "fitted", "original"

    Note:
        - This function does not returns the fit parameters. to do that use get_fit_param().

    '''

    f = bound_open(filename)

    if verbose: print_debug("Getting data data from \'%s\'"%filename)

    try:
        reso_grp = f['Resonators']
    except KeyError:
        err_msg = "Cannot find the resonator group inside the file"
        print_error(err_msg)
        raise ValueError(err_msg)


    ret = []
    resonator_index = 0
    for resonator in reso_grp:
        resonator_group_name = "reso_%d" % resonator_index
        ret.append({
            "frequency":np.asarray(reso_grp[resonator_group_name]['freq']),
            "fitted":np.asarray(reso_grp[resonator_group_name]["fitted_S21"]),
            "original":np.asarray(reso_grp[resonator_group_name]["base_S21"])
            })
        resonator_index += 1

    if verbose: print_debug("Resonator data collected")
    f.close()
    return ret

def get_fit_param(filename, verbose = False):
    '''
    Retrive fit parameters from a file.

    Arguments:
        - filename the name of the h5 file containing the data.
        - verbose: print some debug information.

     Return:
        - List of dictionaries with keys named after parameters. Specifically: f0, A, phi, D, Qi, Qr, Qe, a

    '''
    f = bound_open(filename)

    if verbose: print_debug("Getting fit param from \'%s\'"%filename)

    try:
        reso_grp = f['Resonators']
    except KeyError:
        err_msg = "Cannot find the resonator group inside the file"
        print_error(err_msg)
        raise ValueError(err_msg)

    ret = []
    resonator_index = 0
    for resonator in reso_grp:
        resonator_group_name = "reso_%d" % resonator_index
        ret.append({
            'f0':reso_grp[resonator_group_name].attrs.get("f0"),
            'A':reso_grp[resonator_group_name].attrs.get("A"),
            'phi':reso_grp[resonator_group_name].attrs.get("phi"),
            'D':reso_grp[resonator_group_name].attrs.get("D"),
            'Qi':reso_grp[resonator_group_name].attrs.get("Qi"),
            'Qr':reso_grp[resonator_group_name].attrs.get("Qr"),
            'Qe':reso_grp[resonator_group_name].attrs.get("Qe"),
            'a':reso_grp[resonator_group_name].attrs.get("a")
        })
        resonator_index += 1

    if verbose: print_debug("Resonator parameters collected")
    f.close()
    return ret


def get_best_readout(filename, verbose = False):
    '''
    Get the best readout frequency from a fitted resonator keeping in account the nonlinear firt model.

    Argumets:
        - filename: the h5 file containing the fit data.
        - verbose: print some debug line.

    Return:
        - A list of frequencies (one per each fitted resonator).
    '''

    R = get_fit_param(filename, verbose)

    ret = []
    if verbose: print_debug("Best readout frequency deltas:")
    for resonator in R:
        delta_r = 1./resonator['Qr']
    	brf = 1e6*resonator['f0'] * (1 - resonator['a']*delta_r)
        if verbose: print_debug("Resonator %.2f is shifted %.2fkHz"%(resonator['f0'], 1e3*np.abs(brf/1e6-resonator['f0'])))
        ret.append(brf)

    return ret

def min_readout_spacing(filename, verbose = False):
    '''
    Calculate the minimum spacing between f0s of a fitted VNA file.
    '''

    f0s = get_best_readout(filename, verbose = verbose)
    M = [[np.abs(a-b) if a!=b else np.inf for a in f0s] for b in f0s]
    ret = np.min(M)
    print_debug("Minium channel spacing required is %.2f Hz"%ret)
    return ret


def plot_resonators(filenames, reso_freq = None, backend = 'matplotlib', title_info = None, verbose = False, output_filename = None, auto_open = True, attenuation = None, **kwargs):
    '''
    Plot the resonators and the resonator fits.

    Arguments:
        - filenames: the funcion accept a list of filenames where to source data. A single filename is fine also.
        - reso_freq: list of resonator frequency in MHz. This arguments is useful to plot only selected resonator from file. The resonator will be selected with the closest approximation of the f0.
        - backend: the backend used to plot. 'matplotlib' and 'plotly' are currently supported. Both will save a file to disk. Default is matplotlib.
        - verbose: print some debug line.
        - output_filename: set hte name of the output file without the extension (that depends on the backend).
        - auto_open: in case of plotly backend this enable or disable the opening of the plot in the browser.
        - attenuation: readout powe will be displayed in the legend, this value helps matching the SDR output power with the on-chip power. The value is in dB.
        - keyword args:
            - figsize: figure size for matplotlib backend.
            - add_info: listo of strings. Must be the same lresonreso_grp[resonator]atorength of the file list. Add information to the legend ion the plot.
            - title: Change the title of the plot.
            - single_plots: if this option is True a folder named resonators_<vna_filename> will be created/accessed and one plot per each resonator created. This option only works with the matplotlib backend.

    Return:
        - The filename of the saved plot.
    '''

    print("Plotting resonators...")
    if verbose: print_debug("Froms file(s):")

    filenames = to_list_of_str(filenames)

    if verbose:
        for name in filenames:
            print_debug("\t%s"%name)

    try:
        fig_size = kwargs['figsize']
    except KeyError:
        fig_size = None

    try:
        single_plots = kwargs['single_plots']
    except KeyError:
        single_plots = False

    try:
        add_info_labels = kwargs['add_info']
        if len(add_info_labels) != len(filenames):
            print_warning("Cannot add info labels. add_info has to be the same length of filenames")
            add_info_labels = None
    except KeyError:
        add_info_labels = None

    try:
        title = kwargs['title']
    except KeyError:
        if len(filenames) == 1:
            title = "Resonator(s) plot from file %s"%filenames[0]
        else:
            title = "Resonator(s) comparison plot"

    if len(filenames) == 0:
        err_msg = "File list empty, cannot plot resonators"
        print_error(err_msg)
        raise ValueError(err_msg)

    if output_filename is None:
        output_filename = "Resonators"
        if len(filenames)>1:
            output_filename+="_compare"
        output_filename+="_"+get_timestamp()

    if attenuation is None:
        attenuation = 0

    resonators = []
    fit_info = []
    brf = [] #best readout frequency
    if verbose: print_debug("Collecting data...")

    for filename in filenames:
        resonators.append( get_fit_data(filename, verbose) )
        fit_info.append(get_fit_param(filename, verbose) )
        brf.append( get_best_readout(filename, verbose) )

    if backend == "matplotlib":

        if verbose: print_debug("Using matplotlib backend...")

        if not single_plots:
            gridsize = (3,3)
            fig = pl.figure()
            ax_IQ = pl.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
            ax_mag = pl.subplot2grid(gridsize, (2, 0), colspan=3, rowspan=1)
            ax_pha = pl.subplot2grid(gridsize, (0, 2), colspan=1, rowspan=2)

            if fig_size is None:
                fig_size = (16, 10)

            fig.set_size_inches(fig_size[0], fig_size[1])
            formatter0 = EngFormatter(unit='Hz')
            fig.suptitle(title)
            line_labels = []
            legend_handler_list = []
            for i in range(len(filenames)):
                for j in range(len(resonators[i])):

                    # label informations
                    label = ""
                    if len(filenames) == 1:
                        pass
                    else:
                        label += "file: %s\n"%filenames[i]

                    label += "$f_0$: %.2f MHz\n"%(fit_info[i][j]['f0'])
                    Qi = 1./(1./fit_info[i][j]['Qr'] - 1./np.real(fit_info[i][j]['Qe']))
                    label+= "$Q_i$: %.2fk, $Q_r$: %.2fk\n"%(Qi/1e3,fit_info[i][j]['Qr']/1e3)

                    r_power = get_readout_power(filenames[i],0)# note that VNA has only one channel
                    if attenuation == 0 or attenuation is None:
                        label+='Output per-tone readout power: %.2f dB'%(r_power)
                    else:
                        label+= 'On-chip per-tone readout power: %.2f dB'%(r_power-attenuation)


                    color = get_color(i+j)
                    mag_fit = vrms2dbm( np.abs(resonators[i][j]['fitted']) )
                    phase_fit = np.angle(resonators[i][j]['fitted'])
                    mag_orig = vrms2dbm( np.abs(resonators[i][j]['original']) )
                    phase_orig = np.angle(resonators[i][j]['original'])
                    freq_axis = resonators[i][j]['frequency'] - fit_info[i][j]['f0']*1e6
                    readout_point = find_nearest(freq_axis,brf[i][j]- fit_info[i][j]['f0']*1e6)
                    readout_point_IQ = resonators[i][j]['original'][readout_point]


                    if verbose:
                        print_debug("Adding plot of %.2f MHz resonator"%(fit_info[i][j]['f0']))

                    #IQ plot is untouched
                    ax_IQ.plot(resonators[i][j]['fitted'].real, resonators[i][j]['fitted'].imag,color = color, linestyle = 'dotted')
                    fitl, = ax_IQ.plot(resonators[i][j]['original'].real, resonators[i][j]['original'].imag,color = color)
                    ax_IQ.scatter(readout_point_IQ.real, readout_point_IQ.imag, color=color)
                    ax_IQ.set_ylabel("I [ADC]")
                    ax_IQ.set_xlabel("Q [ADC]")


                    # magnitude and phase will be relative to F0
                    ax_mag.plot(freq_axis,mag_fit,color = color, linestyle = 'dotted')
                    ax_mag.plot(freq_axis,mag_orig,color = color)
                    ax_mag.grid()
                    ax_mag.scatter([freq_axis[readout_point],],[mag_orig[readout_point],], color = color)
                    ax_mag.set_ylabel("Magnitude [dB]")
                    ax_mag.set_xlabel("$\Delta f$ [Hz]")
                    ax_mag.xaxis.set_major_formatter(formatter0)

                    ax_pha.plot(phase_fit,freq_axis,color = color, linestyle = 'dotted')
                    ax_pha.plot(phase_orig,freq_axis,color = color)
                    ax_pha.scatter([phase_orig[readout_point],],[freq_axis[readout_point],], color = color)
                    #ax_pha.yaxis.tick_right()
                    ax_pha.grid()
                    ax_pha.set_ylabel("$\Delta f$ [Hz]")
                    ax_pha.set_xlabel("Phase [Rad]")
                    ax_pha.yaxis.set_major_formatter(formatter0)

                    line_labels += [label,]
                    legend_handler_list += [fitl,]

            ax_IQ.set_aspect('equal','datalim')
            ax_IQ.grid()
            ncol = 4
            fig.legend(handles = legend_handler_list,     # The line objects
                       labels=line_labels,   # The labels for each line
                       borderaxespad=0.3,    # Small spacing around legend box
                       ncol = ncol,
                       bbox_to_anchor=(0.95,-0.1),
                       loc="lower right",
                       bbox_transform=fig.transFigure,
                       title = "Solid lines are original data, marker is readout point."
                       )
            final_output_name = output_filename+".png"
            fig.savefig(final_output_name, bbox_inches = 'tight')

        else:
            ret_names = []

            #create output folder
            for i in range(len(filenames)):
                folder_name = "Resoplot_"+filenames[i].split('.')[0]
                try:
                    os.mkdir(folder_name)
                except OSError:
                    pass

                os.chdir(folder_name)
                filenames[i] = '../'+filenames[i]

            #creare and save each plot
                for j in range(len(resonators[i])):

                    output_filename = "Channel_%d"%j
                    gridsize = (3,3)
                    fig = pl.figure()
                    ax_IQ = pl.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
                    ax_mag = pl.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=1)
                    ax_pha = pl.subplot2grid(gridsize, (0, 2), colspan=1, rowspan=2)

                    if fig_size is None:
                        fig_size = (16, 10)

                    fig.set_size_inches(fig_size[0], fig_size[1])
                    formatter0 = EngFormatter(unit='Hz')
                    fig.suptitle(title)
                    line_labels = []
                    legend_handler_list = []

                    # label informations
                    label = ""
                    if len(filenames) == 1:
                        pass
                    else:
                        label += "file: %s\n"%filenames[i]

                    label += "$f_0$: %.2f MHz\n"%(fit_info[i][j]['f0'])
                    Qi = 1./(1./fit_info[i][j]['Qr'] - 1./np.real(fit_info[i][j]['Qe']))
                    label+= "$Q_i$: %.2fk, $Q_r$: %.2fk\n"%(Qi/1e3,fit_info[i][j]['Qr']/1e3)

                    r_power = get_readout_power(filenames[i],0)# note that VNA has only one channel
                    if attenuation == 0 or attenuation is None:
                        label+='Output per-tone readout power: %.2f dB'%(r_power)
                    else:
                        label+= 'On-chip per-tone readout power: %.2f dB'%(r_power-attenuation)


                    color = 'black'
                    mag_fit = vrms2dbm( np.abs(resonators[i][j]['fitted']) )
                    phase_fit = np.angle(resonators[i][j]['fitted'])
                    mag_orig = vrms2dbm( np.abs(resonators[i][j]['original']) )
                    phase_orig = np.angle(resonators[i][j]['original'])
                    freq_axis = resonators[i][j]['frequency'] - fit_info[i][j]['f0']*1e6
                    readout_point = find_nearest(freq_axis,brf[i][j]- fit_info[i][j]['f0']*1e6)
                    if verbose:
                        print_debug("Plotting %.2f MHz resonator"%(fit_info[i][j]['f0']))

                    #IQ plot is untouched
                    ax_IQ.plot(resonators[i][j]['fitted'].real, resonators[i][j]['fitted'].imag,color = color, linestyle = 'dotted')
                    fitl, = ax_IQ.plot(resonators[i][j]['original'].real, resonators[i][j]['original'].imag,color = color)
                    ax_IQ.set_ylabel("I [ADC]")
                    ax_IQ.set_xlabel("Q [ADC]")
                    readout_point_IQ = resonators[i][j]['original'][readout_point]
                    ax_IQ.scatter(readout_point_IQ.real, readout_point_IQ.imag, color=color)
                    ax_IQ.set_aspect('equal','datalim')

                    ax_IQ.grid()
                    # magnitude and phase will be relative to F0
                    ax_mag.plot(freq_axis,mag_fit,color = color, linestyle = 'dotted')
                    ax_mag.plot(freq_axis,mag_orig,color = color)
                    ax_mag.scatter([freq_axis[readout_point],],[mag_orig[readout_point],], color = color)
                    ax_mag.grid()
                    ax_mag.set_ylabel("Magnitude [dB]")
                    ax_mag.set_xlabel("$\Delta f$ [Hz]")
                    ax_mag.xaxis.set_major_formatter(formatter0)

                    ax_pha.plot(phase_fit,freq_axis,color = color, linestyle = 'dotted')
                    ax_pha.plot(phase_orig,freq_axis,color = color)
                    ax_pha.yaxis.tick_right()
                    ax_pha.yaxis.set_label_position("right")
                    ax_pha.scatter([phase_orig[readout_point],],[freq_axis[readout_point],], color = color)
                    ax_pha.grid()
                    ax_pha.set_ylabel("$\Delta f$ [Hz]")
                    ax_pha.set_xlabel("Phase [Rad]")
                    ax_pha.yaxis.set_major_formatter(formatter0)

                    line_labels += [label,]
                    legend_handler_list += [fitl,]

                    ax_mag.legend(handles = legend_handler_list,     # The line objects
                               labels=line_labels,   # The labels for each line
                               borderaxespad=0.3,    # Small spacing around legend box
                               bbox_to_anchor=(1.04,1),
                               title = "Solid lines are original data"
                               )
                    final_output_name = output_filename+".png"
                    fig.savefig(final_output_name, bbox_inches = 'tight')
                    pl.close(fig)
                    ret_names.append(folder_name+'/'+final_output_name)
                os.chdir('..')

            #return a list of filenames
            final_output_name = ret_names

    elif backend == "plotly":
    	fig = tools.make_subplots(
            rows=3, cols=3,
            specs=[
                [
                    {'rowspan': 2, 'colspan': 2 },
                    None,
                    {'rowspan': 3}
                ],
                [None, None,None],
                [{'colspan': 2}, None,None],
            ],
            print_grid=True,subplot_titles=('IQ circle', 'Phase','Magnitude')
        )
        title_fig = r'Fit (dashed) vs data (solid). Zero is set to f0 of each resonator, marker is the readout frequency.<br>'
        if len(filenames) == 1:
            title_fig+="From file %s"%filenames[0]
        else:
            title_fig+="From multiple (%d) files"%(len(filenames))
        for i in range(len(filenames)):
            for j in range(len(resonators[i])):

                if len(filenames) == 1:
                    label = ""
                else:
                    label += "file: %s<br>"%filenames[i]

                label += r'f_0: %.2f MHz<br>'%(fit_info[i][j]['f0'])
                Qi = 1./(1./fit_info[i][j]['Qr'] - 1./np.real(fit_info[i][j]['Qe']))
                label+= r'Qi: %.2fk, Qr: %.2fk<br>'%(Qi/1e3,fit_info[i][j]['Qr']/1e3)

                r_power = get_readout_power(filenames[i],0)# note that VNA has only one channel
                if attenuation == 0 or attenuation is None:
                    label+='Output per-tone readout power: %.2f dB'%(r_power)
                else:
                    label+= 'On-chip per-tone readout power: %.2f dB'%(r_power-attenuation)
                #sorry for monoblock programming, python sometimes goes crazy with indent
                x = i*len(resonators[i])+j
                c = get_color(i+j)
                mag_fit = vrms2dbm( np.abs(resonators[i][j]['fitted']) )
                phase_fit = np.angle(resonators[i][j]['fitted'])
                mag_orig = vrms2dbm( np.abs(resonators[i][j]['original']) )
                phase_orig = np.angle(resonators[i][j]['original'])
                freq_axis = resonators[i][j]['frequency'] - fit_info[i][j]['f0']*1e6
                readout_point = find_nearest(freq_axis,brf[i][j]- fit_info[i][j]['f0']*1e6)
                readout_point_IQ = resonators[i][j]['original'][readout_point]
                trace_magnitude = go.Scatter(x=freq_axis, y=mag_orig,name = label,legendgroup = str(x),line = dict(color = c))
                trace_phase = go.Scatter(y=freq_axis, x=phase_orig, showlegend = False,legendgroup = str(x),line = dict(color = c))
                trace_cirlce = go.Scatter(x=resonators[i][j]['original'].real, y=resonators[i][j]['original'].imag, showlegend = False,legendgroup = str(x),line = dict(color = c))
                trace_fit_magnitude = go.Scatter(x=freq_axis, y=mag_fit,showlegend = False,legendgroup = str(x),line = dict(dash = 'dash',color = c))
                trace_fit_phase = go.Scatter(y=freq_axis, x=phase_fit,legendgroup = str(x),showlegend = False,line = dict(dash = 'dash',color = c))
                trace_fit_circle = go.Scatter(x=resonators[i][j]['fitted'].real, y=resonators[i][j]['fitted'].imag,legendgroup = str(x),showlegend = False,line = dict(dash = 'dash',color = c))
                trace_f0_circle = go.Scatter(x = [readout_point_IQ.real,], y = [readout_point_IQ.imag,], name = "Best responsivity point",mode = 'markers',marker=dict(size=10,opacity = 1,symbol='circle',color = c,line = dict(width = 1,color = "black")),legendgroup = str(x))
                trace_f0_mag = go.Scatter(x=[freq_axis[readout_point],],y=[mag_orig[readout_point],],showlegend = False,mode = 'markers',marker=dict(size=10,opacity = 1,symbol='circle',color = c,line = dict(width = 1,color = "black")),legendgroup = str(x))
                trace_f0_phase = go.Scatter(x=[phase_orig[readout_point],],y=[freq_axis[readout_point],],showlegend = False,mode = 'markers',marker=dict(size=10,opacity = 1,symbol='circle',color = c,line = dict(width = 1,color = "black")),legendgroup = str(x))
                fig.append_trace(trace_cirlce,1,1)
                fig.append_trace(trace_f0_circle,1,1)
                fig.append_trace(trace_phase,1,3)
                fig.append_trace(trace_f0_phase,1,3)
                fig.append_trace(trace_magnitude,3,1)
                fig.append_trace(trace_f0_mag,3,1)
                fig.append_trace(trace_fit_circle,1,1)
                fig.append_trace(trace_fit_phase,1,3)
                fig.append_trace(trace_fit_magnitude,3,1)
                fig['layout'].update(title=title_fig)
                fig['layout']['xaxis3'].update(title='Relative Frequency [Hz]')
                fig['layout']['yaxis2'].update(title='Relarive Frequency [Hz]')
                fig['layout']['xaxis1'].update(title='Q [ADC]')
                fig['layout']['yaxis3'].update(title='Magnitude [dB]')
                fig['layout']['xaxis2'].update(title='Phase [Rad]')
                fig['layout']['yaxis1'].update(title='I [ADC]',scaleanchor = "x",)

        final_output_name = output_filename+".html"
        plotly.offline.plot(fig, filename=final_output_name,auto_open=auto_open)

    else:
        print_error("Resonator plot ha no %s backend implemented"%backend)

    return final_output_name

def plot_reso_stat(filenames, reso_freq = None, backend = 'matplotlib', title_info = None, additional_info = None, verbose = False, output_filename = None, auto_open = True, attr =  None):
    '''
    Plot the resonators parameters in function of the readout power or a custom attribute.

    Arguments:
        - filenames: the funcion accept a list of filenames where to source data. A single filename is fine also.
        - attr: if instead of plotting in function of power you want to plot in function of a custom attribute contained in the raw_data0 group, provvide here the name of the attribute as a string.
        - reso_freq: list of resonator frequency in MHz. This arguments is useful to plot only selected resonator from file. The resonator will be selected with the closest approximation of the f0.
        - backend: the backend used to plot. 'matplotlib' and 'plotly' are currently supported. Both will save a file to disk. Default is matplotlib.
        - verbose: print some debug line.
        - output_filename: set hte name of the output file without the extension (that depends on the backend).
        - auto_open: in case of plotly backend this enable or disable the opening of the plot in the browser.
        - keyword arguments:
            - figsize: fihure size for matplotlib backend.
            - add_info: listo of strings. Must be the same length of the file list. Add information to the legend ion the plot.
            - title: add information to the title of the plot.

    Return:
        - None
     '''
    return

def get_tones(filename, frontends = None, verbose = False):
    '''
    Get the central frequency and the list with relative tones.

    :param filename: the filename containing the fits (i.e. a VNA file).
    :param frontends: can be an antenna name ['A_RX2','B_RX2'], 'all' or None.
    :param verbose: print some debug line.

    :return if the argument 'all' is False return a tuple containing the central frequency and a list of relative tones; if None return the first frontend found in RX mode; if frontend name return the tones found in that frontend.

    '''
    if frontends is None:
        tones = get_best_readout(filename, verbose = verbose)
        info = get_rx_info(filename, ant=None)
        tones = tones - info['rf']
        if len(tones) ==0: print_warning("get_tones() returned an empty array")
        return info['rf'], tones
    elif frontends == 'all':
        front_end_list = ['A_RX2','B_RX2']
        ret_list = []
        for fe in front_end_list:
            info = get_rx_info(filename, ant=fe)
            if info['mode'] == 'RX':
                tones = get_best_readout(filename, verbose = verbose)
                tones = tones - info['rf']
                min_range = min(info['freq'][0],info['chirp_f'][0])
                max_range = max(info['freq'][0],info['chirp_f'][0])
                mask_ = tones>min_range and tones<max_range
                tones = tones[mask_]
                ret_list.append((
                    info['rf'],
                    tones
                ))
        return ret_list
    else:
        tones = get_best_readout(filename, verbose = verbose)
        info = get_rx_info(filename, ant=frontends)
        tones = tones - info['rf']
        min_range = min(info['freq'][0],info['chirp_f'][0])
        max_range = max(info['freq'][0],info['chirp_f'][0])
        tones = tones[tones>min_range]
        tones = tones[tones<max_range]
        if len(tones) ==0: print_warning("get_tones() returned an empty array")
        return info['rf'], tones
