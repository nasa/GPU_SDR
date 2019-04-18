########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################

import numpy as np
import scipy.signal as signal
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
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

# needed to print the data acquisition process
import progressbar

# import submodules
from USRP_low_level import *
from USRP_files import *
from USRP_VNA import *
from USRP_low_level import *
from scipy import optimize



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


def FWMH(freq, magnitude):
    magnitude = np.abs(magnitude)
    min_point = freq[np.argmax(magnitude)]
    MH = (np.max(magnitude) - np.mean([magnitude[0], magnitude[-1]])) / 2.
    sel_freq = freq[magnitude > MH]
    return np.abs(min(sel_freq) - max(sel_freq))


def do_fit(freq, re, im, p0=None):
    '''
    Notes:
        - f0 in p0 is in MHz
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

def initialize_peaks(filename, N_peaks = 1, smoothing = None, peak_width = 90e3, Qr_cutoff=5e3, verbose = False, exclude_center = True, diagnostic_plots = False):
    """
    This function uses a filter on quality factor estimated using the nonlinear resonator model. This function considers the resonator around the maximum of the unwraped phase trace, tries to fit the resonator and make a decision if it's a resonator or not by parsing the quality factor of the fit with the Qt_vutoff argument. Before iterating excludes a zone of peak_width Hz around the previously considered point.
    Stores in the H5 file the result. This does not count as a fit as only the initialization is stored.

    Arguments:
        - Filename: filename of the h5 vna file.
        - smoothing: if the vna is too noise one may consider decimation and fir filtering. dmoothing is the decimation factor.
        - peak_width: minimum distance between each peak.
        - Qr_cutoff: thrashold above wich a fit will result in a resonaton being stored.
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

    resolution = np.abs(info['freq'] - info['chirp_f'])/float(len(S21))
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
        os.mkdir(diagnostic_folder)
        print_debug("Generating diagnostic plots in folder \'%s\'...")

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
            if np.abs(freq[ii] - center) < (10 * resolution):
                mask[ii] = False

    # Optimizing on the magnitude of derivative of S21
    gradS21 = np.abs(np.gradient(S21_val))

    iteration_number = 0

    while(sum(mask)>0):

        #find maximum and fit
        maximum = np.argmax(gradS21*mask)
        low_index = max(maximum-peak_width,0)
        high_index =maximum+peak_width
        try:
            with nostdout():
                f0,Qi,Qr,zfit,modelwise = do_fit(
                    freq[low_index:high_index],
                    S21_val.real[low_index:high_index],
                    S21_val.imag[low_index:high_index],
                    p0=None
                )
        except RuntimeError:
            Qr = 0

        if Qr>Qr_cutoff:
            print_debug("Resonator found at %.2f MHz"%(freq[maximum]/1.e6))
            max_diag.append(maximum)
            q_diag.append(Qr)
            f0s.append(f0)
            label_set = "Accepted init:\nQr: %.2fk / %.2fk"%(Qr/1e3, Qr_cutoff/1e3)
            col = 'green'
        else:
            label_set = "Refused init:\nQr: %.2fk / %.2fk"%(Qr/1e3, Qr_cutoff/1e3)
            col = 'red'

        if diagnostic_plots:
            os.chdir(diagnostic_folder)
            fig, ax = pl.subplots(nrows=1, ncols=1)
            fig.suptitle("Peak initialization diagnosic #%d"%iteration_number)
            ax.plot(
                freq[low_index:high_index],
                magnitudedb[low_index:high_index],
                label = label_set,
                color = col
            )
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("S21 Magnitude [dB]")
            ax.grid()
            ax.legend()
            fig.savefig("init_diag_%d.png"%iteration_number, bbox_inches="tight")
            pl.close(fig)
            os.chdir("..")

        # If the number of peaks expected is reached, stop
        if len(max_diag) >= N_peaks:
            break

        # Remove points from mask
        for i in range(len(phase)):
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
            pass

        results = [freq[j] for j in max_diag]
        reso_grp.attrs.__setitem__("tones_init", results)

        fv.close()

    print("Initialize_peaks() found " +str(len(max_diag))+ " resonators.")

    if N_peaks!=len(results):
        return False
    else:
        return True

def get_init_peaks(filename):
    '''
    Get initialized peaks froma a VNA file.

    Arguments:
        - filename: the name of the file containing the peaks.

    Return:
        - Numpy array containing the frequency of each ninitialized peak in MHz.
    '''

    file = bound_open(filename)

    try:
        inits = file["Resonators"].attrs.get("tones_init")
    except ValueError:
        inits = np.asarray([])
        print_warning("get_init_peaks() did not find any initialized peak")
    except KeyError:
        inits = np.asarray([])
        print_warning("get_init_peaks() did not find any initialized peak")
    file.close()

    return np.asarray(inits)

def vna_fit(filename, p0=None, fit_range = 12e4, verbose = False):
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
    resolution = np.abs(info['freq'] - info['chirp_f'])/float(len(S21))
    model = nonlinear_model

    if len(peaks_init) == 0:
        err_msg = "Cannot find any initialized peak"
        print_error(err_msg)
        raise ValueError(err_msg)

    fv = h5py.File(filename+".h5",'r+')

    # there is no try catch on this experssion because it's preceded by get_init_peaks()
    reso_grp = fv['Resonators']

    # WARNING: this number is coherent only in a single file: it may be NOT consistent arcoss multiple files!
    fit_number = 0

    for tone in peaks_init:
        if verbose: print_debug("Fitting resonator initialized at %.2f MHz ..."%(tone/1e3))

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

            single_reso_grp = reso_grp.create_group("reso_%d"%fit_number)

            # Write fitting data
            single_reso_grp.create_dataset("freq", base_fit_freq)
            single_reso_grp.create_dataset("base_S21", S21[selection])
            single_reso_grp.create_dataset("fitted_S21", zfit)

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

    if len(peaks_init)-1!=fit_number:
        print_warning("One or more fit went wrong")

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
    for resonator in reso_grp:
        ret.append({
            "frequency":np.asarray(resonator['frequency']),
            "fitted":np.asarray(resonator["fitted_S21"]),
            "original":np.asarray(resonator["base_S21"])
            })

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
    for resonator in reso_grp:
        ret.append({
            'f0':resonator["f0"],
            'A':resonator["A"],
            'phi':resonator["phi"],
            'D':resonator["D"],
            'Qi':resonator["Qi"],
            'Qr':resonator["Qr"],
            'Qe':resonator["Qe"],
            'a':resonator["a"]
        })
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
    for resonator in reso_grp:
        delta_r = 1./Qr
    	brf = 1e6*f0 * (1 - a*delta_r)
        if verbose: print_debug("Resonator %.2f is shifted %.2fkHz"%(f0, 1e3*np.abs(brf/1e6-f0)))
        ret.append(brf)

    return ret

def plot_resonators(filenames, reso_freq = None, backend = 'matplotlib', title_info = None, verbose = False, output_filename = None, auto_open = True, **kwargs):
    '''
    Plot the resonators and the resonator fits.

    Arguments:
        - filenames: the funcion accept a list of filenames where to source data. A single filename is fine also.
        - reso_freq: list of resonator frequency in MHz. This arguments is useful to plot only selected resonator from file. The resonator will be selected with the closest approximation of the f0.
        - backend: the backend used to plot. 'matplotlib' and 'plotly' are currently supported. Both will save a file to disk. Default is matplotlib.
        - verbose: print some debug line.
        - output_filename: set hte name of the output file without the extension (that depends on the backend).
        - auto_open: in case of plotly backend this enable or disable the opening of the plot in the browser.
        - keyword args:
            - figsize: figure size for matplotlib backend.
            - add_info: listo of strings. Must be the same length of the file list. Add information to the legend ion the plot.
            - title: Change the title of the plot.

    Return:
        - The filename of the saved plot.
    '''

    print("Plotting resonators")
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

    resonators = []
    fit_info = []
    brf = [] #best readout frequency
    r_powers = []
    if verbose: print_debug("Collecting data...")

    for filename in filenames:
        resonators += get_fit_data(filename, verbose)
        fit_info += get_fit_param(filename, verbose)
        brf.append( get_best_readout(filename, verbose) )
        r_powers.append( get_readout_power(filename) )

    if backend == "matplotlib":

        if verbose: print_debug("Using matplotlib backend...")

        gridsize = (3,3)
        fig = pl.figure()
        ax_IQ = pl.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
        ax_mag = pl.subplot2grid(gridsize, (3, 0), colspan=2, rowspan=1)
        ax_pha = pl.subplot2grid(gridsize, (0, 2), colspan=1, rowspan=2)

        if fig_size is None:
            fig_size = (16, 10)

        fig.set_size_inches(fig_size[0], fig_size[1])

        fig.suptitle(title)

        for i in range(len(filenames)):

            mag_fit = vrms2dbm( np.abs(resonators['fitted']) )
            phase_fit = np.angle(resonators['fitted'])
            mag_orig = vrms2dbm( np.abs(resonators['original']) )
            phase_orig = np.angle(resonators['original'])

            ax_IQ.plot(resonators['fitted'].real, resonators['fitted'].imag,)
            ax_IQ.plot(resonators['original'].real, resonators['original'].imag,)

            ax_mag.plot(resonators['frequency'],mag_fit,)
            ax_mag.plot(resonators['frequency'],mag_orig,)

            phase_orig.plot(phase_fit,resonators['frequency'],)
            phase_orig.plot(phase_orig,resonators['frequency'],)






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

def get_tones(filename, verbose = False):
    return
