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
from threading import Thread,Condition
import multiprocessing
from joblib import Parallel, delayed
from subprocess import call
import time
import gc
import datetime

#plotly stuff
from plotly.graph_objs import Scatter, Layout
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl

#matplotlib stuff
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

#needed to print the data acquisition process
import progressbar

#import submodules
from USRP_low_lever import *
from USRP_files import *
from USRP_parameters import *


def linear_phase(phase):
    '''
    Unwrap the pgase and subtract linear and constrant offset.
    '''
    phase = np.unwrap(phase)
    x = np.arange(len(phase))
    m, q = np.polyfit(x, phase, 1)

    linear_phase = m * x + q
    phase -= linear_phase

    return phase


def spec_from_samples(samples, sampling_rate=1, welch=None, dbc=False, rotate=False, verbose=True):
    '''
    Calculate real and immaginary part of the spectra of a complex array using the Welch method.

    Arguments:
        - Samples: complex array representing samples.
        - sampling_rate: sampling_rate
        - welch: in how many segment to divide the samples given for applying the Welch method
        - dbc: scales samples to calculate dBc spectra.
        - rotate: if True rotate the IQ plane
    Returns:
        - Frequency array,
        - Immaginary spectrum array
        - Real spectrum array
    '''
    if verbose: print_debug("[Welch worker]")
    try:
        L = len(samples)
    except TypeError:
        print_error("Expecting complex array for spectra calculation, got something esle.")
        return None, None, None

    if welch == None:
        welch = L
    else:
        welch = int(L / welch)

    samples = samples.astype(np.complex128)

    if rotate:
        samples = samples * (np.abs(np.mean(samples)) / np.mean(samples))

    if dbc:
        samples = samples / np.mean(samples)
        samples = samples - np.mean(samples)

    Frequencies, RealPart = signal.welch(samples.real, nperseg=welch, fs=sampling_rate, detrend='linear',
                                         scaling='density')
    Frequencies, ImaginaryPart = signal.welch(samples.imag, nperseg=welch, fs=sampling_rate, detrend='linear',
                                              scaling='density')

    return Frequencies, RealPart, ImaginaryPart


def calculate_noise(filename, welch=None, dbc=False, rotate=False, usrp_number=0, ant=None, verbose=True, clip=0.5):
    '''
    Generates the FFT of each channel stored in the .h5 file and stores the results in the same file.
    '''

    if verbose: print_debug("Calculating noise spectra for " + filename)

    if verbose: print_debug("Reading attributes...")

    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)

    if ant is None:
        ant = parameters.get_active_rx_param()
    else:
        ant = to_list_of_str(ant)

    if len(ant) > 1:
        print_error("multiple RX devices not yet supported")
        return

    active_RX_param = parameters.parameters[ant[0]]

    try:
        sampling_rate = active_RX_param['rate'] / active_RX_param['fft_tones']

    except TypeError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1

    except ZeroDivisionError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1

    if sampling_rate != 1:
        clip_samples = clip * sampling_rate
    else:
        clip_samples = None

    if verbose: print_debug("Opening file...")

    samples, errors = openH5file(
        filename,
        ch_list=None,
        start_sample=clip_samples,
        last_sample=None,
        usrp_number=usrp_number,
        front_end=None,  # this will change for double RX
        verbose=True,
        error_coord=True
    )
    if len(errors) > 0:
        print_error("Cannot evaluate spectra of samples containing transmission error")
        return

    if verbose: print_debug("Calculating spectra...")

    Results = Parallel(n_jobs=N_CORES, verbose=1, backend=parallel_backend)(
        delayed(spec_from_samples)(
            np.asarray(i), sampling_rate=sampling_rate, welch=welch, dbc=dbc, rotate=rotate, verbose=verbose
        ) for i in samples
    )

    if verbose: print_debug("Saving result on file " + filename + " ...")

    fv = h5py.File(filename, 'r+')

    noise_group_name = "Noise" + str(int(usrp_number))

    try:
        noise_group = fv.create_group(noise_group_name)
    except ValueError:
        noise_group = fv[noise_group_name]

    try:
        noise_subgroup = noise_group.create_group(ant[0])
    except ValueError:
        if verbose: print_debug("Overwriting Noise subgroup %s in h5 file" % ant[0])
        del noise_group[ant[0]]
        noise_subgroup = noise_group.create_group(ant[0])

    noise_subgroup.attrs.create(name="welch", data=welch)
    noise_subgroup.attrs.create(name="dbc", data=dbc)
    noise_subgroup.attrs.create(name="rotate", data=rotate)
    noise_subgroup.attrs.create(name="rate", data=sampling_rate)
    noise_subgroup.attrs.create(name="n_chan", data=len(Results))

    noise_subgroup.create_dataset("freq", data=Results[0][0], compression=H5PY_compression)

    for i in range(len(Results)):
        tone_freq = active_RX_param['rf'] + active_RX_param['freq'][i]
        ds = noise_subgroup.create_dataset("real_" + str(i), data=Results[i][1], compression=H5PY_compression,
                                           dtype=np.dtype('Float32'))
        ds.attrs.create(name="tone", data=tone_freq)
        ds = noise_subgroup.create_dataset("imag_" + str(i), data=Results[i][2], compression=H5PY_compression,
                                           dtype=np.dtype('Float32'))
        ds.attrs.create(name="tone", data=tone_freq)

    if verbose: print_debug("calculate_noise_spec() has done.")
    fv.close()
