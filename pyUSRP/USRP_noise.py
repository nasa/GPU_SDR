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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from matplotlib.ticker import EngFormatter

# needed to print the data acquisition process
import progressbar

# import submodules
from USRP_low_level import *
from USRP_files import *
from USRP_delay import *

def dual_get_noise(tones_A, tones_B, measure_t, rate, decimation = None, amplitudes_A = None, amplitudes_B = None, RF_A = None, RF_B = None, tx_gain_A = 0, tx_gain_B = 0, output_filename = None,
              Device = None, delay = None, pf_average = 4, **kwargs):
    '''
    Perform a noise acquisition using fixed tone technique on both frontend with a symmetrical PFB setup

    Arguments:
        - tones_A/B: list of ABSOLUTE tones frequencies in Hz for frontend A/B.
        - measure_t: duration of the measure in seconds.
        - decimation: the decimation factor to use for the acquisition. Default is minimum. Note that with the PFB the decimation factor can only be >= N_tones.
        - amplitudes_A/B: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF_A/B: central up/down mixing frequency. Default is deducted by other arguments.
        - tx_gain_A/B: gain to use in the transmission side.
        - output_filename: eventual filename. default is datetime.
        - Device: the on-server device number to use. default is 0.
        - delay: delay between TX and RX processes. Default is taken from the INTERNAL_DELAY variable.
        - pf_average: pfb averaging factor.
        - kwargs:
            * verbose: additional prints. Default is False.
            * push_queue: queue for post writing samples.

    Note:
        - In the PFB acquisition scheme the decimation factor and bin width are directly correlated. This function execute a check
          on the input parameters to determine the number of FFT bins to use.

    Returns:
        - filename of the measure file.
    '''

    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY, USRP_power

    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    try:
        push_queue = kwargs['push_queue']
    except KeyError:
        push_queue = None

    if output_filename is None:
        output_filename = "USRP_Noise_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    print("Begin dual noise acquisition, file %s ..."%output_filename)

    if measure_t <= 0:
        print_error("Cannot execute a noise measure with "+str(measure_t)+"s duration.")
        return ""

    tx_gain_A = int(np.abs(tx_gain_A))
    tx_gain_B = int(np.abs(tx_gain_B))
    if tx_gain_A > 31:
        print_warning("Max tx_gain_A is usually 31 dB. %d dB selected"%tx_gain_A)
    if tx_gain_B > 31:
        print_warning("Max tx_gain_B is usually 31 dB. %d dB selected"%tx_gain_B)

    if not (int(rate) in USRP_accepted_rates):
        print_warning("Requested rate will cause additional CIC filtering in USRP firmware. To avoid this use one of these rates (MHz): "+str(USRP_accepted_rates))

    if RF_A is None or RF_B is None:
        print_warning("Assuming tones are in absolute units (detector bandwith)")

        #calculate the splitting Frequency
        err_msg = 'Automatic frequency calculation has not been implemented for dual noise acquisition'
        print_error(err_msg)
        raise ValueError(err_msg)

    else:
        prev = len(tones_A) + len(tones_B)
        tones_A = tones_A[np.abs(tones_A) -  rate/2 < 0]
        tones_B = tones_B[np.abs(tones_B) -  rate/2 < 0]
        if len(tones_B) + len(tones_A) < prev:
            print_warning("Some tone has been discrarded because out of bandwidth")

        '''
        common = list(set(tones_A).intersection(tones_B))
        if len(common) > 0:
            print_warning("Some tone is compatible with both frontend tunings, see next messages.")
            for cc in common:
                if len(tones_A)>=len(tones_B):
                    print_debug("length of tones A: %d, length of tones B: %d, removing from %s"%(len(tones_A),len(tones_B),'A'))
                    np.delete(tones_A, np.where(tones_A == cc)[0][0])
                else:
                    print_debug("length of tones A: %d, length of tones B: %d, removing from %s"%(len(tones_A),len(tones_B),'B'))
                    np.delete(tones_B, np.where(tones_B == cc)[0][0])
        '''
        print_debug("RF_A\tRF_B\t[MHz]")
        print_debug("%.2f\t%.2f"%(RF_A/1e6,RF_B/1e6))
        print_debug("TONES_A\tTONES_B\t[MHz]")
        tones_strings = []
        for j in range(len(tones_A)):
            tones_strings.append("%.2f\t"%(tones_A[j]/1e6))
        for j in range(len(tones_B)):
            try:
                tones_strings[j]+="%.2f"%(tones_B[j]/1e6)
            except IndexError:
                tones_strings.append("\t%.2f"%(tones_B[j]/1e6))

    if amplitudes_A is not None:
        if len(amplitudes_A) != len(tones_A):
            print_warning("Amplitudes for RF A profile length is %d and it's different from tones array length that is %d. Amplitude profile will be ignored."%(len(amplitudes_A),len(tones_A)))
            amplitudes_A = [1. / len(tones_A) for x in tones_A]
        print_debug("Using %.1f percent of A DAC range"%(np.sum(amplitudes_A)*100))
    else:
        print_debug("Using 100 percent of RF A DAC ranges.")
        amplitudes_A = [1. / len(tones_A) for x in tones_A]

    if amplitudes_B is not None:
        if len(amplitudes_B) != len(tones_B):
            print_warning("Amplitudes for RF B profile length is %d and it's different from tones array length that is %d. Amplitude profile will be ignored."%(len(amplitudes_B),len(tones_B)))
            amplitudes_B = [1. / len(tones_B) for x in tones_B]
        print_debug("Using %.1f percent of /b DAC range"%(np.sum(amplitudes_B)*100))
    else:
        print_debug("Using 100 percent of RF B DAC ranges.")
        amplitudes_B = [1. / len(tones_B) for x in tones_B]

    TX_frontend_A = "A_TXRX"
    RX_frontend_A = "A_RX2"
    TX_frontend_B = "B_TXRX"
    RX_frontend_B = "B_RX2"

    if delay is None:
        try:
            delay = LINE_DELAY[str(int(rate/1e6))]
            delay *= 1e-9
        except KeyError:
            print_warning("Cannot find associated line delay for a rate of %d Msps. Setting to 0s."%(int(rate/1e6)))
            delay = 0
    else:
        print_debug("Using a delay of %d ns" % int(delay*1e9))

    # Calculate the number of channel needed per rf frontend A
    if len(tones_A)>1:
        min_required_space_A = np.min([ x for x in np.abs([[tones_A[i]-tones_A[j] if i!=j else 1e8 for j in range(len(tones_A))] for i in range(len(tones_A))]).flatten()])
        print_debug("Minimum bin width required for frontend A is %.2f MHz"%(min_required_space_A/1e6))
        min_required_fft_A = int(np.ceil(float(rate) / float(min_required_space_A)))
    else:
        min_required_fft_A = 10

    if decimation is not None:
        if decimation < min_required_fft_A:
            print_warning("Cannot use a decimation factor of %d as the minimum required number of bin in the PFB of frontend A is %d" % (decimation, min_required_fft_A))
            final_fft_bins_A = min_required_fft_A
        else:
            final_fft_bins_A = int(decimation)
    else:
        final_fft_bins_A = int(min_required_fft_A)

    if final_fft_bins_A<10:
        # Less that 10 pfb bins cause a bottleneck in the GPU: too many instruction to fetch.
        final_fft_bins_A = 10

    print_debug("Using %d PFB channels"%final_fft_bins_A)
    for i in range(len(tones_A)):
        if tones_A[i] > rate / 2:
            print_error("Out of bandwidth tone!")
            raise ValueError("Out of bandwidth tone requested in frontend A. %.2f MHz / %.2f MHz (Nyq)" %(tones_A[i]/1e6, rate / 2e6) )

    #tones_A = tones_A - RF_A
    tones_A = quantize_tones(tones_A, rate, final_fft_bins_A)

    # Calculate the number of channel needed per rf frontend B
    if len(tones_B)>1:
        min_required_space_B = np.min([ x for x in np.abs([[tones_B[i]-tones_B[j] if i!=j else 1e8 for j in range(len(tones_B))] for i in range(len(tones_B))]).flatten()])
        print_debug("Minimum bin width required for frontend B is %.2f MHz"%(min_required_space_B/1e6))
        min_required_fft_B = int(np.ceil(float(rate) / float(min_required_space_B)))
    else:
        min_required_fft_B = 10

    if decimation is not None:
        if decimation < min_required_fft_B:
            print_warning("Cannot use a decimation factor of %d as the minimum required number of bin in the PFB of frontend A is %d" % (decimation, min_required_fft_B))
            final_fft_bins_B = min_required_fft_B
        else:
            final_fft_bins_B = int(decimation)
    else:
        final_fft_bins_B = int(min_required_fft_B)

    if final_fft_bins_B<10:
        # Less that 10 pfb bins cause a bottleneck in the GPU: too many instruction to fetch.
        final_fft_bins_B = 10

    print_debug("Using %d PFB channels"%final_fft_bins_B)
    for i in range(len(tones_B)):
        if tones_B[i] > rate / 2:
            print_error("Out of bandwidth tone!")
            raise ValueError("Out of bandwidth tone requested in frontend A. %.2f MHz / %.2f MHz (Nyq)" %(tones_A[i]/1e6, rate / 2e6) )

    #tones_B = tones_B - RF_B
    tones_B = quantize_tones(tones_B, rate, final_fft_bins_B)

    number_of_samples = rate * measure_t


    print_warning("overriding number of bins: calculation above it's wrong")
    final_fft_bins_B = int(decimation)
    final_fft_bins_A = int(decimation)
    '''
    print("RF Frontend A")
    print("Tone [MHz]\tPower [dBm]\tOffset [MHz]")
    for i in range(len(tones_A)):
        print("%.1f\t%.2f\t%.3f" % ((RF_A + tones_A[i]) / 1e6, USRP_power + 20 * np.log10(amplitudes_A[i]) + tx_gain_A, tones_A[i] / 1e6))



    print("RF Frontend B")
    print("Tone [MHz]\tPower [dBm]\tOffset [MHz]")
    for i in range(len(tones_B)):
        print("%.1f\t%.2f\t%.3f" % ((RF_B + tones_B[i]) / 1e6, USRP_power + 20 * np.log10(amplitudes_B[i]) + tx_gain_B, tones_B[i] / 1e6))
    '''
    expected_samples_B = int(number_of_samples/final_fft_bins_B)
    expected_samples_A = int(number_of_samples/final_fft_bins_A)

    noise_command = global_parameter()

    noise_command.set(TX_frontend_A, "mode", "TX")
    noise_command.set(TX_frontend_A, "buffer_len", 1e6)
    noise_command.set(TX_frontend_A, "gain", tx_gain_A)
    noise_command.set(TX_frontend_A, "delay", 1)
    noise_command.set(TX_frontend_A, "samples", number_of_samples)
    noise_command.set(TX_frontend_A, "rate", rate)
    noise_command.set(TX_frontend_A, "bw", 2 * rate)
    #noise_command.set(TX_frontend, 'tuning_mode', 0)
    noise_command.set(TX_frontend_A, "wave_type", ["TONES" for x in tones_A])
    noise_command.set(TX_frontend_A, "ampl", amplitudes_A)
    noise_command.set(TX_frontend_A, "freq", tones_A)
    noise_command.set(TX_frontend_A, "rf", RF_A)

    # This parameter does not have an effect (except suppress a warning from the server)
    noise_command.set(TX_frontend_A, "fft_tones", 100)

    noise_command.set(RX_frontend_A, "mode", "RX")
    #noise_command.set(RX_frontend, 'tuning_mode', 0)
    noise_command.set(RX_frontend_A, "buffer_len", 1e6)
    noise_command.set(RX_frontend_A, "gain", 0)
    noise_command.set(RX_frontend_A, "delay", 1 + delay)
    noise_command.set(RX_frontend_A, "samples", number_of_samples)
    noise_command.set(RX_frontend_A, "rate", rate)
    noise_command.set(RX_frontend_A, "bw", 2 * rate)

    noise_command.set(RX_frontend_A, "wave_type", ["TONES" for x in tones_A])
    noise_command.set(RX_frontend_A, "freq", tones_A)
    noise_command.set(RX_frontend_A, "rf", RF_A)
    noise_command.set(RX_frontend_A, "fft_tones", final_fft_bins_A)
    noise_command.set(RX_frontend_A, "pf_average", pf_average)

    # With the polyphase filter the decimation is realized increasing the number of channels.
    # This parameter will average in the GPU a certain amount of PFB outputs.
    noise_command.set(RX_frontend_A, "decim", 0)

    noise_command.set(TX_frontend_B, "mode", "TX")
    noise_command.set(TX_frontend_B, "buffer_len", 1e6)
    noise_command.set(TX_frontend_B, "gain", tx_gain_B)
    noise_command.set(TX_frontend_B, "delay", 1)
    noise_command.set(TX_frontend_B, "samples", number_of_samples)
    noise_command.set(TX_frontend_B, "rate", rate)
    noise_command.set(TX_frontend_B, "bw", 2 * rate)
    #noise_command.set(TX_frontend, 'tuning_mode', 0)
    noise_command.set(TX_frontend_B, "wave_type", ["TONES" for x in tones_B])
    noise_command.set(TX_frontend_B, "ampl", amplitudes_B)
    noise_command.set(TX_frontend_B, "freq", tones_B)
    noise_command.set(TX_frontend_B, "rf", RF_B)

    # This parameter does not have an effect (except suppress a warning from the server)
    noise_command.set(TX_frontend_B, "fft_tones", 100)

    noise_command.set(RX_frontend_B, "mode", "RX")
    #noise_command.set(RX_frontend, 'tuning_mode', 0)
    noise_command.set(RX_frontend_B, "buffer_len", 1e6)
    noise_command.set(RX_frontend_B, "gain", 0)
    noise_command.set(RX_frontend_B, "delay", 1 + delay)
    noise_command.set(RX_frontend_B, "samples", number_of_samples)
    noise_command.set(RX_frontend_B, "rate", rate)
    noise_command.set(RX_frontend_B, "bw", 2 * rate)

    noise_command.set(RX_frontend_B, "wave_type", ["TONES" for x in tones_B])
    noise_command.set(RX_frontend_B, "freq", tones_B)
    noise_command.set(RX_frontend_B, "rf", RF_B)
    noise_command.set(RX_frontend_B, "fft_tones", final_fft_bins_B)
    noise_command.set(RX_frontend_B, "pf_average", pf_average)

    # With the polyphase filter the decimation is realized increasing the number of channels.
    # This parameter will average in the GPU a certain amount of PFB outputs.
    noise_command.set(RX_frontend_B, "decim", 0)

    if noise_command.self_check():
        if (verbose):
            print_debug("Noise command successfully checked")
            noise_command.pprint()

        Async_send(noise_command.to_json())

    else:
        err_msg = "Something went wrong in the noise acquisition command self_check()"
        print_error(err_msg)
        raise ValueError(err_msg)

    Packets_to_file(
        parameters=noise_command,
        timeout=None,
        filename=output_filename,
        dpc_expected=max(expected_samples_A,expected_samples_B),
        meas_type="Noise",
        push_queue = push_queue,
        **kwargs
    )

    print_debug("Noise acquisition terminated.")

    return output_filename

def Get_noise(tones, measure_t, rate, decimation = None, amplitudes = None, RF = None, tx_gain = 0, output_filename = None, Front_end = None,
              Device = None, delay = None, pf_average = 4, **kwargs):
    '''
    Perform a noise acquisition using fixed tone technique.

    Arguments:
        - tones: list of tones frequencies in Hz (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - decimation: the decimation factor to use for the acquisition. Default is minimum. Note that with the PFB the decimation factor can only be >= N_tones.
        - amplitudes: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - tx_gain: gain to use in the transmission side.
        - output_filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - delay: delay between TX and RX processes. Default is taken from the INTERNAL_DELAY variable.
        - pf_average: pfb averaging factor.
        - kwargs:
            * verbose: additional prints. Default is False.
            * push_queue: queue for post writing samples.


    Note:
        - In the PFB acquisition scheme the decimation factor and bin width are directly correlated. This function execute a check
          on the input parameters to determine the number of FFT bins to use.

    Returns:
        - filename of the measure file.
    '''

    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY, USRP_power

    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    try:
        push_queue = kwargs['push_queue']
    except KeyError:
        push_queue = None

    if output_filename is None:
        output_filename = "USRP_Noise_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    print("Begin noise acquisition, file %s ..."%output_filename)

    if measure_t <= 0:
        print_error("Cannot execute a noise measure with "+str(measure_t)+"s duration.")
        return ""

    tx_gain = int(np.abs(tx_gain))
    if tx_gain > 31:
        print_warning("Max gain is usually 31 dB. %d dB selected"%tx_gain)

    if not (int(rate) in USRP_accepted_rates):
        print_warning("Requested rate will cause additional CIC filtering in USRP firmware. To avoid this use one of these rates (MHz): "+str(USRP_accepted_rates))

    if RF is None:
        print_warning("Assuming tones are in absolute units (detector bandwith)")

        # Calculate the optimal RF central frequency
        RF = np.mean(tones)
        tones = np.asarray(tones) - RF
        print_debug("RF central frequency will be %.2f MHz"%(RF/1e6))

    if amplitudes is not None:
        if len(amplitudes) != len(tones):
            print_warning("Amplitudes profile length is %d and it's different from tones array length that is %d. Amplitude profile will be ignored."%(len(amplitudes),len(tones)))
            amplitudes = [1./len(tones) for x in tones]

        used_DAC_range = np.sum(amplitudes)

        print_debug("Using %.1f percent of DAC range"%(used_DAC_range*100))

    else:
        print_debug("Using 100 percent of the DAC range.")

        amplitudes = [1. / len(tones) for x in tones]

    if Front_end is None:
        Front_end = 'A'

    if not Front_end_chk(Front_end):
        err_msg = "Cannot detect front_end: "+str(Front_end)
        print_error(err_msg)
        raise ValueError(err_msg)
    else:
        TX_frontend = Front_end+"_TXRX"
        RX_frontend = Front_end+"_RX2"

    if delay is None:
        try:
            delay = LINE_DELAY[str(int(rate/1e6))]
            delay *= 1e-9
        except KeyError:
            print_warning("Cannot find associated line delay for a rate of %d Msps. Setting to 0s."%(int(rate/1e6)))
            delay = 0
    else:
        print_debug("Using a delay of %d ns" % int(delay*1e9))

    # Calculate the number of channel needed
    if len(tones)>1:
        min_required_space = np.min([ x for x in np.abs([[i-j for j in tones] for i in tones]).flatten() if x > 0])
        print_debug("Minimum bin width required is %.2f MHz"%(min_required_space/1e6))
        min_required_fft = int(np.ceil(float(rate) / float(min_required_space)))
    else:
        min_required_fft = 10

    if decimation is not None:
        if decimation < min_required_fft:
            print_warning("Cannot use a decimation factor of %d as the minimum required number of bin in the PFB is %d" % (decimation, min_required_fft))
            final_fft_bins = min_required_fft
        else:
            final_fft_bins = int(decimation)
    else:
        final_fft_bins = int(min_required_fft)

    if final_fft_bins<10:
        # Less that 10 pfb bins cause a bottleneck in the GPU: too many instruction to fetch.
        final_fft_bins = 10

    print_debug("Using %d PFB channels"%final_fft_bins)
    for i in range(len(tones)):
        if tones[i] > rate / 2:
            print_error("Out of bandwidth tone!")
            raise ValueError("Out of bandwidth tone requested. %.2f MHz / %.2f MHz (Nyq)" %(tones[i]/1e6, rate / 2e6) )

    tones = quantize_tones(tones, rate, final_fft_bins)
    number_of_samples = rate * measure_t

    print("Tone [MHz]\tPower [dBm]\tOffset [MHz]")
    for i in range(len(tones)):
        print("%.1f\t%.2f\t%.3f" % ((RF + tones[i]) / 1e6, USRP_power + 20 * np.log10(amplitudes[i]), tones[i] / 1e6))

    expected_samples = int(number_of_samples/final_fft_bins)
    noise_command = global_parameter()

    noise_command.set(TX_frontend, "mode", "TX")
    noise_command.set(TX_frontend, "buffer_len", 1e6)
    noise_command.set(TX_frontend, "gain", tx_gain)
    noise_command.set(TX_frontend, "delay", 1)
    noise_command.set(TX_frontend, "samples", number_of_samples)
    noise_command.set(TX_frontend, "rate", rate)
    noise_command.set(TX_frontend, "bw", 2 * rate)
    #noise_command.set(TX_frontend, 'tuning_mode', 0)
    noise_command.set(TX_frontend, "wave_type", ["TONES" for x in tones])
    noise_command.set(TX_frontend, "ampl", amplitudes)
    noise_command.set(TX_frontend, "freq", tones)
    noise_command.set(TX_frontend, "rf", RF)

    # This parameter does not have an effect (except suppress a warning from the server)
    noise_command.set(TX_frontend, "fft_tones", 100)

    noise_command.set(RX_frontend, "mode", "RX")
    #noise_command.set(RX_frontend, 'tuning_mode', 0)
    noise_command.set(RX_frontend, "buffer_len", 1e6)
    noise_command.set(RX_frontend, "gain", 0)
    noise_command.set(RX_frontend, "delay", 1 + delay)
    noise_command.set(RX_frontend, "samples", number_of_samples)
    noise_command.set(RX_frontend, "rate", rate)
    noise_command.set(RX_frontend, "bw", 2 * rate)

    noise_command.set(RX_frontend, "wave_type", ["TONES" for x in tones])
    noise_command.set(RX_frontend, "freq", tones)
    noise_command.set(RX_frontend, "rf", RF)
    noise_command.set(RX_frontend, "fft_tones", final_fft_bins)
    noise_command.set(RX_frontend, "pf_average", pf_average)

    # With the polyphase filter the decimation is realized increasing the number of channels.
    # This parameter will average in the GPU a certain amount of PFB outputs.
    noise_command.set(RX_frontend, "decim", 0)

    if noise_command.self_check():
        if (verbose):
            print_debug("Noise command successfully checked")
            noise_command.pprint()

        Async_send(noise_command.to_json())

    else:
        err_msg = "Something went wrong in the noise acquisition command self_check()"
        print_error(err_msg)
        raise ValueError(err_msg)

    Packets_to_file(
        parameters=noise_command,
        timeout=None,
        filename=output_filename,
        dpc_expected=expected_samples,
        meas_type="Noise",
        push_queue = push_queue,
        **kwargs
    )

    print_debug("Noise acquisition terminated.")

    return output_filename


def spec_from_samples(samples, sampling_rate=1, welch=None, dbc=False, rotate=True, verbose=True, clip_samples = False):
    '''
    Calculate real and imaginary part of the spectra of a complex array using the Welch method.

    Arguments:
        - Samples: complex array representing samples.
        - sampling_rate: sampling_rate
        - welch: in how many segment to divide the samples given for applying the Welch method
        - dbc: scales samples to calculate dBc spectra.
        - rotate: if True rotate the IQ plane
    Returns:
        - Frequency array,
        - Imaginary spectrum array
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

    if not clip_samples:
        end_clip_samples = len(samples)
        start_clip_samples = 0
    else:
        end_clip_samples = int(len(samples) - clip_samples)
        start_clip_samples = int(clip_samples)

    if verbose: print_debug("Selecting from %d sample to %d sample" % (start_clip_samples, end_clip_samples))


    if rotate:
        samples = samples * (np.abs(np.mean(samples)) / np.mean(samples))

    if dbc:
        samples = samples / np.mean(samples)
        samples = samples - np.mean(samples)

    Frequencies, RealPart = signal.welch(samples[start_clip_samples:end_clip_samples].real, nperseg=welch, fs=sampling_rate, detrend='linear',
                                         scaling='density')
    Frequencies, ImaginaryPart = signal.welch(samples[start_clip_samples:end_clip_samples].imag, nperseg=welch, fs=sampling_rate, detrend='linear',
                                              scaling='density')



    return Frequencies, 10 * np.log10(RealPart), 10 * np.log10(ImaginaryPart)


def calculate_noise(filename, welch=None, dbc=False, rotate=True, usrp_number=0, ant=None, verbose=False, clip=0.1):
    '''
    Generates the FFT of each channel stored in the .h5 file and stores the results in the same file.
    Arguments:
        - welch: in how many segment to divide the samples given for applying the Welch method.
        - dbc: scales samples to calculate dBc spectra.
        - rotate: if True rotate the IQ plane.

    TODO:
        Default behaviour should be getting all the available RX antenna.
    '''

    print("Calculating noise spectra for " + filename)

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
        if active_RX_param['decim']>1:
            sampling_rate /= active_RX_param['decim']

    except TypeError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1

    except ZeroDivisionError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1

    if clip is not False:
        clip_samples = int(clip * sampling_rate)
    else:
        clip_samples = None

    f, samples, errors = openH5file(
        filename,
        ch_list=None,
        start_sample=clip_samples,  # ignored
        last_sample=None,  # ignored
        usrp_number=usrp_number,
        front_end=None,  # this will change for double RX
        verbose=verbose,
        error_coord=True,
        big_file=True
    )

    if len(errors) > 0:
        print_error("Cannot evaluate spectra of samples containing transmission error")
        return

    if verbose: print_debug("Calculating spectra...")


    Results = Parallel(n_jobs=min(N_CORES,12), verbose=1, backend=parallel_backend)(
        delayed(spec_from_samples)(
            i, sampling_rate=sampling_rate, welch=welch, dbc=dbc, rotate=rotate, clip_samples = clip_samples,
            verbose=verbose
        ) for i in samples
    )


    f.close()

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
        print_warning("Overwriting Noise subgroup %s in h5 file" % ant[0])
        del noise_group[ant[0]]
        noise_subgroup = noise_group.create_group(ant[0])

    if welch is None:
        welch = 0

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

    print_debug("calculate_noise_spec() done.")
    fv.close()

def plot_noise_spec(filenames, channel_list=None, max_frequency=None, title_info=None, backend='matplotlib',
                    cryostat_attenuation=0, auto_open=True, output_filename=None, **kwargs):
    '''
    Plot the noise spectra of given, pre-analized, H5 files.

    Arguments:
        - filenames: list of strings containing the filenames.
        - channel_list:
        - max_frequency: maximum frequency to plot.
        - title_info: add a custom line to the plot title
        - backend: see plotting backend section for informations.
        - auto_open: open the plot in default system browser if plotly backend is selected (non-blocking) or open the matplotlib figure (blocking). Default is True.
        - output_filename: output filename without any system extension. Default is Noise_timestamp().xx
        - kwargs:
            * usrp_number and front_end can be passed to the openH5file() function.
            * tx_front_end can be passed to manually determine the tx frontend to calculate the readout power.
            * add_info could be a list of the same length oF filenames containing additional legend information.
            * html will make the function retrn html code instead of saving a html file in case of plotly backend.
            * fig_size: matplotlib fig size in inches (xx,yy).

        :return the name of the file saved
    '''

    filenames = to_list_of_str(filenames)

    if not (backend in ['matplotlib', 'plotly']):
        err_msg = "Cannot plot noise with backend \'%s\': not implemented"%backend
        print_error(err_msg)
        raise ValueError(err_msg)
    try:
        fig_size = kwargs['figsize']
    except KeyError:
        fig_size = None

    try:
        html = kwargs['html']
    except KeyError:
        html = False

    if len(filenames)>1:
        print("Plotting noise from files:")
        for f in filenames:
            print("\t%s"%f)
    else:
        print("Plotting noise from file %s ..."%filenames[0])

    add_info_labels = None
    try:
        add_info_labels = kwargs['add_info']
    except KeyError:
        pass

    if output_filename is None:
        output_filename = "Noise_"
        if len(filenames)>1:
            output_filename+="compare_"
        output_filename += get_timestamp()

    plot_title = 'USRP Noise spectra from '
    if len(filenames) < 2:
        plot_title += "file: " + filenames[0] + "."
    else:
        plot_title += "multiple files."

    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=1, ncols=1)
        if fig_size is None:
            fig_size = (16, 10)

        fig.set_size_inches(fig_size[0], fig_size[1])

        ax.set_xlabel("Frequency [Hz]")


    elif backend == 'plotly':
        fig = tools.make_subplots(rows=1, cols=1)
        fig['layout']['xaxis1'].update(title="Frequency [Hz]", type='log', exponentformat='SI', ticksuffix='Hz')

    y_name_set = True
    rate_tag_set = True

    try:
        usrp_number = kwargs['usrp_number']
    except KeyError:
        usrp_number = None
    try:
        front_end = kwargs['front_end']
    except KeyError:
        front_end = None
    try:
        tx_front_end = kwargs['tx_front_end']
    except KeyError:
        tx_front_end = None
    f_count = 0
    for filename in filenames:
        info, freq, real, imag = get_noise(
            filename,
            usrp_number=usrp_number,
            front_end=front_end,
            channel_list=channel_list
        )
        if max_frequency is not None:
            for ii in range(len(imag)):
                index_cut = find_nearest(freq, max_frequency)
                index_cut = np.min([len(freq),len(real[ii]),index_cut])
                freq = freq[:index_cut]
                imag[ii] = imag[ii][:index_cut]
                real[ii] = real[ii][:index_cut]

        if y_name_set:
            y_name_set = False

            if backend == 'matplotlib':
                if info['dbc']:
                    ax.set_ylabel("PSD [dBc]")
                else:
                    ax.set_ylabel("PSD [dBm/sqrt(Hz)]")

            elif backend == 'plotly':
                if info['dbc']:
                    fig['layout']['yaxis1'].update(title="PSD [dBc]")
                else:
                    fig['layout']['yaxis1'].update(title="PSD [dBm/sqrt(Hz)]")

        if rate_tag_set:
            rate_tag_set = False
            if info['rate'] / 1e6 > 1.:
                plot_title += "Effective rate: %.2f Msps" % (info['rate'] / 1e6)
            else:
                plot_title += "Effective rate: %.2f ksps" % (info['rate'] / 1e3)

        for i in range(len(info['tones'])):
            readout_power = get_readout_power(filename, i, tx_front_end, usrp_number) - cryostat_attenuation
            label = filename+"<br>"
            label += "%.2f MHz" % (info['tones'][i] / 1e6)
            R = real[i]
            I = imag[i]
            if backend == 'matplotlib':
                label += "\nReadout pwr %.1f dBm" % (readout_power)
                if add_info_labels is not None:
                    label += "\n" + add_info_labels[f_count]
                ax.semilogx(freq, R, '--', color=get_color(f_count + i), label="Real " + label)
                ax.semilogx(freq, I, color=get_color(f_count + i), label="Imag " + label)
            elif backend == 'plotly':
                label += "<br>Readout pwr %.1f dBm" % (readout_power)
                if add_info_labels is not None:
                    label += "<br>" + add_info_labels[f_count]
                fig.append_trace(go.Scatter(
                    x=freq,
                    y=R,
                    name="Real " + label,
                    legendgroup="group" + str(i) + "file" + str(f_count),
                    line=dict(color=get_color(f_count + i)),
                    mode='lines'
                ), 1, 1)
                fig.append_trace(go.Scatter(
                    x=freq,
                    y=I,
                    name="Imag " + label,
                    legendgroup="group" + str(i) + "file" + str(f_count),
                    line=dict(color=get_color(f_count + i), dash='dot'),
                    mode='lines'
                ), 1, 1)
        # increase file counter
        f_count += 1
    if backend == 'matplotlib':
        if title_info is not None:
            plot_title += "\n" + title_info
        fig.suptitle(plot_title)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        formatter0 = EngFormatter(unit='Hz')
        ax.xaxis.set_major_formatter(formatter0)
        ax.grid(True)
        output_filename += '.png'
        print_debug("Saving %s..."%output_filename)
        fig.savefig(output_filename, bbox_inches="tight")
        pl.close(fig)


    elif backend == 'plotly':
        if title_info is not None:
            plot_title += "<br>" + title_info

        fig['layout'].update(title=plot_title)
        #fig['layout']['xaxis'].update(exponentformat='SI')
        #fig['layout']['xaxis'].update(ticksuffix='Hz')
        output_filename += ".html"
        if html:
            print_debug("Noise plotting done")
            return  plotly.offline.plot(fig, filename=output_filename, auto_open=False, output_type = 'div')
        plotly.offline.plot(fig, filename=output_filename, auto_open=auto_open)

    print_debug("Noise plotting done")
    return output_filename


def calculate_frequency_timestream(noise_frequency, noise_data, fit_param):
	"""
    Convert IQ timestreams into frequency and quality factor timestreams.
	Derived from Albert's function to convert noise data in f0 stream data.
    The original function has been stripped of the matplotlib capabilities and adapted to the scope of this library.

	Arguments:
		- noise_frequency: float, Noise acquisition tone in Hz.
		- noise_data: list of complex, Noise data already scaled as S21 (see diagnosic() function).
		- fit_param: if fit parameters are given in the form (f0, A, phi, D, Qi, Qr, Qe_re, Qe_im,a, _, _, pcov), the fit won't be executed again.

	Returns:
		- X noise
		- Qr noise

	"""

	try:
		f0, A, phi, D, Qi, Qr, Qe_re, Qe_im,a = fit_param
	except:
        err_msg = "Fit parameter given to calculate_frequency_timestream() are not good."
		print_error(err_msg)
		raise ValueError(err_msg)

	Qe = Qe_re + 1.j*Qe_im

	dQe = 1./Qe

	f0 *= 1e6

	#1. Remove the cable phase and the amplitude scaling from the time streams
	n_amplitude =   A * np.exp(2.j*pi*(1e-6*D*(noise_frequency  -f0) + phi))

	#print "noise offet: %.2f: "%np.abs(n_amplitude)
	noise_data /= n_amplitude

	qrx_noise = dQe/(1.-noise_data)

	return 1./qrx_noise.real, f0*qrx_noise.imag/2.


def copy_resonator_group(VNA_filename, NOISE_filename):
    '''
    Copy the resonator groups from a VNA file to a mnoise file.

    Arguments:
        - VNA_filename: name of the file containing the resonator group (can also be an other noise file).
        - NOISE_filename: name of the file in which to copy the resonator group. If an other resonator group is in place, it will be rewrited.

    Returns:
        - None
    '''
    VNA_filename = format_filename(VNA_filename)
    #check if the orinal ha the resonator group

    NOISE_filename = format_filename(NOISE_filename)

    #if the group is already present overwrite

    print_debug("Copying resonator group from \'%s\' to \'%s\' ..."%(VNA_filename,NOISE_filename))


    return

def get_frequency_timestreams(NOISE_filename, start = None, end = None, channel_freq = None, frontend = None):
    '''
    Returns the frequency and quality factor timestreams from a noise file in which a resonator group has been already copied.
    To copy the resonator group refer to copy_resonator_group() function.

    Arguments:
        - NOISE_filename: Name of the noise file.
        - start: start time in seconds. Default is from the beginning of the file.
        - end: end of the data in seconds. Default is up to file's end.
        - channel_freq: list of frequency of the channels to return. Default is all of them.
        - frontend: from which frontend to take the noise data. Default is A.

    Returns:
        - tuple containing frequency timestreams and quality factor timestreams. Each element of the tuple is a list of timestreams.

    Example:
        >>> frequencies, Q_factors = get_frequency_timestreams("noisefile.h5", start = 1, end = 1.5, channel_freq = 325.5):
        >>> # This will retrive frequency and quality factor timestreams of the 325.5 MHz channel (or closest) from the file "noisefile.h5" between 1 and 1.5 seconds of acquisition.
    '''

    NOISE_filename = format_filename(NOISE_filename)
    print_debug("Opening file \'%s\'..."%NOISE_filename)
    if frontend is not None:
        if frontend == 'A':
            ant = "A_RX2"
        elif frontend == 'B':
            ant = "B_RX2"
        else:
            err_msg = "cannot recognize frontend code \'%s\' in get_frequency_timestreams()"%frontend
            print_error(err_msg)
            raise ValueError(err_msg)
    else:
        ant = frontend

    info = get_rx_info(NOISE_filename, ant=ant)
    if start is not None:
        time_conv = float(info['rate'])/info['fft_tones']
        start_sample = time_conv*start
        end_sample = time_conv*end
    else:
        start_sample = start

    tones = np.asarray(info['tones'])+info['rf']

    if channel_freq is not None:
        print_debug("Channel selected: ")
        numeric_channel_list = []
        for x in channel_freq:
            j = find_nearest(tones,x)
            numeric_channel_list.append(j)
            print_debug("%d) %.2f MHz"%(len(numeric_channel_list),tones[j]/1e6))

    else:
        numeric_channel_list = channel_freq

    params = get_fit_param(NOISE_filename, verbose = False)

    data = openH5file(NOISE_filename, ch_list=numeric_channel_list, start_sample=start_sample, last_sample=last_sample, usrp_number=None, front_end=frontend,
                   verbose=False, error_coord=False, big_file = False)
    result_f = []
    result_q = []
    for i in range(len(data)):
        f0, A, phi, D, Qi, Qr, Qe_re, Qe_im,a
        fit_param = (params[i]['f0'], params[i]['A'], params[i]['phi'], params[i]['D'], params[i]['Qi'], params[i]['Qr'], np.real(params[i]['Qe']),np.imag(params[i]['Qe']),params[i]['a'])
        f_ts, q_ts = calculate_frequency_timestream(tones[i], data[i], fit_param)
        result_f.append(f_ts)
        result_q.append(q_ts)

    return result_f, result_q

def plot_frequency_timestreams():
    # copied from plot_raw
    return

def calculate_NEF_spectra():
    #copied from calculate_spec
    return

def get_NEF_spec():
    return

def plot_NEF_spectra():
    #plot_spec
    return
