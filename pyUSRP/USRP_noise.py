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
from USRP_fitting import get_fit_param
from USRP_fitting import get_fit_data

def dual_get_noise(tones_A, tones_B, measure_t, rate, decimation = None, amplitudes_A = None, amplitudes_B = None, RF_A = None, RF_B = None, tx_gain_A = 0, tx_gain_B = 0, output_filename = None,
              Device = None, delay = None, pf_average = None, mode = "DIRECT" ,**kwargs):
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
        - pf_average: pfb averaging factor. Default is 4 for PFB mode and 1 for DIRECT mode.
        - mode: noise acquisition kernels. DIRECT uses direct demodulation PFB use the polyphase filter bank technique. Note that PF average will refer to something slightly different in DIRECT mode (moving average ratio: 1 has no overlap).
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

    if ((mode != "PFB") and (mode != "DIRECT")):
        error_msg = "Noise acquisition mode %s not defined" % str(mode)
        print_error(err_msg)
        raise ValueError(err_msg)

    if pf_average is None:
        if mode == "DIRECT":
            pf_average = 1
        elif mode == "PFB":
            pf_average = 4
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

    if mode == "PFB":
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

    elif mode == "DIRECT":
        expected_samples_B = number_of_samples/int(decimation)
        expected_samples_A = expected_samples_B
        err_msg = "Double noise measurement not implemented yet. It's not hard to port from the single noise measure anyway."
        print_error(err_msg)
        raise(err_msg)

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
              Device = None, delay = None, pf_average = 4, mode = "DIRECT", trigger = None, **kwargs):
    '''
    Perform a noise acquisition using fixed tone technique.

    Arguments:
        - tones: list of tones frequencies in Hz (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - decimation: the decimation factor to use for the acquisition. Default for PFB mode is minimum. Note that with the PFB the decimation factor can only be >= N_tones. This parameter represent actual decimation for the DIRECT mode.
        - amplitudes: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - tx_gain: gain to use in the transmission side.
        - output_filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - delay: delay between TX and RX processes. Default is taken from the INTERNAL_DELAY variable.
        - pf_average: pfb averaging factor. Default is 4 for PFB mode and 1 for DIRECT mode.
        - mode: noise acquisition kernels. DIRECT uses direct demodulation PFB use the polyphase filter bank technique. Note that PF average will refer to something slightly different in DIRECT mode (moving average ratio: 1 has no overlap).
        - trigger: class used for triggering. (See trigger section for more info). Default is no trigger.
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

    if ((mode != "PFB") and (mode != "DIRECT")):
        error_msg = "Noise acquisition mode %s not defined" % str(mode)
        print_error(err_msg)
        raise ValueError(err_msg)

    if pf_average is None:
        if mode == "DIRECT":
            pf_average = 1
        elif mode == "PFB":
            pf_average = 4

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
    if mode == "PFB":
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

    elif mode =="DIRECT":
        buffer_len = int(1e6)
        decimation = int(decimation)
        number_of_samples = rate * measure_t
        if decimation != 0:
            expected_samples = int(float(number_of_samples)/decimation)
            if buffer_len % decimation != 0:
                error_msg = "Cannot use a decimation factor of %d with a buffer len of %d as decimation % buffer_len must be 0"
                print_error(error_mas)
                raise ValueError(error_msg)
        else:
            expected_samples = int(number_of_samples)
            print_debug("GPU FIR filter disabled")

        if trigger is not None:
            expected_samples = None

        #Quantize tones to 1Hz resolution
        tones = [int(tt) for tt in tones]

        noise_command = global_parameter()
        noise_command.set(TX_frontend, "mode", "TX")
        noise_command.set(TX_frontend, "buffer_len", buffer_len)
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
        noise_command.set(RX_frontend, "buffer_len", buffer_len)
        noise_command.set(RX_frontend, "gain", 0)
        noise_command.set(RX_frontend, "delay", 1 + delay)
        noise_command.set(RX_frontend, "samples", number_of_samples)
        noise_command.set(RX_frontend, "rate", rate)
        noise_command.set(RX_frontend, "bw", 2 * rate)

        noise_command.set(RX_frontend, "wave_type", ["DIRECT" for x in tones])
        noise_command.set(RX_frontend, "freq", tones)
        noise_command.set(RX_frontend, "rf", RF)
        noise_command.set(RX_frontend, "fft_tones", 0) # in this case this value is discarded
        noise_command.set(RX_frontend, "pf_average", pf_average)

        noise_command.set(RX_frontend, "decim", decimation)

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
        trigger = trigger,
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

    Frequencies, RealPart = signal.welch(samples[start_clip_samples:end_clip_samples].real, nperseg=welch, fs=sampling_rate, detrend='linear',scaling='density')
    Frequencies, ImaginaryPart = signal.welch(samples[start_clip_samples:end_clip_samples].imag, nperseg=welch, fs=sampling_rate, detrend='linear',scaling='density')


    return Frequencies, 10 * np.log10(RealPart), 10 * np.log10(ImaginaryPart)


def calculate_noise(filename, welch=None, dbc=False, rotate=True, usrp_number=0, ant=None, verbose=False, clip=0.1):
    '''
    Generates the FFT of each channel stored in the .h5 file and stores the results in the same file.

    :param welch: in how many segment to divide the samples given for applying the Welch method.
    :param dbc: scales samples to calculate dBc spectra.
    :param rotate: if True rotate the IQ plane.

    TODO:
    * Default behaviour should be getting all the available RX antenna.
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
        if active_RX_param['wave_type'][0] == "DIRECT":
            if (active_RX_param['decim']>0):
                sampling_rate = float(active_RX_param['rate']) / active_RX_param['decim']
            else:
                sampling_rate = float(active_RX_param['rate'])
        else:
            sampling_rate = float(active_RX_param['rate']) / active_RX_param['fft_tones']
            if active_RX_param['decim']>1:
                sampling_rate /= float(active_RX_param['decim'])
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


    Results = Parallel(n_jobs=min(N_CORES,3), verbose=1, backend=parallel_backend)(
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
        if len(add_info_labels) != len(filenames):
            print_warning("Cannot add info labels on noise plot. len(add_info_labels)(%d)!=len(filenames)(%d)"%(len(add_info_labels),len(filenames)))
            add_info_labels = None
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
        fig['layout']['xaxis1'].update(title="Frequency [Hz]")#), type='log')

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
                    ax.set_ylabel("PSD [dBc/Hz]")
                else:
                    ax.set_ylabel("PSD [dBm/Hz]")

            elif backend == 'plotly':
                if info['dbc']:
                    fig['layout']['yaxis1'].update(title="PSD [dBc/Hz]")
                else:
                    fig['layout']['yaxis1'].update(title="PSD [dBm/Hz]")

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
        f0, A, phi, D, Qi, Qr, Qe_re, Qe_im, a = fit_param
    except:
        err_msg = "Fit parameter given to calculate_frequency_timestream() are not good."
        print_error(err_msg)
        raise ValueError(err_msg)

    Qe = Qe_re + 1.j*Qe_im

    dQe = 1./Qe

    f0 *= 1e6

    #1. Remove the cable phase and the amplitude scaling from the time streams
    n_amplitude =   A * np.exp(2.j*np.pi*(1e-6*D*(noise_frequency  -f0) + phi))

    #print "noise offet: %.2f: "%np.abs(n_amplitude)
    noise_data /= n_amplitude

    qrx_noise = dQe/(1.-noise_data)

    return f0*qrx_noise.imag/2., 1./qrx_noise.real


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
    resonator_grp_name = "Resonators"
    VNA_fv = h5py.File(VNA_filename, 'r')
    if resonator_grp_name not in VNA_fv.keys():
        err_msg = 'VNA file:%s does not contain the Resonators group'%VNA_filename
        print_error(err_msg)
        raise ValueError(err_msg)

    NOISE_filename = format_filename(NOISE_filename)
    NOISE_fv = h5py.File(NOISE_filename, 'r+')

    print_debug("Copying resonator group from \'%s\' to \'%s\' ..."%(VNA_filename,NOISE_filename))


    resonator_grp = VNA_fv[resonator_grp_name]

    try:
        NOISE_fv_reso = NOISE_fv.create_group(resonator_grp_name)
    except ValueError:
        print_warning("Overwriting Noise subgroup %s in h5 file" % resonator_grp_name)
        del NOISE_fv[resonator_grp_name]
    else:
        del NOISE_fv[resonator_grp_name]
    # noise_resonator_group = noise_group.create_group(resonator_group_name)
    NOISE_fv.copy(resonator_grp, NOISE_fv)

    VNA_fv.close()
    NOISE_fv.close()

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
    last_sample = None
    if start is not None:
        if info['wave_type'] == "TONES":
            decimation_factor = info['fft_tones']/max(info['decim'],1.)
        else:
            decimation_factor = info['decim']
        time_conv = float(info['rate'])/decimation_factor
        start_sample = time_conv*start
        if end is not None:
            last_sample = time_conv*end
    else:
        start_sample = 0

    tones = np.asarray(info['freq'])+info['rf']

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
    for i in range(len(params)):
        # f0, A, phi, D, Qi, Qr, Qe_re, Qe_im,a
        fit_param = (params[i]['f0'], params[i]['A'], params[i]['phi'], params[i]['D'], params[i]['Qi'], params[i]['Qr'], np.real(params[i]['Qe']),np.imag(params[i]['Qe']),params[i]['a'])
        f_ts, q_ts = calculate_frequency_timestream(tones[i], data[i], fit_param)
        result_f.append(f_ts - np.mean(f_ts))
        result_q.append(q_ts - np.mean(q_ts))

    return result_f, result_q

def plot_frequency_timestreams(filenames, decimation=None, displayed_samples=None, low_pass=None, backend='matplotlib', output_filename=None,
                  channel_list=None, start_time=None, end_time=None, auto_open=True, **kwargs):
        '''
        Plot frequency timestreams from given H5 noise files.

        Arguments:
            - a list of strings containing the files to plot.
            - decimation: eventually deciamte the signal before plotting.
            - displayed_samples: calculate decimation to display a certain number of samples.
            - low pass: floating point number controlling the cut-off frequency of a low pass filter that is eventually applied to the data.
            - backend: [string] choose the return type of the plot. Allowed backends for now are:
                * matplotlib: creates a matplotlib figure, plots in non-blocking mode and return the matplotlib figure object. kwargs in this case accept:
                    - size: size of the plot in the form of a tuple (inches,inches). Default is matplotlib default.
                * plotly: plot using plotly and webgl interface, returns the html code descibing the plot. kwargs in this case accept:
                    - size: size of the plot. Default is plotly default.
                * bokeh: use bokeh to generate an interactive html file containing the IQ plane and the magnitude/phase timestream.

            - output_filename: string: name of the file saved. Default is a timestamp.
            - channel_list: select only al list of channels to plot.
            - start_time: time where to start plotting. Default is 0.
            - end_time: time where to stop plotting. Default is end of the measure.
            - auto_open: open the plot in default system browser if plotly backend is selected (non-blocking). Default is True.
            - kwargs:
                * usrp_number and front_end can be passed to the openH5file() function.
                * fig_size: the size of matplotlib figure.
                * add_info: list of strings as long as the file list to add info to the legend.

        Returns:
            - the complete name of the saved file or None in case no file is saved.

        Note:
            - Possible errors are signaled on the plot.
        '''
        downsample_warning = True
        overwriting_decim_waring = True
        try:
            fig_size = kwargs['figsize']
        except KeyError:
            fig_size = None

        if output_filename is None:
            output_filename = "USRP_freq_timestream_"+get_timestamp()
        try:
            add_info_labels = kwargs['add_info']
        except KeyError:
            add_info_labels = None
        plot_title = 'USRP frequency/Qr timestream '
        if backend == 'matplotlib':
            fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)
            if fig_size is None:
                fig_size = (16, 10)
            fig.set_size_inches(fig_size[0], fig_size[1])
            ax[1].set_xlabel("Time [s]")
            formatter0 = EngFormatter(unit='s')
            ax[1].xaxis.set_major_formatter(formatter0)
            formatter1 = EngFormatter(unit='')
            ax[0].set_ylabel("Frequency Shift [Hz]")
            ax[1].set_ylabel("Qr Shift [-]")
            ax[0].yaxis.set_major_formatter(formatter1)
            ax[1].yaxis.set_major_formatter(formatter1)


        elif backend == 'plotly':
            fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('f0 timestream', 'Qr timestream'),
                                      shared_xaxes=True)
            fig['layout']['yaxis1'].update(title='Frequency Shift [Hz]')
            fig['layout']['yaxis2'].update(title='Qr Shift [-]')
            fig['layout']['xaxis'].update(exponentformat='SI')
            fig['layout']['xaxis1'].update(title='Time [s]')

        filenames = to_list_of_str(filenames)

        print_debug("Plotting from files:")
        for i in range(len(filenames)):
            print_debug("%d) %s" % (i, filenames[i]))
        try:
            usrp_number = kwargs['usrp_number']
        except KeyError:
            usrp_number = None
        try:
            front_end = kwargs['front_end']
        except KeyError:
            front_end = None
        file_count = 0
        for filename in filenames:

            filename = format_filename(filename)
            parameters = global_parameter()
            parameters.retrive_prop_from_file(filename)
            ant = parameters.get_active_rx_param()

            if len(ant) > 1:
                print_error("multiple RX devices not yet supported")
                return
            freq = None
            if parameters.get(ant[0], 'wave_type')[0] == "TONES":
                decimation_factor = parameters.get(ant[0], 'fft_tones')
                freq = np.asarray(parameters.get(ant[0], 'freq')) + parameters.get(ant[0], 'rf')
                if parameters.get(ant[0], 'decim') != 0:
                    decimation_factor *= parameters.get(ant[0], 'decim')

                effective_rate = parameters.get(ant[0], 'rate') / float(decimation_factor)

            elif parameters.get(ant[0], 'wave_type')[0] == "CHIRP":
                if parameters.get(ant[0], 'decim') != 0:
                    effective_rate = parameters.get(ant[0], 'swipe_s')[0] / parameters.get(ant[0], 'chirp_t')[0]
                else:
                    effective_rate = parameters.get(ant[0], 'rate')

            else:
                decimation_factor = max(1, parameters.get(ant[0], 'fft_tones'))

                if parameters.get(ant[0], 'decim') != 0:
                    decimation_factor *= parameters.get(ant[0], 'decim')

                effective_rate = parameters.get(ant[0], 'rate') / float(decimation_factor)

            if start_time is not None:
                file_start_time = start_time * effective_rate
            else:
                file_start_time = 0
            if end_time is not None:
                file_end_time = end_time * effective_rate
            else:
                file_end_time = None

            freq_ts, qr_ts = get_frequency_timestreams(filename,
                start = file_start_time,
                end = file_end_time,
                channel_freq = None,
                frontend = None
            )
            # samples, errors = openH5file(
            #     filename,
            #     ch_list=channel_list,
            #     start_sample=file_start_time,
            #     last_sample=file_end_time,
            #     usrp_number=usrp_number,
            #     front_end=front_end,
            #     verbose=False,
            #     error_coord=True
            # )

            #print_debug("plot_raw_data() found %d channels each long %d samples" % (len(samples), len(samples[0])))
            if channel_list == None:
                ch_list = range(len(freq_ts))
            else:
                if max(channel_list) > len(freq_ts):
                    print_warning(
                        "Channel list selected in plot_raw_data() is bigger than avaliable channels. plotting all available channels")
                    ch_list = range(len(freq_ts))
                else:
                    ch_list = channel_list


            # prepare samples TODO
            for i in ch_list:


                Y1 = freq_ts[i]
                Y2 = qr_ts[i]

                if displayed_samples is not None:
                    if decimation is not None and overwriting_decim_waring:
                        print_warning("Overwriting offline decimation arguments with displayed_samples")
                        overwriting_decim_waring = False
                    decimation = int(len(freq_ts[i])/displayed_samples)
                    if decimation <= 1 and downsample_warning:
                        print_warning("Channel does not require decimation to reach the number of displayed samples")
                        downsample_warning = False
                if decimation is not None and decimation > 1:
                    decimation = int(np.abs(decimation))
                    Y1 = signal.decimate(Y1, decimation, ftype='fir')
                    Y2 = signal.decimate(Y2, decimation, ftype='fir')
                else:
                    decimation = 1

                X = np.arange(len(Y1)) / float(effective_rate / decimation) + file_start_time / float(effective_rate)

                if effective_rate / 1e6 > 1:
                    rate_tag = 'rate: %.2f Msps' % (effective_rate / 1e6)
                else:
                    rate_tag = 'rate: %.2f ksps' % (effective_rate / 1e3)

                if freq is None:
                    label = "Channel %d" % i
                else:
                    label = "Channel %.2f MHz" % (freq[i] / 1.e6)

                if backend == 'matplotlib':
                    label += "\n" + filename
                    if add_info_labels is not None:
                        label += "\n" + add_info_labels[file_count]
                    ax[0].plot(X[decimation:-decimation], Y1[decimation:-decimation], color=get_color(i + file_count), label=label)
                    ax[1].plot(X[decimation:-decimation], Y2[decimation:-decimation], color=get_color(i + file_count))
                elif backend == 'plotly':
                    label += "<br>" + filename
                    if add_info_labels is not None:
                        label += "<br>" + add_info_labels[file_count]
                    fig.append_trace(go.Scatter(
                        x=X[decimation:-decimation],
                        y=Y1[decimation:-decimation],
                        name=label,
                        legendgroup="group" + str(i) + "file" + str(file_count),
                        line=dict(color=get_color(i + file_count)),
                        mode='lines'
                    ), 1, 1)
                    fig.append_trace(go.Scatter(
                        x=X[decimation:-decimation],
                        y=Y2[decimation:-decimation],
                        # name = "channel %d"%i,
                        showlegend=False,
                        legendgroup="group" + str(i) + "file" + str(file_count),
                        line=dict(color=get_color(i + file_count)),
                        mode='lines'
                    ), 2, 1)
            file_count += 1
        final_filename = ""
        if backend == 'matplotlib':
            fig.suptitle(plot_title + "\n" + rate_tag)
            handles, labels = ax[0].get_legend_handles_labels()
            ax[0].legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left")
            ax[0].grid(True)
            ax[1].grid(True)
            final_filename = output_filename + '.png'
            fig.savefig(final_filename, bbox_inches="tight")
            pl.close(fig)

        if backend == 'plotly':
            final_filename = output_filename + ".html"
            fig['layout'].update(title=plot_title + "<br>" + rate_tag)
            plotly.offline.plot(fig, filename=final_filename, auto_open=auto_open)

        return final_filename


def diagnostic_VNA_noise(noise_filename, points = None, VNA_file = None, ant = "A_RX2", backend = 'matplotlib', **kwargs):
    '''
    Plot the VNA traces and the noise (averaged or in N points) on the same plot to check for acquisition consistency.
    The noise file has to contain the Resonators group in order to use this function; to copy that from a VNA file use the function #copy_resonator_group().

    :param noise_filename: noise acquisition filename.
    :param points: the default behaviour is to average every channel in a single point; if this argument is >1 the noise will be decimated in that number of points.
    :param VNA_file: if the fit information has to be taken from an external VNA file fill this argument with the filename.
    :param backend: Choose the plotting backend. Currently implemented: plotly and matplotlib
    :param ant: specify the antenna used to take noise data. Default is A_RX2
    :param kwargs:
        * auto_open: plotly backend specific, determines if after saving the plot the browser is called.
        * figsize: matplotlib specific argument: figure size of the plot or each plot.

    TODO: allow this function to interpret multiple VNA sources and multiple noise files.

    '''
    def db(value):
        return 20*np.log10(value)

    noise_filename = format_filename(noise_filename)
    print("Plotting diagnostic data from \'%s\'"%noise_filename)
    resonator_grp_name = "Resonators"
    info = get_rx_info(noise_filename, ant=ant)
    tx_info = get_tx_info(noise_filename, ant=ant.split('_')[0]+"_TXRX")
    noise_file = h5py.File(noise_filename, 'r')

    try:
        fig_size = kwargs['figsize']
    except KeyError:
        fig_size = None

    #source fit data
    fit_source_filename = noise_filename
    extra_source_info = ""
    if VNA_file is not None:
        fit_source_filename = format_filename(VNA_file)
        extra_source_info = " Fit data taken from \'%s\'."%fit_source_filename
        print_debug(extra_source_info)
    fit_param = get_fit_param(fit_source_filename)
    fit_data = get_fit_data(fit_source_filename)

    #check backend existance
    if ((backend!='matplotlib') and (backend!='plotly')):
        err_msg = "backend %s not implemented in diagnostic_VNA_noise() function"%backend
        print_error(err_msg)
        raise ValueError(err_msg)

    #check Resonators group existance
    if resonator_grp_name not in noise_file.keys():
        err_msg = "Cannot find the Resonator group in the file %s" % noise_filename
        print_error(err_msg)
        raise ValueError(err_msg)

    #check resonator group length matching
    if (len(fit_data) != len(info['wave_type'])):
        warning_msg = "The length of the resonator group (%d) does not match the number of tones (%d) in file %s."%(len(fit_data), len(info['wave_type']), fit_source_filename)
        print_warning(warning_msg)

    #check frequency matching
    tones = np.asarray(info['freq']) + info['rf']
    fit_freqs = np.asarray([p['f0'] for p in fit_param])

    #retrive calibrations
    calibrations = [(1./tx_info['ampl'][i])*USRP_calibration/(10**((USRP_power + tx_info['gain'])/20.)) for i in range(len(tx_info['ampl']))]


    #do averages
    print_debug("Averaging...")
    if points is None:
        noise_points = np.asarray([
            np.mean(dataset) for dataset in noise_file['raw_data0'][ant]['data']
        ])
    else:
        decimation = int(np.shape(noise_file['raw_data0'][ant]['data'])[1]/noise_points)
        print_debug("Decimating %d"%decimation)
        noise_points = np.asarray([
            signal.decimate(dataset,decimation,ftype="fir") for dataset in noise_file['raw_data0'][ant]['data']

        ])

    title = "Diagnostic plot: overlaying averaged noise acquisition and VNA traces"
    #plot
    print_debug("Plotting...")
    if backend == 'matplotlib':
        if fig_size is None:
            fig_size = (16, 10)
        diagnostic_folder_name = "Diagnostic_"+noise_filename.split(".")[0]
        try:
            os.mkdir(diagnostic_folder_name)
        except OSError:
            pass


        os.chdir(diagnostic_folder_name)
        for i in range(len(fit_data)):
            #Plot single channel data
            fig = pl.figure()
            fig.set_size_inches(fig_size[0], fig_size[1])
            ax = fig.add_subplot(111)

            label = "Channel %.2f MHz"%(fit_param[i]['f0'])
            current_color = get_color(i)
            edge_color = 'k'
            if current_color == 'black': edge_color = 'grey'

            original_data = fit_data[i]['original']

            ax.plot(original_data.real, original_data.imag , color = current_color, label = label, alpha = 0.4)
            ax.scatter(noise_points[i].real  * calibrations[i], noise_points[i].imag  * calibrations[i], color = current_color, edgecolors=edge_color,linewidth=2,s = 110)
            fig.suptitle(title)
            ax.set_aspect('equal','datalim')
            ax.legend()
            ax.grid()
            fig.savefig("diagnostic_channel_%d_IQ.png"%i)
            pl.close(fig)


            #plot phase and magnitude
            fig = pl.figure()
            fig.set_size_inches(fig_size[0], fig_size[1])
            ax = fig.add_subplot(111)
            ax.plot(fit_data[i]["frequency"],db(np.abs(original_data)),  label = "VNA data",  color = current_color)
            if points is not None:
                freq_data = [tx_info["rf"] + tx_info["freq"][i] for tt in range(len(noise_points[i]))]
                magdata = db(np.abs(noise_points[i]) *calibrations[i])
                diff_mag = np.mean(magdata) - vrms2dbm(np.abs(original_data)[find_nearest(fit_data[i]["frequency"], freq_data[0])])
            else:
                freq_data = [tx_info["rf"] + tx_info["freq"][i]]
                magdata = [db(np.abs(noise_points[i]) * calibrations[i])]
                diff_mag = magdata[0] - db(np.abs(original_data)[find_nearest(fit_data[i]["frequency"], freq_data[0])])
            ax.scatter(freq_data, magdata , label = "Averaged noise data", color = current_color, edgecolors=edge_color,linewidth=2,s = 110)
            ax.legend()
            ax.grid()
            fig.suptitle(label+"\n average discrepancy: %.2fdB"%diff_mag)
            fig.savefig("diagnostic_channel_%d_mag.png"%i)
            pl.close(fig)

        os.chdir("..")




    elif backend == 'plotly':
        pass
    else:
        err_msg = "%s backend not implemented in diagnostic function" % str(backend)
        print_error(err_msg)
        raise ValueError(err_msg)

    return

def calculate_NEF_spectra():
    #copied from calculate_spec
    return

def get_NEF_spec():
    return

def plot_NEF_spectra():
    #plot_spec
    return
