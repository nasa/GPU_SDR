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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

#needed to print the data acquisition process
import progressbar

#import submodules
from USRP_low_level import *
from USRP_connections import *
from USRP_plotting import *
from USRP_files import *
from USRP_data_analysis import *
from USRP_delay import *

def Dual_VNA(start_f_A, last_f_A, start_f_B, last_f_B, measure_t, n_points, tx_gain_A, tx_gain_B, Rate = None, decimation = True, RF_A = None, RF_B = None,
               Device = None, output_filename = None, Multitone_compensation_A = None, Multitone_compensation_B = None, Iterations = 1, verbose = False, **kwargs):

    '''
    Perform a VNA scan using a two different frontens of a single USRP device.

    Arguments:
        - start_f_A: frequency in Hz where to start scanning for frontend A (absolute if RF is not given, relative to RF otherwise).
        - last_f_A: frequency in Hz where to stop scanning for frontend A (absolute if RF is not given, relative to RF otherwise).
        - start_f_B: frequency in Hz where to start scanning for frontend B (absolute if RF is not given, relative to RF otherwise).
        - last_f_B: frequency in Hz where to stop scanning for frontend B (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - n_points: number of points to use in the VNA scan.
        - tx_gain_A: transmission amplifier gain for frontend A.
        - tx_gain_B: transmission amplifier gain for frontend B.
        - Rate: Optional parameter to control the scan rate. Default is calculate from start_f an last_f args.
        - decimation: if True the decimation of the signal will occur on-server. Default is True.
        - RF_A: central up/down mixing frequency for frontend A. Default is deducted by other arguments.
        - RF_B: central up/down mixing frequency for frontend B. Default is deducted by other arguments.
        - output_filename: eventual filename. default is datetime.
        - Device: the on-server device number to use. default is 0.
        - Multitone_compensation_A: integer representing the number of tones: compensate the amplitude of the signal to match a future multitones accuisition for frontend A.
        - Multitone_compensation_B: integer representing the number of tones: compensate the amplitude of the signal to match a future multitones accuisition for frontend B.
        - Iterations: by default a single VNA scan pass is performed.
        - verbose: if True outputs on terminal some diagnostic info. deafult is False.
        - keyword arguments: Each keyword argument will be interpreted as an attribute to add to the raw_data group of the h5 file.

    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''
    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY

    if measure_t <= 0:
        err_msg = "Cannot execute a VNA measure with "+str(measure_t)+"s duration."
        print_error(err_msg)
        raise ValueError(err_msg)
    if n_points <= 0:
        err_msg = "Cannot execute a VNA measure with "+str(n_points)+" points."
        print_error(err_msg)
        raise ValueError(err_msg)

    if RF_A == None:
        delta_f_A = np.abs(start_f_A - last_f_A)
        RF_A = delta_f_A/2.
        start_f_A -= RF_A
        last_f_A -= RF_A
        print "Setting RF (frontend A) central frequency to %.2f MHz"%(RF_A/1.e6)
    else:
        delta_f_A = max(start_f_A,last_f_A) - min(start_f_A,last_f_A)

    if delta_f_A > 1.6e8:
        err_msg = "Frequency range for the VNA scan (frontend A) is too large compared to maximum system bandwidth"
        print_error(err_msg)
        raise ValueError(err_msg)
    elif delta_f_A > 1e8:
        err_msg = "Frequency range for the VNA (frontend A) scan is too large compared to actual system bandwidth"
        print_error(err_msg)
        raise ValueError(err_msg)

    if RF_B == None:
        delta_f_B = np.abs(start_f_B - last_f_B)
        RF_B = delta_f_B/2.
        start_f_B -= RF_B
        last_f_B -= RF_B
        print "Setting RF (frontend B) central frequency to %.2f MHz"%(RF_B/1.e6)
    else:
        delta_f_B = max(start_f_B,last_f_B) - min(start_f_B,last_f_B)

    if delta_f_B > 1.6e8:
        err_msg = "Frequency range for the VNA scan (frontend B) is too large compared to maximum system bandwidth"
        print_error(err_msg)
        raise ValueError(err_msg)
    elif delta_f_B > 1e8:
        err_msg = "Frequency range for the VNA (frontend B) scan is too large compared to actual system bandwidth"
        print_error(err_msg)
        raise ValueError(err_msg)

    if not Device_chk(Device):
        err_msg = "Something is wrong with the device check in the VNA function."
        print_error(err_msg)
        raise ValueError(err_msg)

    if Multitone_compensation_A == None:
        Amplitude_A = 1.
    else:
        Amplitude_A = 1./Multitone_compensation_A

    if Multitone_compensation_B == None:
        Amplitude_B = 1.
    else:
        Amplitude_B = 1./Multitone_compensation_B

    if decimation:
        decimation = 1
    else:
        decimation = 0

    if Iterations <= 0:
        print_warning("Iterations can only be a bigger than 0 integer. Setting it to 1")
        Iterations = 1
    else:
        Iterations = int(Iterations)

    if Rate is None:
        Rate = 100e6

    try:
        delay = LINE_DELAY[str(int(Rate/1e6))]
        delay *= 1e-9
    except KeyError:
        print_warning("Cannot find associated line delay for a rate of %d Msps. Performance may be negatively affected"%(int(rate/1e6)))
        delay = 0

    if output_filename is None:
        output_filename = "USRP_VNA_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    print_debug("Writing VNA data on file \'%s\'"%output_filename)

    number_of_samples = Rate* measure_t*Iterations

    vna_command = global_parameter()
    TX_frontend_A = "A_TXRX"
    RX_frontend_A = "A_RX2"
    TX_frontend_B = "B_TXRX"
    RX_frontend_B = "B_RX2"
    vna_command.set(TX_frontend_A,"mode", "TX")
    vna_command.set(TX_frontend_A,"buffer_len", 1e6)
    vna_command.set(TX_frontend_A,"gain", tx_gain_A)
    vna_command.set(TX_frontend_A,"delay", 1)
    vna_command.set(TX_frontend_A,"samples", number_of_samples)
    vna_command.set(TX_frontend_A,"rate", Rate)
    vna_command.set(TX_frontend_A,"bw", 2*Rate)

    vna_command.set(TX_frontend_A,"wave_type", ["CHIRP"])
    vna_command.set(TX_frontend_A,"ampl", [Amplitude_A])
    vna_command.set(TX_frontend_A,"freq", [start_f_A])
    vna_command.set(TX_frontend_A,"chirp_f", [last_f_A])
    vna_command.set(TX_frontend_A,"swipe_s", [n_points])
    vna_command.set(TX_frontend_A,"chirp_t", [measure_t])
    vna_command.set(TX_frontend_A,"rf", RF_A)

    vna_command.set(RX_frontend_A,"mode", "RX")
    vna_command.set(RX_frontend_A,"buffer_len", 1e6)
    vna_command.set(RX_frontend_A,"gain", 0)
    vna_command.set(RX_frontend_A,"delay", 1+delay)
    vna_command.set(RX_frontend_A,"samples", number_of_samples)
    vna_command.set(RX_frontend_A,"rate", Rate)
    vna_command.set(RX_frontend_A,"bw", 2*Rate)

    vna_command.set(RX_frontend_A,"wave_type", ["CHIRP"])
    vna_command.set(RX_frontend_A,"ampl", [Amplitude_A])
    vna_command.set(RX_frontend_A,"freq", [start_f_A])
    vna_command.set(RX_frontend_A,"chirp_f", [last_f_A])
    vna_command.set(RX_frontend_A,"swipe_s", [n_points])
    vna_command.set(RX_frontend_A,"chirp_t", [measure_t])
    vna_command.set(RX_frontend_A,"rf", RF_A)
    vna_command.set(RX_frontend_A,"decim", decimation) # THIS only activate the decimation.

    vna_command.set(TX_frontend_B,"mode", "TX")
    vna_command.set(TX_frontend_B,"buffer_len", 1e6)
    vna_command.set(TX_frontend_B,"gain", tx_gain_A)
    vna_command.set(TX_frontend_B,"delay", 1)
    vna_command.set(TX_frontend_B,"samples", number_of_samples)
    vna_command.set(TX_frontend_B,"rate", Rate)
    vna_command.set(TX_frontend_B,"bw", 2*Rate)

    vna_command.set(TX_frontend_B,"wave_type", ["CHIRP"])
    vna_command.set(TX_frontend_B,"ampl", [Amplitude_B])
    vna_command.set(TX_frontend_B,"freq", [start_f_B])
    vna_command.set(TX_frontend_B,"chirp_f", [last_f_B])
    vna_command.set(TX_frontend_B,"swipe_s", [n_points])
    vna_command.set(TX_frontend_B,"chirp_t", [measure_t])
    vna_command.set(TX_frontend_B,"rf", RF_B)

    vna_command.set(RX_frontend_B,"mode", "RX")
    vna_command.set(RX_frontend_B,"buffer_len", 1e6)
    vna_command.set(RX_frontend_B,"gain", 0)
    vna_command.set(RX_frontend_B,"delay", 1+delay)
    vna_command.set(RX_frontend_B,"samples", number_of_samples)
    vna_command.set(RX_frontend_B,"rate", Rate)
    vna_command.set(RX_frontend_B,"bw", 2*Rate)

    vna_command.set(RX_frontend_B,"wave_type", ["CHIRP"])
    vna_command.set(RX_frontend_B,"ampl", [Amplitude_B])
    vna_command.set(RX_frontend_B,"freq", [start_f_B])
    vna_command.set(RX_frontend_B,"chirp_f", [last_f_B])
    vna_command.set(RX_frontend_B,"swipe_s", [n_points])
    vna_command.set(RX_frontend_B,"chirp_t", [measure_t])
    vna_command.set(RX_frontend_B,"rf", RF_B)
    vna_command.set(RX_frontend_B,"decim", decimation) # THIS only activate the decimation.
    if vna_command.self_check():
        if(verbose):
            print "VNA command succesfully checked"
            vna_command.pprint()

        Async_send(vna_command.to_json())

    else:
        print_warning("Something went wrong with the setting of VNA command.")
        return ""

    if decimation:
        expected_samples = Iterations * n_points
    else:
        expected_samples = number_of_samples

    Packets_to_file(
        parameters = vna_command,
        timeout = None,
        filename = output_filename,
        dpc_expected = expected_samples,
        meas_type = "VNA", **kwargs
    )

    print_debug("VNA acquisition terminated.")

    return output_filename



def Single_VNA(start_f, last_f, measure_t, n_points, tx_gain, Rate = None, decimation = True, RF = None, Front_end = None,
               Device = None, output_filename = None, Multitone_compensation = None, Iterations = 1, verbose = False, **kwargs):

    '''
    Perform a VNA scan using a single frontend of a single USRP device.

    Arguments:
        - start_f: frequency in Hz where to start scanning (absolute if RF is not given, relative to RF otherwise).
        - last_f: frequency in Hz where to stop scanning (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - n_points: number of points to use in the VNA scan.
        - tx_gain: transmission amplifier gain.
        - Rate: Optional parameter to control the scan rate. Default is calculate from start_f an last_f args.
        - decimation: if True the decimation of the signal will occur on-server. Default is True.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - output_filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - Multitone_compensation: integer representing the number of tones: compensate the amplitude of the signal to match a future multitones accuisition.
        - Iterations: by default a single VNA scan pass is performed.
        - verbose: if True outputs on terminal some diagnostic info. deafult is False.
        - keyword arguments: Each keyword argument will be interpreted as an attribute to add to the raw_data group of the h5 file.

    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''

    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY

    if measure_t <= 0:
        err_msg = "Cannot execute a VNA measure with "+str(measure_t)+"s duration."
        print_error(err_msg)
        raise ValueError(err_msg)
    if n_points <= 0:
        err_msg = "Cannot execute a VNA measure with "+str(n_points)+" points."
        print_error(err_msg)
        raise ValueError(err_msg)

    if RF == None:
        delta_f = np.abs(start_f - last_f)
        RF = delta_f/2.
        start_f -= RF
        last_f -= RF
        print "Setting RF central frequency to %.2f MHz"%(RF/1.e6)
    else:
        delta_f = max(start_f,last_f) - min(start_f,last_f)

    if delta_f > 1.6e8:
        err_msg = "Frequency range for the VNA scan may be too large compared to maximum system bandwidth"
        print_warning(err_msg)
        #raise ValueError(err_msg)
    elif delta_f > 1e8:
        err_msg = "Frequency range for the VNA scan may be too large compared to actual system bandwidth"
        print_warning(err_msg)
        #raise ValueError(err_msg)

    if not Device_chk(Device):
        err_msg = "Something is wrong with the device check in the VNA function."
        print_error(err_msg)
        raise ValueError(err_msg)

    if Front_end is None:
        Front_end = 'A'

    if not Front_end_chk(Front_end):
        err_msg = "Cannot detect front_end: "+str(Front_end)
        print_error(err_msg)
        raise ValueError(err_msg)
    else:
        TX_frontend = Front_end+"_TXRX"
        RX_frontend = Front_end+"_RX2"

    if Multitone_compensation == None:
        Amplitude = 1.
    else:
        Amplitude = 1./Multitone_compensation

    if decimation:
        decimation = 1
    else:
        decimation = 0

    if Iterations <= 0:
        print_warning("Iterations can only be a bigger than 0 integer. Setting it to 1")
        Iterations = 1
    else:
        Iterations = int(Iterations)

    if Rate is None:
        Rate = 100e6

    try:
        delay = LINE_DELAY[str(int(Rate/1e6))]
        delay *= 1e-9
    except KeyError:
        print_warning("Cannot find associated line delay for a rate of %d Msps. Performance may be negatively affected"%(int(Rate/1e6)))
        delay = 0


    if output_filename is None:
        output_filename = "USRP_VNA_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    print_debug("Writing VNA data on file \'%s\'"%output_filename)

    number_of_samples = Rate* measure_t*Iterations

    vna_command = global_parameter()

    vna_command.set(TX_frontend,"mode", "TX")
    vna_command.set(TX_frontend,"buffer_len", 1e6)
    vna_command.set(TX_frontend,"gain", tx_gain)
    vna_command.set(TX_frontend,"delay", 1)
    vna_command.set(TX_frontend,"samples", number_of_samples)
    vna_command.set(TX_frontend,"rate", Rate)
    vna_command.set(TX_frontend,"bw", 2*Rate)

    vna_command.set(TX_frontend,"wave_type", ["CHIRP"])
    vna_command.set(TX_frontend,"ampl", [Amplitude])
    vna_command.set(TX_frontend,"freq", [start_f])
    vna_command.set(TX_frontend,"chirp_f", [last_f])
    vna_command.set(TX_frontend,"swipe_s", [n_points])
    vna_command.set(TX_frontend,"chirp_t", [measure_t])
    vna_command.set(TX_frontend,"rf", RF)

    vna_command.set(RX_frontend,"mode", "RX")
    vna_command.set(RX_frontend,"buffer_len", 1e6)
    vna_command.set(RX_frontend,"gain", 0)
    vna_command.set(RX_frontend,"delay", 1+delay)
    vna_command.set(RX_frontend,"samples", number_of_samples)
    vna_command.set(RX_frontend,"rate", Rate)
    vna_command.set(RX_frontend,"bw", 2*Rate)

    vna_command.set(RX_frontend,"wave_type", ["CHIRP"])
    vna_command.set(RX_frontend,"ampl", [Amplitude])
    vna_command.set(RX_frontend,"freq", [start_f])
    vna_command.set(RX_frontend,"chirp_f", [last_f])
    vna_command.set(RX_frontend,"swipe_s", [n_points])
    vna_command.set(RX_frontend,"chirp_t", [measure_t])
    vna_command.set(RX_frontend,"rf", RF)
    vna_command.set(RX_frontend,"decim", decimation) # THIS only activate the decimation.

    if vna_command.self_check():
        if(verbose):
            print "VNA command succesfully checked"
            vna_command.pprint()

        Async_send(vna_command.to_json())

    else:
        print_warning("Something went wrong with the setting of VNA command.")
        return ""

    if decimation:
        expected_samples = Iterations * n_points
    else:
        expected_samples = number_of_samples

    Packets_to_file(
        parameters = vna_command,
        timeout = None,
        filename = output_filename,
        dpc_expected = expected_samples,
        meas_type = "VNA", **kwargs
    )

    print_debug("VNA acquisition terminated.")

    return output_filename


def VNA_timestream_analysis(filename, usrp_number = 0):
    '''
    Open a H5 file containing data collected with the function single_VNA() and analyze them as multiple VNA scans, one per iteration.

    :param filename: string containing the name of the H5 file.
    :param usrp_number: usrp server number.
    '''

    usrp_number = int(usrp_number)

    try:
        filename = format_filename(filename)
    except:
        print_error("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")
        raise ValueError("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")

    print("Analyzing VNA file \'%s\'..."%filename)

    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)

    front_ends = ["A_RX2", "B_RX2"]
    active_front_ends = []
    info = []
    for ant in front_ends:
        if (parameters.parameters[ant]['mode'] == "RX") and (parameters.parameters[ant]['wave_type'][0] == "CHIRP"):
            info.append(parameters.parameters[ant])
            active_front_ends.append(ant)

    # Nedded for the calibration calculations
    front_ends_tx= ["A_TXRX", "B_TXRX"]
    gains = []
    ampls = []
    for ant in front_ends_tx:
        if parameters.parameters[ant]['mode'] == "TX" and parameters.parameters[ant]['wave_type'][0] == "CHIRP":
            gains.append(parameters.parameters[ant]['gain'])
            ampls.append(parameters.parameters[ant]['ampl'][0])

    print_debug("Found %d active frontends"%len(info))

    freq_axis = np.asarray([])
    S21_axis = np.asarray([])
    length = []
    calibration = []
    fr = 0

    for single_frontend in info:
        iterations = int((single_frontend['samples']/single_frontend['rate'])/single_frontend['chirp_t'][0])
        print_debug("Frontend \'%s\' has %d VNA iterations" % (front_ends[fr], iterations))

        #effective calibration
        calibration.append( (1./ampls[fr])*USRP_calibration/(10**((USRP_power + gains[fr])/20.)) )
        print_debug("Calculating calibration with %d dB gain and %.3f amplitude correction"%(gains[fr],ampls[fr]))

        if single_frontend['decim'] == 1:
            # Lock-in decimated case -> direct map.
            freq_axis_tmp = np.linspace(single_frontend['freq'][0],single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                        dtype = np.float64) + single_frontend['rf']

            S21_axis_tmp = np.split(openH5file(filename, front_end = active_front_ends[fr])[0], iterations)

            length.append(single_frontend['swipe_s'])

        elif single_frontend['decim'] > 1:
            # Over decimated case.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0]/single_frontend['decim'],
                                    dtype=np.float64) + single_frontend['rf']

            S21_axis_tmp = np.split(openH5file(filename, front_end=active_front_ends[fr])[0], iterations)

            length.append(single_frontend['swipe_s'][0]/single_frontend['decim'])

        else:
            # Undecimated case. Decimation has to happen here.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                                    dtype=np.float64) + single_frontend['rf']
            S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end = active_front_ends[fr])[0], single_frontend['swipe_s'][0]), axis = 1)

            length.append(single_frontend['swipe_s'][0])

        if fr == 0:
            freq_axis = freq_axis_tmp
            S21_axis = S21_axis_tmp
        else:
            freq_axis = np.concatenate((freq_axis, freq_axis_tmp),axis = 1)
            S21_axis = np.concatenate((S21_axis, S21_axis_tmp), axis = 1)
        fr+=1



    try:
        f = h5py.File(filename, 'r+')
    except IOError as msg:
        print_error("Cannot open "+str(filename)+" file in VNA_timestream_analysis function"+ str(msg))
        raise ValueError("Cannot open "+str(filename)+" file in VNA_timestream_analysis function: "+ str(msg))

    try:
        vna_grp = f.create_group("VNA_dynamic_%d"%(usrp_number))
    except ValueError:
        print_warning("Overwriting VNA group")
        del f["VNA_dynamic_%d"%(usrp_number)]
        vna_grp = f.create_group("VNA_dynamic_%d"%(usrp_number))

    vna_grp.attrs.create("scan_lengths", length)
    vna_grp.attrs.create("calibration", calibration)

    vna_grp.create_dataset("frequency", data = freq_axis, dtype=np.float64)
    vna_grp.create_dataset("S21", data = S21_axis, dtype=np.complex128)

    f.close()

    print_debug("Analysis of file \'%s\' concluded."%filename)

# def get_dynamic_VNA_data(filename):

def VNA_timestream_plot(filename, backend='matplotlib', mode = 'magnitude', unwrap_phase=False, verbose=False, output_filename=None, **kwargs):
    '''
    Plot the VNA timestream analysis result.

    :param filename: string containing the name of the H5 file.
    :param backend: the backend used to plot the data.
    :param mode: the mode used to plot the data:
        * magnitude: magnitude of the data.
        * phase: S21 phase.
        * df: magnitude of the derivarive of S21 in the frequency direction.
        * dt : magnitude of the derivarive of S21 in the time direction.
    '''
    print("Plotting Dynamic VNA(s)...")

    # try:
    #     html_output = kwargs['html']
    # except KeyError:
    #     html_output = False
    #

    try:
        att = kwargs['att']
    except KeyError:
        att = None

    try:
        auto_open = kwargs['auto_open']
    except KeyError:
        auto_open = True

    try:
        fig_size = kwargs['figsize']
    except KeyError:
        fig_size = None

    try:
        add_info_labels = kwargs['add_info']
        if add_info_labels is not None:
            add_info_labels = str(add_info_labels)
    except KeyError:
        add_info_labels = None
    except:
        print_warning("Cannot add info labels to dynamic VNA plot")
        add_info_labels = None


    final_filename = ""
    reso_axes = []

    if verbose: print_debug("Plotting VNA from file \'%s\'"%filename)
    freq_tmp, S21_tmp = get_dynamic_VNA_data(filename)

    freq_axes = np.array(freq_tmp)
    S21_axes = np.array(S21_tmp)

    # Set up the mesh grid for the dataset
    num_vnas = S21_axes.shape[0]
    y_axis = np.arange(num_vnas)
    X, Y = np.meshgrid(freq_axes/1e6, y_axis)

    del S21_tmp
    del freq_tmp

    if output_filename is None:
        output_filename = "Dynamic_VNA"
        output_filename+="_"+get_timestamp()
    fit_label = ""

    info = get_rx_info(filename, ant=None)
    if backend == "matplotlib":
        if verbose: print_debug("Using matplotlib backend...")

        if fig_size is None:
            fig_size = (25, 10)
        fig, ax = pl.subplots(figsize=fig_size)

        mag = vrms2dbm(np.abs(S21_axes))

        if unwrap_phase:
            phase = linear_phase(np.angle(S21_axes))
        else:
            phase = np.angle(S21_axes)

        label = filename
        resolution = freq_axes[1] - freq_axes[0]

        if resolution < 1e3:
            label += "\nResolution %d Hz" % int(resolution)
        else:
            label += "\nResolution %.1f kHz" % (resolution/1e3)

        readout_power = get_readout_power(filename, 0)

        if att is None:
            label+="    Readout power: %.2f dBm"%readout_power
        else:
            label += "    On-chip power: %.2f dBm"%(readout_power-att)

        if add_info_labels is not None:
            label += "\n"+str(add_info_labels[i])

        extent = (freq_axes[0], freq_axes[-1], num_vnas*info['chirp_t'][0], 0)
        norm = Normalize(vmin=mag.min(), vmax=mag.max())

        img = ax.imshow(mag, cmap=cm.viridis, origin='upper', extent=extent, aspect='auto', interpolation='none')
        #img.set_clim(-52, -40)

        ax.set_title(label)
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("Frequency [Hz]")

        formatter0 = EngFormatter(unit='Hz')
        ax.xaxis.set_major_formatter(formatter0)

        fig.colorbar(img, label='|S21|')
        final_filename = output_filename+".png"
        print final_filename
        pl.savefig(final_filename, bbox_inches="tight")


def VNA_analysis(filename, usrp_number = 0):
    '''
    Open a H5 file containing data collected with the function single_VNA() and analyze them as a VNA scan.
    Write the results in a corresponding VNA# group in the root of the H5 file.

    :param filename: string containing the name of the H5 file.
    :param usrp_number: usrp server number.

    '''

    usrp_number = int(usrp_number)

    try:
        filename = format_filename(filename)
    except:
        print_error("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")
        raise ValueError("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")

    print("Anlyzing VNA file \'%s\'..."%filename)

    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)

    front_ends = ["A_RX2", "B_RX2"]
    active_front_ends = []
    info = []
    for ant in front_ends:
        if (parameters.parameters[ant]['mode'] == "RX") and (parameters.parameters[ant]['wave_type'][0] == "CHIRP"):
            info.append(parameters.parameters[ant])
            active_front_ends.append(ant)

    # Nedded for the calibration calculations
    front_ends_tx= ["A_TXRX", "B_TXRX"]
    gains = []
    ampls = []
    for ant in front_ends_tx:
        if parameters.parameters[ant]['mode'] == "TX" and parameters.parameters[ant]['wave_type'][0] == "CHIRP":
            gains.append(parameters.parameters[ant]['gain'])
            ampls.append(parameters.parameters[ant]['ampl'][0])

    print_debug("Found %d active frontends"%len(info))

    freq_axis = np.asarray([],dtype = np.float64)
    S21_axis = np.asarray([],dtype = np.complex128)
    length = []
    calibration = []
    fr = 0
    for single_frontend in info:
        iterations = int((single_frontend['samples']/single_frontend['rate'])/single_frontend['chirp_t'][0])
        print_debug("Frontend \'%s\' has %d VNA iterations" % (front_ends[fr], iterations))

        #effective calibration
        calibration.append( (1./ampls[fr])*USRP_calibration/(10**((USRP_power + gains[fr])/20.)) )
        print_debug("Calculating calibration with %d dB gain and %.3f amplitude correction"%(gains[fr],ampls[fr]))

        if single_frontend['decim'] == 1:
            # Lock-in decimated case -> direct map.
            freq_axis_tmp = np.linspace(single_frontend['freq'][0],single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                        dtype = np.float64) + single_frontend['rf']
            if iterations>1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end = active_front_ends[fr])[0], iterations),axis = 0)
            else:
                S21_axis_tmp  = openH5file(filename, front_end = active_front_ends[fr])[0]

            length.append(single_frontend['swipe_s'])

        elif single_frontend['decim'] > 1:
            # Over decimated case.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0]/single_frontend['decim'],
                                    dtype=np.float64) + single_frontend['rf']
            if iterations > 1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end=active_front_ends[fr])[0], iterations), axis=0)
            else:
                S21_axis_tmp = openH5file(filename, front_end=active_front_ends[fr])[0]
            length.append(single_frontend['swipe_s'][0]/single_frontend['decim'])

        else:
            # Undecimated case. Decimation has to happen here.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                                    dtype=np.float64) + single_frontend['rf']
            if iterations>1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end = active_front_ends[fr])[0], iterations), axis = 0)
            else:
                S21_axis_tmp  = openH5file(filename, front_end = active_front_ends[fr])[0]

            S21_axis_tmp  = np.mean(np.split(S21_axis_tmp, single_frontend['swipe_s'][0]), axis = 1)
            length.append(single_frontend['swipe_s'][0])

        fr+=1

        freq_axis = np.concatenate((freq_axis, freq_axis_tmp))
        S21_axis = np.concatenate((S21_axis, S21_axis_tmp))

    try:
        f = h5py.File(filename, 'r+')
    except IOError as msg:
        print_error("Cannot open "+str(filename)+" file in Single_VNA_analysis function"+ str(msg))
        raise ValueError("Cannot open "+str(filename)+" file in Single_VNA_analysis function: "+ str(msg))

    try:
        vna_grp = f.create_group("VNA_%d"%(usrp_number))
    except ValueError:
        print_warning("Overwriting VNA group")
        del f["VNA_%d"%(usrp_number)]
        vna_grp = f.create_group("VNA_%d"%(usrp_number))

    vna_grp.attrs.create("scan_lengths", length)
    vna_grp.attrs.create("calibration", calibration)

    vna_grp.create_dataset("frequency", data = freq_axis, dtype=np.float64)
    vna_grp.create_dataset("S21", data = S21_axis, dtype=np.complex128)

    f.close()

    print_debug("Analysis of file \'%s\' concluded."%filename)


def plot_VNA(filenames, backend = "matplotlib", output_filename = None, unwrap_phase = True, verbose = False, **kwargs):
    '''
    Plot the VNA data from various files.

    :param filenames: list of strings containing the filenames to be plotted.
    :param backend: "matplotlib", "plotly" are supported.
    :param output_filename: filename of the output figure without extension. Default is VNA(_compare)_timestamp.xxx.
    :param unwrap_phase: if False the angle of S21 is not unwrapped.
    :param verbose: print some debug line.
    :param kwargs:
        - figsize=(xx,yy) inches for matplotlib backends.
        - add_info = ["..","..",".."] fore commenting each file in the legend.
        - html only to return a html text instead of nothing in case of plotly backend.
        - title is a string containing the title.
        - att: external attenuation: changes the label in the plot from readout power to on-chip power if given.
        - auto_open: for plotly backend. If True opens the plot in a browser. Default is True.

   :return The filename of the file just created. if kwargs['html'] is True returns the html of the file instead.

    '''

    print("Plotting VNA(s)...")

    try:
        html_output = kwargs['html']
    except KeyError:
        html_output = False

    try:
        att = kwargs['att']
    except KeyError:
        att = None

    try:
        auto_open = kwargs['auto_open']
    except KeyError:
        auto_open = True

    try:
        fig_size = kwargs['figsize']
    except KeyError:
        fig_size = None

    filenames = to_list_of_str(filenames)

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
            title = "VNA plot from file %s"%filenames[0]
        else:
            title = "VNA comparison plot"

    if len(filenames) == 0:
        err_msg = "File list empty, cannot plot VNA"
        print_error(err_msg)
        raise ValueError(err_msg)

    freq_axes = []
    S21_axes = []
    final_filename = ""
    reso_axes = []
    for filename in filenames:
        if verbose: print_debug("Plotting VNA from file \'%s\'"%filename)
        freq_tmp, S21_tmp = get_VNA_data(filename)

        freq_axes.append(freq_tmp)
        S21_axes.append(S21_tmp)
        reso_axes.append( get_init_peaks(filename, verbose = verbose))

    del S21_tmp
    del freq_tmp

    if output_filename is None:
        output_filename = "VNA"
        if len(filenames)>1:
            output_filename+="_compare"
        output_filename+="_"+get_timestamp()

    fit_label = ""

    if backend == "matplotlib":
        if verbose: print_debug("Using matplotlib backend...")

        fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)

        if fig_size is None:
            fig_size = (16, 10)

        fig.set_size_inches(fig_size[0], fig_size[1])

        fig.suptitle(title)

        for i in range(len(filenames)):

            mag = vrms2dbm(np.abs(S21_axes[i]))

            if unwrap_phase:
                phase = linear_phase(np.angle(S21_axes[i]))
            else:
                phase = np.angle(S21_axes[i])

            label = filenames[i]
            resolution = freq_axes[i][1] - freq_axes[i][0]

            if resolution < 1e3:
                label += "\nResolution %d Hz" % int(resolution)
            else:
                label += "\nResolution %.1f kHz" % (resolution/1e3)

            readout_power = get_readout_power(filenames[i], 0)

            if att is None:
                label+="\nReadout power: %.2f dBm"%readout_power
            else:
                label += "\nOn-chip power: %.2f dBm"%(readout_power-att)

            if add_info_labels is not None:
                label += "\n"+str(add_info_labels[i])

            color = get_color(i)
            if color == 'black':
                other_color = 'red'
            else:
                other_color = 'black'

            ax[0].plot(freq_axes[i], mag, color = color, label = label)
            ax[1].plot(freq_axes[i], phase, color = color)

            # if there are initialized resoantor in the file...
            if len(reso_axes[i]) > 0:
                x_points = []
                mag_y_points = []
                pha_y_points = []
                for point in reso_axes[i]:
                    index = find_nearest(freq_axes[i], point)
                    x_points.append(freq_axes[i][index])
                    mag_y_points.append(mag[index])
                    pha_y_points.append(phase[index])
                if fit_label is not None:
                    fit_label = "Fit initialization"
                ax[0].scatter(x_points, mag_y_points, s=80, facecolors='none', edgecolors=other_color, label = fit_label)
                ax[1].scatter(x_points, pha_y_points, s=80, facecolors='none', edgecolors=other_color)
                fit_label = None

        ax[0].set_ylabel("Magnitude [dB]")
        ax[1].set_ylabel("Phase [Rad]")
        ax[1].set_xlabel("Frequency [Hz]")

        formatter0 = EngFormatter(unit='Hz')
        ax[1].xaxis.set_major_formatter(formatter0)

        ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax[0].grid()
        ax[1].grid()
        final_filename = output_filename+".png"
        pl.savefig(final_filename, bbox_inches="tight")

    elif backend == "plotly":

        if verbose: print_debug("Using plotly backend")

        fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.003)

        fig['layout'].update(title=title)
        fig['layout'].update(autosize=True)
        fig['layout']['xaxis1'].update(title='Frequency [Hz]')
        fig['layout']['xaxis1'].update(exponentformat='SI')
        #fig['layout']['xaxis1'].update(ticksuffix='Hz') # does not work well with logscale

        fig['layout']['yaxis1'].update(title='Magnitude [dB]')
        fig['layout']['yaxis2'].update(title='Phase [Rad]')

        for i in range(len(filenames)):

            color = get_color(i)
            if color == 'black':
                other_color = 'red'
            else:
                other_color = 'black'
            label = filenames[i]
            resolution = freq_axes[i][1] - freq_axes[i][0]

            if resolution < 1e3:
                label += "<br>Resolution %d Hz" % int(resolution)
            else:
                label += "<br>Resolution %.1f kHz" % (resolution / 1e3)

            readout_power = get_readout_power(filenames[i], 0)

            if att is None:
                label += "<br>Readout power: %.2f dBm" % readout_power
            else:
                label += "<br>On-chip power: %.2f dBm" % (readout_power - att)

            if add_info_labels is not None:
                label += "<br>" + str(add_info_labels[i])


            mag = vrms2dbm(np.abs(S21_axes[i]))

            if unwrap_phase:
                phase = linear_phase(np.angle(S21_axes[i]))
            else:
                phase = np.angle(S21_axes[i])

            traceM = go.Scattergl(
                x=freq_axes[i],
                y=mag,
                name=label,
                mode='lines',
                line=dict(color=color),
                legendgroup=filenames[i]
            )

            traceP = go.Scattergl(
                x=freq_axes[i],
                y=phase,
                name="Phase",
                mode='lines',
                line=dict(color=color),
                legendgroup=filenames[i],
                showlegend=False
            )

            fig.append_trace(traceM, 1, 1)
            fig.append_trace(traceP, 2, 1)

            if len(reso_axes[i]) > 0:
                x_points = []
                mag_y_points = []
                pha_y_points = []
                for point in reso_axes[i]:
                    index = find_nearest(freq_axes[i], point)
                    x_points.append(freq_axes[i][index])
                    mag_y_points.append(mag[index])
                    pha_y_points.append(phase[index])


                fit_label = "Fit initialization"
                traceP_fit = go.Scattergl(
                    x=x_points,
                    y=pha_y_points,
                    name=fit_label,
                    mode='markers',
                    marker = dict(
                        color = 'rgba(0, 0, 0, 0.)',
                        size = 10,
                        line = dict(
                            color = other_color,
                            width = 2
                        )
                    ),
                    legendgroup=filenames[i],
                    showlegend=False
                )
                traceM_fit = go.Scattergl(
                    x=x_points,
                    y=mag_y_points,
                    name=fit_label,
                    mode='markers',
                    marker = dict(
                        color = 'rgba(0, 0, 0, 0.)',
                        size = 10,
                        line = dict(
                            color = other_color,
                            width = 2
                        )
                    ),
                    legendgroup=filenames[i],
                    showlegend=True
                )

                fig.append_trace(traceM_fit, 1, 1)
                fig.append_trace(traceP_fit, 2, 1)

        final_filename = output_filename + ".html"
        plotly.offline.plot(fig, filename=final_filename, auto_open=auto_open)

    else:
        err_msg = "Backend \'%s\' is not implemented. Cannot plot VNA"%backend
        print_error(err_msg)
        raise ValueError(err_msg)
    print("VNA plotting complete")
    return final_filename
