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
from USRP_low_level import *
from USRP_connections import *
from USRP_plotting import *
from USRP_files import *
from USRP_data_analysis import *
from USRP_delay import *


def Single_VNA(start_f, last_f, measure_t, n_points, tx_gain, Rate = None, decimation = True, RF = None, Front_end = None,
               Device = None, output_filename = None, Multitone_compensation = None, Iterations = 1, verbose = False, **kwargs):

    '''
    Perform a VNA scan.
    
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
        
    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''

    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY
    
    if measure_t <= 0:
        print_error("Cannot execute a VNA measure with "+str(measure_t)+"s duration.")
        return ""
    if n_points <= 0:
        print_error("Cannot execute a VNA measure with "+str(n_points)+" points.")
        return ""
    if RF == None:
        delta_f = np.abs(start_f - last_f)
        RF = delta_f/2.
        start_f -= RF
        last_f -= RF
        print "Setting RF central frequency to %.2f MHz"%(RF/1.e6)
    else:
        delta_f = max(start_f,last_f) - min(start_f,last_f)
        
    if delta_f > 1.6e8:
        print_error("Frequency range for the VNA scan is too large compared to maximum system bandwidth")
        return ""
    elif delta_f > 1e8:
        print_error("Frequency range for the VNA scan is too large compared to actual system bandwidth")
        return ""
    
    if not Device_chk(Device):
        return ""
        
    if not Front_end_chk(Front_end):
        return ""
        
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
        print_warning("Cannot find associated line delay for a rate of %d Msps. Performance may be negatively affected"%(int(rate/1e6)))
        delay = 0


    if output_filename is None:
        output_filename = "USRP_VNA_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    print_debug("Writing VNA data on file \'%s\'"%output_filename)

    number_of_samples = Rate* measure_t*Iterations
        
    vna_command = global_parameter()
    
    vna_command.set("A_TXRX","mode", "TX")
    vna_command.set("A_TXRX","buffer_len", 1e6)
    vna_command.set("A_TXRX","gain", tx_gain)
    vna_command.set("A_TXRX","delay", 1)
    vna_command.set("A_TXRX","samples", number_of_samples)
    vna_command.set("A_TXRX","rate", Rate)
    vna_command.set("A_TXRX","bw", 2*Rate)
    
    vna_command.set("A_TXRX","wave_type", ["CHIRP"])
    vna_command.set("A_TXRX","ampl", [Amplitude])
    vna_command.set("A_TXRX","freq", [start_f])
    vna_command.set("A_TXRX","chirp_f", [last_f])
    vna_command.set("A_TXRX","swipe_s", [n_points])
    vna_command.set("A_TXRX","chirp_t", [measure_t])
    vna_command.set("A_TXRX","rf", RF)
    
    vna_command.set("A_RX2","mode", "RX")
    vna_command.set("A_RX2","buffer_len", 1e6)
    vna_command.set("A_RX2","gain", 0)
    vna_command.set("A_RX2","delay", 1+delay)
    vna_command.set("A_RX2","samples", number_of_samples)
    vna_command.set("A_RX2","rate", Rate)
    vna_command.set("A_RX2","bw", 2*Rate)
    
    vna_command.set("A_RX2","wave_type", ["CHIRP"])
    vna_command.set("A_RX2","ampl", [Amplitude])
    vna_command.set("A_RX2","freq", [start_f])
    vna_command.set("A_RX2","chirp_f", [last_f])
    vna_command.set("A_RX2","swipe_s", [n_points])
    vna_command.set("A_RX2","chirp_t", [measure_t])
    vna_command.set("A_RX2","rf", RF)
    vna_command.set("A_RX2","decim", decimation) # THIS only activate the decimation.
    
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

def VNA_analysis(filename):
    '''
    Open a H5 file containing data collected with the function single_VNA() and analyze them as a VNA scan.
    Write the results in a corresponding VNA# group in the root of the H5 file.

    :param filename: string containing the name of the H5 file.

    '''


    try:
        filename = format_filename(filename)
    except:
        print_error("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")
        raise ValueError("Cannot interpret filename while opening a H5 file in Single_VNA_analysis function")

    print_debug("Anlyzing VNA file \'%s\'..."%filename)

    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)

    front_ends = ["A_RX2", "B_RX2"]
    info = []
    for ant in front_ends:
        if parameters.parameters[ant]['mode'] == "RX" and parameters.parameters[ant]['wave_type'][0] == "CHIRP":
            info.append(parameters.parameters[ant])

    print_debug("Found %d active frontends"%len(info))

    freq_axis = np.asarray([],dtype = np.float64)
    S21_axis = np.asarray([],dtype = np.complex128)
    length = []
    calibration = []
    fr = 0
    for single_frontend in info:
        iterations = int((single_frontend['samples']/single_frontend['rate'])/single_frontend['chirp_t'][0])
        print_debug("Frontend \'%s\' has %d VNA iterations" % (front_ends[fr], iterations))
        calibration.append( (1./single_frontend['ampl'][0])*USRP_calibration/10**((USRP_power + single_frontend['gain'])/20.) )
        if single_frontend['decim'] == 1:
            # Lock-in decimated case -> direct map.
            freq_axis_tmp = np.linspace(single_frontend['freq'][0],single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                        dtype = np.float64) + single_frontend['rf']
            if iterations>1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end = front_ends[fr])[0], iterations),axis = 0)
            else:
                S21_axis_tmp  = openH5file(filename, front_end = front_ends[fr])[0]

            length.append(single_frontend['swipe_s'])

        elif single_frontend['decim'] > 1:
            # Over decimated case.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0]/single_frontend['decim'],
                                    dtype=np.float64) + single_frontend['rf']
            if iterations > 1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end=front_ends[fr])[0], iterations), axis=0)
            else:
                S21_axis_tmp = openH5file(filename, front_end=front_ends[fr])[0]
            length.append(single_frontend['swipe_s'][0]/single_frontend['decim'])

        else:
            # Undecimated case. Decimation has to happen here.
            freq_axis_tmp  = np.linspace(single_frontend['freq'][0], single_frontend['chirp_f'][0], single_frontend['swipe_s'][0],
                                    dtype=np.float64) + single_frontend['rf']
            if iterations>1:
                S21_axis_tmp = np.mean(np.split(openH5file(filename, front_end = front_ends[fr])[0], iterations), axis = 0)
            else:
                S21_axis_tmp  = openH5file(filename, front_end = front_ends[fr])[0]

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
        vna_grp = f.create_group("VNA")
    except ValueError:
        print_warning("Overwriting VNA group")
        del f["VNA"]
        vna_grp = f.create_group("VNA")

    vna_grp.attrs.create("scan_lengths", length)
    vna_grp.attrs.create("calibration", calibration)
    
    vna_grp.create_dataset("frequency", data = freq_axis, dtype=np.float64)
    vna_grp.create_dataset("S21", data = S21_axis, dtype=np.complex128)

    f.close()

    print_debug("Analysis of file \'%s\' concluded."%filename)

def is_VNA_analyzed(filename):
    '''
    Check if the VNA file has been preanalyzed. Basically checks the presence of the VNA group inside the file.
    :param filename: The file to check.
    :return: boolean results of the check.
    '''
    filename = format_filename(filename)
    f = bound_open(filename)
    try:
        grp = f['VNA']
        if grp['frequency'] is not None: pass
        if grp['S21'] is not None: pass
        return True
    except KeyError:
        return False

def get_VNA_data(filename):
    '''
    Get the frequency and S21 data in a preanalyzed vna file.
    :param filename: the name of the HDF5 file containing the data.
    :return: frequency and S21 axis.
    '''
    if is_VNA_analyzed(filename):
        filename = format_filename(filename)
        f = bound_open(filename)
    else:
        err_msg = "Cannot get VNA data from file \'%s\' as it is not analyzed." % filename
        print_error(err_msg)
        raise ValueError(err_msg)

    return np.asarray(f['VNA']['frequency']), np.asarray(f['VNA']['S21'])


def plot_VNA(filenames, backend = "matplotlib", **kwargs):
    '''
    Plot the VNA data from various files.
    :param filenames: list of strings containing the filenames to be plotted.
    :param backend: "matplotlib", "plotly" or "bokeh" are supported.
    :param kwargs: figsize=(xx,yy) for bokeh and matplotlib backends; Comments = ["..","..",".."] fore commenting each
           file in the legend; html only to return a html text instead of nothing in case of bokeh or plotly backend.
    '''

    filenames = to_list_of_str(filenames)
    freq_axes = []
    S21_axes = []
    for filename in filenames:
        freq_tmp, S21_tmp = get_VNA_data(filename)

        freq_axes.append(freq_tmp)
        S21_axes.append(S21_tmp)

    del S21_tmp
    del freq_tmp


def Get_noise(tones, measure_t, rate, decimation = None, powers = None, RF = None, filename = None, Front_end = None, Device = None):
    '''
    Perform a noise acquisition using fixed tone technique.
    
    Arguments:
        - tones: list of tones frequencies in Hz (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - decimation: the decimation factor to use for the acquisition. Default is maximum.
        - powers: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        
    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''
    
    
