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
from USRP_low_lever import *

def format_filename(filename):
    return os.path.splitext(filename)[0]+".h5"

def bound_open(filename):

    try:
        filename = format_filename(filename)
        f = h5py.File(filename,'r')
    except IOError as msg:
        print_error("Cannot open the specified file: "+str(msg))
        f = None
    return f

def chk_multi_usrp(h5file):
    n = 0
    for i in range(len(h5file.keys())):
        if h5file.keys()[i][:8] == 'raw_data':
            n+=1
    return n

def get_receivers(h5group):
    receivers = []
    subs = h5group.keys()
    for i in range( len(subs) ):
        mode = (h5group[subs[i]]).attrs.get("mode")
        if mode == "RX":
            receivers.append(str(h5group.keys()[i]))
    return receivers


def get_rx_info(filename, ant=None):
    '''
    Retrive RX information from file.

    Arguments:
        - optional ant string to specify receiver. Default is the first found.

    Return:
        Parameter dictionary
    '''
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if ant is None:
        ant = parameters.get_active_rx_param()[0]
    else:
        ant = str(ant)

    return parameters.parameters[ant]


def openH5file(filename, ch_list=None, start_sample=None, last_sample=None, usrp_number=None, front_end=None,
               verbose=False, error_coord=False):
    '''
    Arguments:
        error_coord: if True returns (samples, err_coord) where err_coord is a list of tuples containing start and end sample of each faulty packet.

    '''

    try:
        filename = filename.split(".")[0]
    except:
        print_error("cannot interpret filename while opening a H5 file")
        return None

    if (verbose):
        print_debug("Opening file \"" + filename + ".h5\"... ")

    f = bound_open(filename)
    if not f:
        return np.asarray([])

    if (verbose):
        print_debug("Checking openH5 file function args... ")

    if chk_multi_usrp(f) == 0:
        print_error("No USRP data found in the hdf5 file")
        return np.asarray([])

    if (not usrp_number) and chk_multi_usrp(f) != 1:
        this_warning = "Multiple usrp found in the file but no preference given to open file function. Assuming usrp " + str(
            (f.keys()[0]).split("ata")[1])
        print_warning(this_warning)
        group_name = "raw_data" + str((f.keys()[0]).split("ata")[1])

    if (not usrp_number) and chk_multi_usrp(f) == 1:
        group_name = "raw_data0"  # f.keys()[0]

    if (usrp_number != None):
        group_name = "raw_data" + str(int(usrp_number))

    try:
        group = f[group_name]
    except KeyError:
        print_error("Cannot recognize group format")
        return np.asarray([])

    recv = get_receivers(group)

    if len(recv) == 0:
        print_error(
            "No USRP data found in the hdf5 file for the selected usrp number. Maybe RX mode attribute is missing or the H5 file is only descriptive of TX commands")
        return np.asarray([])

    if (not front_end) and len(recv) != 1:
        this_warning = "Multiple acquisition frontend subgroups found but no preference given to open file function. Assuming " + \
                       recv[0]
        sub_group_name = recv[0]

    if (not front_end) and len(recv) == 1:
        sub_group_name = recv[0]

    if front_end:
        sub_group_name = str(front_end)

    try:
        sub_group = group[sub_group_name]
    except KeyError:
        print_error(
            "Cannot recognize sub group format. For X300 USRP possible frontends are \"A_TXRX\",\"B_TXRX\",\"A_RX2\",\"B_RX2\"")
        return np.asarray([])

    n_chan = sub_group.attrs.get("n_chan")

    if n_chan == None:
        # print_warning("There is no attribute n_chan in the data group, cannot execute checks. Number of channels will be deducted from the freq attribute")
        n_chan = len(sub_group.attrs.get("wave_type"))
        # print_debug("Getting number of cannels from wave_type attribute shape: %d channel(s) found"%n_chan)

    if ch_list == None:
        ch_list = range(n_chan)

    if n_chan < max(ch_list):
        this_error = "Channel selected: " + str(
            max(ch_list)) + " in channels list in open file function exceed the total number of channels found: " + str(
            n_chan)
        print_error(this_error)
        raise IndexError
        # return np.asarray([])

    if start_sample == None:
        start_sample = 0

    if start_sample < 0:
        print_warning("Start sample selected in open file function < 0: setting it to 0")
        start_sample = 0
    else:
        start_sample = int(start_sample)

    if last_sample == None:
        last_sample = sys.maxint - 1
    else:
        last_sample = int(last_sample)

    if last_sample < 0 or last_sample < start_sample:
        print_warning("Last sample selected in open file function < 0 or < Start sample: setting it to maxint")
        last_sample = sys.maxint

    if (verbose):
        print_debug("Collecting samples...")

    z = []
    err_index = []
    sample_index = 0
    errors = 0

    # check if the opening mode is from the old server or the new one
    try:
        test = sub_group["dataset_1"]
        old_mode = True
    except KeyError:
        old_mode = False

    if old_mode:
        skip_warning = True
        print_debug("Using old dataset mode to open file \'%s\'" % filename)
        # data are contained in multiple dataset

        if verbose:
            widgets = [progressbar.Percentage(), progressbar.Bar()]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=len(sub_group.keys())).start()
            read = 0

        current_len = np.shape(sub_group["dataset_1"])[1]
        N_dataset = len(sub_group.keys())

        print_warning(
            "Raw amples inside " + filename + " have not been rearranged: the read from file can be slow for big files due to dataset reading overhead")

        for i in range(N_dataset):
            try:
                dataset_name = "dataset_" + str(int(1 + i))

                sample_index += current_len

                truncate_final = min(last_sample, last_sample - sample_index)
                if (last_sample >= sample_index):
                    truncate_final = current_len
                elif (last_sample < sample_index):
                    truncate_final = current_len - (sample_index - last_sample)

                if (sample_index > start_sample) and (truncate_final > 0):
                    present_error = sub_group[dataset_name].attrs.get('errors')
                    errors += int(present_error)
                    if present_error != 0:
                        err_index.append((sample_index - current_len, sample_index + current_len))
                    truncate_initial = max(0, current_len - (sample_index - start_sample))

                    z.append(sub_group[dataset_name][ch_list, truncate_initial:truncate_final])
            except KeyError:
                if skip_warning:
                    print_warning("Cannot find one or more datasets in the h5 file")
                    skip_warning = False

            if verbose:
                try:
                    bar.update(read)
                except:
                    print_debug("decrease samples in progeressbar")
                read += 1
        if errors > 0: print_warning("The measure opened contains %d erorrs!" % errors)
        if (verbose): print "Done!"
        f.close()

        if error_coord:
            return np.concatenate(tuple(z), 1), err_index
        return np.concatenate(tuple(z), 1)

    else:
        samples = sub_group["data"].attrs.get("samples")
        if samples is None:
            print_warning("Non samples attrinut found: data extracted from file could include zero padding")
            samples = last_sample
        if len(sub_group["errors"]) > 0:
            print_warning("The measure opened contains %d erorrs!" % len(sub_group["errors"]))
        if error_coord:
            data = sub_group["data"][ch_list, start_sample:last_sample]
            errors = sub_group["errors"][:]
            if errors is None:
                errors = []
            f.close()
            return data, errors

        data = sub_group["data"][ch_list, start_sample:last_sample]
        print_debug(
            "Shape returned from openH5file(%s) call: %s is (channels,samples)" % (filename, str(np.shape(data))))
        f.close()
        return data


def get_noise(filename, usrp_number=0, front_end=None, channel_list=None):
    '''
    Get the noise samples from a a pre-analyzed H5 file.

    Argumers:
        - filename: [string] the name of the file.
        - usrp_number: the server number of the usrp device. default is 0.
        - front_end: [string] name of the front end. default is extracted from data.
        - channel_list: [listo of int] specifies the channels from which to get samples
    Returns:
        - Noise info, Frequency axis, real axis, imaginary axis

    Note:
        Noise info is a dictionary containing the following parameters [whelch, dbc, rotate, rate, tone].
        The first four give information about the fft done to extract the noise; the last one is a list coherent with
        channel list containing the acquisition frequency of each tone in Hz.
    '''
    if usrp_number is None:
        usrp_number = 0

    filename = format_filename(filename)
    fv = h5py.File(filename, 'r')
    noise_group = fv["Noise" + str(int(usrp_number))]
    if front_end is not None:
        ant = front_end
    else:
        if len(noise_group.keys()) > 0:
            ant = noise_group.keys()[0]
        else:
            print_error("get_noise() cannot find valid front end names in noise group!")
            raise IndexError

    noise_subgroup = noise_group[ant]

    info = {}
    info['welch'] = noise_subgroup.attrs.get("welch")
    info['dbc'] = noise_subgroup.attrs.get("dbc")
    info['rotate'] = noise_subgroup.attrs.get("rotate")
    info['rate'] = noise_subgroup.attrs.get("rate")
    info['n_chan'] = noise_subgroup.attrs.get("n_chan")

    if channel_list is None:
        channel_list = range(info['n_chan'])

    info['tones'] = []

    frequency_axis = np.asarray(noise_subgroup['freq'])
    real = []
    imag = []
    for i in channel_list:
        real.append(np.asarray(noise_subgroup['real_' + str(int(i))]))
        imag.append(np.asarray(noise_subgroup['imag_' + str(int(i))]))
        info['tones'].append(noise_subgroup['imag_' + str(int(i))].attrs.get("tone"))

    fv.close()

    return info, frequency_axis, real, imag


def get_readout_power(filename, channel, front_end=None, usrp_number=0):
    '''
    Get the readout power for a given single tone channel.
    '''
    global USRP_power
    if usrp_number is None:
        usrp_number = 0
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if front_end is None:
        ant = parameters.get_active_tx_param()
    else:
        ant = [front_end]
    try:
        ampl = parameters.get(ant[0], 'ampl')[channel]
    except IndexError:
        print_error("Channel %d is not present in file %s front end %s" % (channel, filename, front_end))
        raise IndexError

    gain = parameters.get(ant[0], 'gain')

    return gain + USRP_power + 20 * np.log10(ampl)