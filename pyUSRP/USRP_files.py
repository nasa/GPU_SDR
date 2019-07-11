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

# needed to print the data acquisition process
import progressbar

# import submodules
from USRP_low_level import *

def format_filename(filename):
    return os.path.splitext(filename)[0]+".h5"

def bound_open(filename):
    '''
    Return pointer to file. It's user responsability to call the close() method.
    '''
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

    :param ant (optional) string to specify receiver. Default is the first found.
    :return Parameter dictionary

    '''
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if ant is None:
        ant = parameters.get_active_rx_param()[0]
    else:
        ant = str(ant)

    return parameters.parameters[ant]

def get_tx_info(filename, ant=None):
    '''
    Retrive TX information from file.

    :param ant (optional) string to specify transmitter. Default is the first found.
    :return Parameter dictionary

    '''
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if ant is None:
        ant = parameters.get_active_tx_param()[0]
    else:
        ant = str(ant)

    return parameters.parameters[ant]


def openH5file(filename, ch_list=None, start_sample=None, last_sample=None, usrp_number=None, front_end=None,
               verbose=False, error_coord=False, big_file = False):
    '''
    Retrive Raw data from an hdf5 file generated with pyUSRP.

    :param filename: Name of the file to open
    :param ch_list: a list containing the channel number tp open.
    :param start_sample: first sample returned.
    :param last_sample: last sample returned.
    :param usrp_number: if the file contains more than one USRP data, select the usrp server number.
    :param front_end: select the front end for data sourcing. Default is automatically detected or A.
    :param verbose: print more information about the opening process.
    :param error_coord: If True returns (samples, err_coord) where err_coord is a list of tuples containing start and end sample of each faulty packet.
    :param big_file: default is False. if True last_sample and start_sample are ignored and the hdf5 object containing the raw data is returned. This is usefull when dealing with very large files. IMPORTANT: is user responsability to close the file if big_file is True, see return sepcs.

    :return: array-like object containing the data in the form data[channel][samples].
    :return: In case big_file is True returns the file object (so the user is able to close it) and the raw dataset. (file_pointer, dataset)
    :return: in case of error_coord True returns also the erorrs coordinate ((file_pointer,) dataset, errors)
    '''

    try:
        filename = format_filename(filename)
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

    if (front_end is None) and (len(recv) != 1):
        this_warning = "Multiple acquisition frontend subgroups found but no preference given to open file function. Assuming " + \
                       recv[0]
        sub_group_name = recv[0]

    if (front_end is None) and (len(recv) == 1):
        sub_group_name = recv[0]


    if front_end is not None:
        sub_group_name = str(front_end)

    try:
        sub_group = group[sub_group_name]
    except KeyError:
        print_error(
            "Cannot find sub group name %s. For X300 USRP possible frontends are \"A_TXRX\",\"B_TXRX\",\"A_RX2\",\"B_RX2\""%sub_group_name)
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
                    print_warning("Cannot find one or more dataset(s) in the h5 file")
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
        try:
            trigger = sub_group['trigger']
            print_warning("Gettign data from a triggered measure, time sequentiality of data is not guaranteed. To access the triggering info use get_trigger_info()")
        except KeyError:
            pass
        if samples is None:
            print_warning("Non samples attrinut found: data extracted from file could include zero padding")
            samples = last_sample
        if len(sub_group["errors"]) > 0:
            print_warning("The measure opened contains %d erorrs!" % len(sub_group["errors"]))

        if not big_file:
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
        else:
            if error_coord:
                errors = sub_group["errors"][:]
                return f,sub_group["data"],errors
            return f,sub_group["data"]



def get_noise(filename, usrp_number=0, front_end=None, channel_list=None):
    '''
    Get the noise spectra from a a pre-analyzed H5 file.

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

def get_trigger_info(filename, ant = None):
    '''
    Get the trigger information from a triggered measure.

    :param filename: the name of the measure file.
    :param ant: the name of the antenna. Default is automatically discovered.

    :returns trigger dataset as a numpy array.

    '''
    return

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

class global_parameter(object):
    '''
    Global paramenter object representing a measure.
    '''

    def __init__(self):
        self.initialized = False

    def initialize(self):
        '''
        Initialize the parameter object to a zero configuration.
        '''
        if self.initialized == True:
            print_warning("Reinitializing global parameters to blank!")
        self.initialized = True
        empty_spec = {}
        empty_spec['mode'] = "OFF"
        empty_spec['rate'] = 0
        empty_spec['rf'] = 0
        empty_spec['gain'] = 0
        empty_spec['bw'] = 0
        empty_spec['samples'] = 0
        empty_spec['delay'] = 1
        empty_spec['burst_on'] = 0
        empty_spec['burst_off'] = 0
        empty_spec['buffer_len'] = 0
        empty_spec['freq'] = [0]
        empty_spec['wave_type'] = [0]
        empty_spec['ampl'] = [0]
        empty_spec['decim'] = 0
        empty_spec['chirp_f'] = [0]
        empty_spec['swipe_s'] = [0]
        empty_spec['chirp_t'] = [0]
        empty_spec['fft_tones'] = 0
        empty_spec['pf_average'] = 4

        empty_spec['data_mem_mult'] = 1

        empty_spec['tuning_mode'] = 1  # fractional
        prop = {}
        prop['A_TXRX'] = empty_spec.copy()
        prop['B_TXRX'] = empty_spec.copy()
        prop['A_RX2'] = empty_spec.copy()
        prop['B_RX2'] = empty_spec.copy()
        prop['device'] = 0
        self.parameters = prop.copy()

    def get(self, ant, param_name):
        if not self.initialized:
            print_error("Retriving parameters %s from an uninitialized global_parameter object" % param_name)
            return None
        try:
            test = self.parameters[ant]
        except KeyError:
            print_error("The antenna \'" + ant + "\' is not an accepted frontend name or is not present.")
            return None
        try:
            return test[param_name]
        except KeyError:
            print_error("The parameter \'" + param_name + "\' is not an accepted parameter or is not present.")
            return None

    def set(self, ant, param_name, val):
        '''
        Initialize the global parameters object and set a parameter value.

        Arguments:
            - ant: a string containing one of the following 'A_TXRX', 'B_TXRX', 'A_RX2', 'B_RX2'. Where the first letter ferers to the front end and the rest to the connector.
            - param_name: a string containing the paramenter name one wants to change. For a complete list of accepeted parameters see section ? in the documentation.
            - val: value to assign.

        Returns:
            Boolean value representing the success of the operation.

        Note:
            if the parameter object is already initialized it does not overwrite the other paramenters.
            This function DOES NOT perform any check on the input parameter; for that check the self_check() method.
        '''
        if not self.initialized:
            self.initialize()
        try:
            test = self.parameters[ant]
        except KeyError:
            print_error("The antenna \'" + ant + "\' is not an accepted frontend name.")
            return False
        try:
            test = self.parameters[ant][param_name]
        except KeyError:
            print_error("The parameter \'" + param_name + "\' is not an accepted parameter.")
            return False
        self.parameters[ant][param_name] = val
        return True

    def is_legit(self):
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None
        return \
            self.parameters['A_TXRX']['mode'] != "OFF" or \
            self.parameters['B_TXRX']['mode'] != "OFF" or \
            self.parameters['A_RX2']['mode'] != "OFF" or \
            self.parameters['B_RX2']['mode'] != "OFF"

    def self_check(self):
        '''
        Check if the parameters are coherent.

        Returns:
            booleare representing the result of the check

        Note:
            To know what's wrong, check the warnings.
        '''
        if self.initialized:
            if not self.is_legit():
                return False
            for ant_key in self.parameters:
                if ant_key == 'device':
                    continue
                if self.parameters[ant_key]['mode'] != "OFF":

                    self.parameters[ant_key]['rate'] = int(self.parameters[ant_key]['rate'])
                    self.parameters[ant_key]['rf'] = int(self.parameters[ant_key]['rf'])

                    if isinstance(self.parameters[ant_key]['chirp_f'],np.ndarray):
                        self.parameters[ant_key]['chirp_f'] = self.parameters[ant_key]['chirp_f'].tolist()

                    if isinstance(self.parameters[ant_key]['freq'],np.ndarray):
                        self.parameters[ant_key]['freq'] = self.parameters[ant_key]['freq'].tolist()

                    if isinstance(self.parameters[ant_key]['ampl'],np.ndarray):
                        self.parameters[ant_key]['ampl'] = self.parameters[ant_key]['ampl'].tolist()
                        #receive does not use ampl
                        if self.parameters[ant_key]['mode'] == 'RX':
                            for ii in range(len(self.parameters[ant_key]['ampl'])):
                                self.parameters[ant_key]['ampl'][ii] = 1
                    try:
                        len(self.parameters[ant_key]['chirp_f'])
                    except TypeError:
                        print_warning("\'chirp_f\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['chirp_f'] = [self.parameters[ant_key]['chirp_f']]
                    try:
                        int(self.parameters[ant_key]['chirp_f'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'chirp_f\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['swipe_s'])
                    except TypeError:
                        print_warning("\'swipe_s\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['swipe_s'] = [self.parameters[ant_key]['swipe_s']]
                    try:
                        int(self.parameters[ant_key]['swipe_s'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'swipe_s\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['chirp_t'])
                    except TypeError:
                        print_warning("\'chirp_t\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['chirp_t'] = [self.parameters[ant_key]['chirp_t']]
                    try:
                        int(self.parameters[ant_key]['chirp_t'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'chirp_t\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['freq'])
                        self.parameters[ant_key]['freq'] = [int(xx) for xx in self.parameters[ant_key]['freq']]
                    except TypeError:
                        print_warning("\'freq\" attribute in parameters has to be a list of int, changing value to list...")


                    try:
                        int(self.parameters[ant_key]['freq'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'freq\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['wave_type'])
                    except TypeError:
                        print_warning(
                            "\'wave_type\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['wave_type'] = [self.parameters[ant_key]['wave_type']]

                    for i in range(len(self.parameters[ant_key]['freq'])):
                        self.parameters[ant_key]['freq'][i] = int(self.parameters[ant_key]['freq'][i])

                    try:
                        len(self.parameters[ant_key]['ampl'])
                    except TypeError:
                        print_warning("\'ampl\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['ampl'] = [self.parameters[ant_key]['ampl']]
                    try:
                        int(self.parameters[ant_key]['ampl'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'ampl\' should be a list of numerical values, not a string")
                        return False
                    try:
                        if self.parameters[ant_key]['tuning_mode'] is None:
                            self.parameters[ant_key]['tuning_mode'] = 0
                        else:
                            try:
                                int(self.parameters[ant_key]['tuning_mode'])
                            except ValueError:
                                try:
                                    if "int" in str(self.parameters[ant_key]['tuning_mode']):
                                        self.parameters[ant_key]['tuning_mode'] = 0
                                    elif "frac" in str(self.parameters[ant_key]['tuning_mode']):
                                        self.parameters[ant_key]['tuning_mode'] = 1
                                except:
                                    print_warning("Cannot recognize tuning mode\'%s\' setting to integer mode." % str(
                                        self.parameters[ant_key]['tuning_mode']))
                                    self.parameters[ant_key]['tuning_mode'] = 0
                    except KeyError:
                        self.parameters[ant_key]['tuning_mode'] = 0

                        # matching the integers conversion
                    for j in range(len(self.parameters[ant_key]['freq'])):
                        self.parameters[ant_key]['freq'][j] = int(self.parameters[ant_key]['freq'][j])
                    for j in range(len(self.parameters[ant_key]['swipe_s'])):
                        self.parameters[ant_key]['swipe_s'][j] = int(self.parameters[ant_key]['swipe_s'][j])
                    for j in range(len(self.parameters[ant_key]['chirp_f'])):
                        self.parameters[ant_key]['chirp_f'][j] = int(self.parameters[ant_key]['chirp_f'][j])

                    self.parameters[ant_key]['samples'] = int(self.parameters[ant_key]['samples'])

                    self.parameters[ant_key]['data_mem_mult'] = int(self.parameters[ant_key]['data_mem_mult'])

                    if ((self.parameters[ant_key]['wave_type'][0] == "DIRECT")):
                        self.parameters[ant_key]['data_mem_mult'] = max(np.ceil(len(self.parameters[ant_key]['wave_type'])/max(float(self.parameters[ant_key]['decim']),1)),1)

                # case in which it is OFF:
                else:
                    self.parameters[ant_key]['data_mem_mult'] = 0
                    self.parameters[ant_key]['mode'] = "OFF"
                    self.parameters[ant_key]['rate'] = 0
                    self.parameters[ant_key]['rf'] = 0
                    self.parameters[ant_key]['gain'] = 0
                    self.parameters[ant_key]['bw'] = 0
                    self.parameters[ant_key]['samples'] = 0
                    self.parameters[ant_key]['delay'] = 1
                    self.parameters[ant_key]['burst_on'] = 0
                    self.parameters[ant_key]['burst_off'] = 0
                    self.parameters[ant_key]['buffer_len'] = 0
                    self.parameters[ant_key]['freq'] = [0]
                    self.parameters[ant_key]['wave_type'] = [0]
                    self.parameters[ant_key]['ampl'] = [0]
                    self.parameters[ant_key]['decim'] = 0
                    self.parameters[ant_key]['chirp_f'] = [0]
                    self.parameters[ant_key]['swipe_s'] = [0]
                    self.parameters[ant_key]['chirp_t'] = [0]
                    self.parameters[ant_key]['fft_tones'] = 0
                    self.parameters[ant_key]['pf_average'] = 4
                    self.parameters[ant_key]['tuning_mode'] = 1  # fractional

        else:
            return False

        #print_debug("check function is not complete yet. In case something goes unexpected, double check parameters.")
        return True

    def from_dict(self, ant, dictionary):
        '''
        Initialize the global parameter object from a dictionary.

        Arguments:
            - ant: a string containing one of the following 'A_TXRX', 'B_TXRX', 'A_RX2', 'B_RX2'. Where the first letter ferers to the front end and the rest to the connector.
            - dictionary: a dictionary containing the parameters.

        Note:
            if the object is already initialize it overwrites only the parameters in the dictionary otherwise it initializes the object to flat zero before applying the dictionary.

        Returns:
            - boolean representing the success of the operation
        '''
        print_warning("function not implemented yet")
        return True

    def pprint(self):
        '''
        Output on terminal a diagnosti string representing the parameters.
        '''
        x = self.to_json()
        parsed = json.loads(x)
        print json.dumps(parsed, indent=4, sort_keys=True)

    def to_json(self):
        '''
        Convert the global parameter object to JSON string.

        Returns:
            - the string to be filled with the JSON
        '''
        return json.dumps(self.parameters)

    def get_active_rx_param(self):
        '''
        Discover which is(are) the active receiver designed by the global parameter object.

        Returns:
            list of active rx antenna names : this names correspond to the parameter group in the h5 file and to the dictionary name in the (this) parameter object.
        '''
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None

        if not self.is_legit():
            print_warning("There is no active RX channel in property tree")
            return None

        active_rx = []

        if self.parameters['A_TXRX']['mode'] == "RX":
            active_rx.append('A_TXRX')
        if self.parameters['B_TXRX']['mode'] == "RX":
            active_rx.append('B_TXRX')
        if self.parameters['A_RX2']['mode'] == "RX":
            active_rx.append('A_RX2')
        if self.parameters['B_RX2']['mode'] == "RX":
            active_rx.append('B_RX2')

        return active_rx

    def get_active_tx_param(self):
        '''
        Discover which is(are) the active emitters designed by the global parameter object.

        Returns:
            list of active tx antenna names : this names correspond to the parameter group in the h5 file and to the dictionary name in the (this) parameter object.
        '''
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None

        if not self.is_legit():
            print_warning("There is no active TX channel in property tree")
            return None

        active_tx = []

        if self.parameters['A_TXRX']['mode'] == "TX":
            active_tx.append('A_TXRX')
        if self.parameters['B_TXRX']['mode'] == "TX":
            active_tx.append('B_TXRX')
        if self.parameters['A_RX2']['mode'] == "TX":
            active_tx.append('A_RX2')
        if self.parameters['B_RX2']['mode'] == "TX":
            active_tx.append('B_RX2')

        return active_tx

    def retrive_prop_from_file(self, filename, usrp_number=None):
        def read_prop(group, sub_group_name):
            def missing_attr_warning(att_name, att):
                if att == None:
                    print_warning("Parameter \"" + str(att_name) + "\" is not defined")

            sub_prop = {}
            try:
                sub_group = group[sub_group_name]
            except KeyError:
                sub_prop['mode'] = "OFF"
                return sub_prop

            sub_prop['mode'] = sub_group.attrs.get('mode')
            missing_attr_warning('mode', sub_prop['mode'])

            sub_prop['rate'] = sub_group.attrs.get('rate')
            missing_attr_warning('rate', sub_prop['rate'])

            sub_prop['rf'] = sub_group.attrs.get('rf')
            missing_attr_warning('rf', sub_prop['rf'])

            sub_prop['gain'] = sub_group.attrs.get('gain')
            missing_attr_warning('gain', sub_prop['gain'])

            sub_prop['bw'] = sub_group.attrs.get('bw')
            missing_attr_warning('bw', sub_prop['bw'])

            sub_prop['samples'] = sub_group.attrs.get('samples')
            missing_attr_warning('samples', sub_prop['samples'])

            sub_prop['delay'] = sub_group.attrs.get('delay')
            missing_attr_warning('delay', sub_prop['delay'])

            sub_prop['burst_on'] = sub_group.attrs.get('burst_on')
            missing_attr_warning('burst_on', sub_prop['burst_on'])

            sub_prop['burst_off'] = sub_group.attrs.get('burst_off')
            missing_attr_warning('burst_off', sub_prop['burst_off'])

            sub_prop['buffer_len'] = sub_group.attrs.get('buffer_len')
            missing_attr_warning('buffer_len', sub_prop['buffer_len'])

            sub_prop['freq'] = sub_group.attrs.get('freq').tolist()
            missing_attr_warning('freq', sub_prop['freq'])

            sub_prop['wave_type'] = sub_group.attrs.get('wave_type').tolist()
            missing_attr_warning('wave_type', sub_prop['wave_type'])

            sub_prop['ampl'] = sub_group.attrs.get('ampl').tolist()
            missing_attr_warning('ampl', sub_prop['ampl'])

            sub_prop['decim'] = sub_group.attrs.get('decim')
            missing_attr_warning('decim', sub_prop['decim'])

            sub_prop['chirp_f'] = sub_group.attrs.get('chirp_f').tolist()
            missing_attr_warning('chirp_f', sub_prop['chirp_f'])

            sub_prop['chirp_t'] = sub_group.attrs.get('chirp_t').tolist()
            missing_attr_warning('chirp_t', sub_prop['chirp_t'])

            sub_prop['swipe_s'] = sub_group.attrs.get('swipe_s').tolist()
            missing_attr_warning('swipe_s', sub_prop['swipe_s'])

            sub_prop['fft_tones'] = sub_group.attrs.get('fft_tones')
            missing_attr_warning('fft_tones', sub_prop['fft_tones'])

            sub_prop['pf_average'] = sub_group.attrs.get('pf_average')
            missing_attr_warning('pf_average', sub_prop['pf_average'])

            sub_prop['tuning_mode'] = sub_group.attrs.get('tuning_mode')
            missing_attr_warning('tuning_mode', sub_prop['tuning_mode'])

            return sub_prop

        f = bound_open(filename)
        if f is None:
            return None

        if (not usrp_number) and chk_multi_usrp(f) != 1:
            this_warning = "Multiple usrp found in the file but no preference given to get prop function. Assuming usrp " + str(
                (f.keys()[0]).split("ata")[1])
            print_warning(this_warning)
            group_name = "raw_data0"  # +str((f.keys()[0]).split("ata")[1])

        if (not usrp_number) and chk_multi_usrp(f) == 1:
            group_name = "raw_data0"  # f.keys()[0]

        if (usrp_number != None):
            group_name = "raw_data" + str(int(usrp_number))

        try:
            group = f[group_name]
        except KeyError:
            print_error("Cannot recognize group format")
            return None

        prop = {}
        prop['A_TXRX'] = read_prop(group, 'A_TXRX')
        prop['B_TXRX'] = read_prop(group, 'B_TXRX')
        prop['A_RX2'] = read_prop(group, 'A_RX2')
        prop['B_RX2'] = read_prop(group, 'B_RX2')
        self.initialized = True
        self.parameters = prop


def Device_chk(device):
    '''
    Check if the device is recognised by the server or assign to 0 by default.

    Arguments:
        - device number or None.

    Returns:
        - boolean representing the result of the check or true is assigned by default.
    '''
    if device == None:
        device = 0
        return True

    print_warning("Async HW information has not been implemented yet")
    return True


def Front_end_chk(Front_end):
    '''
    Check if the front end code is recognised by the server or assign to A by default.

    Arguments:
        - front end code (A or B).

    Returns:
        - boolean representing the result of the check or true is assigned by default.
    '''

    if (Front_end != "A") and (Front_end != "B"):
        print_error("Front end \"" + str(Front_end) + "\" not recognised.")
        return False

    return True


def Param_to_H5(H5fp, parameters_class, trigger = None, **kwargs):
    '''
    Generate the internal structure of a H5 file correstonding to the parameters given.

    :param H5fp: already opened H5 file with write permissions
    :param parameters_class: an initialized global_parameter object containing the informations used to drive the GPU server.
    :param kwargs: each additional parameter will be interpreted as a tag to add in the raw data group of the file.
    :param trigger: trigger class (see section on trigger function for deteails)

    Returns:
        - A list of names of H5 groups where to write incoming data.

    Note:
        This function is ment to be used inside the Packets_to_file() function for data collection.
    '''
    if parameters_class.self_check():
        rx_names = parameters_class.get_active_rx_param()
        tx_names = parameters_class.get_active_tx_param()
        usrp_group = H5fp.create_group("raw_data" + str(int(parameters_class.parameters['device'])))

        for tag_name in kwargs:
            usrp_group.attrs.create(name=tag_name, data=kwargs[tag_name])

        for ant_name in tx_names:
            tx_group = usrp_group.create_group(ant_name)
            for param_name in parameters_class.parameters[ant_name]:
                tx_group.attrs.create(name=param_name, data=parameters_class.parameters[ant_name][param_name])

        for ant_name in rx_names:
            rx_group = usrp_group.create_group(ant_name)
            # Avoid dynamical disk space allocation by forecasting the size of the measure
            try:
                n_chan = len(parameters_class.parameters[ant_name]['wave_type'])
            except KeyError:
                print_warning("Cannot extract number of channel from signal processing descriptor")
                n_chan = 0
            if trigger is not None:
                data_len = 0
            else:
                if parameters_class.parameters[ant_name]['wave_type'][0] == "TONES":
                    data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / (
                                parameters_class.parameters[ant_name]['fft_tones'] * max(
                            parameters_class.parameters[ant_name]['decim'], 1))))
                elif parameters_class.parameters[ant_name]['wave_type'][0] == "CHIRP":
                    if parameters_class.parameters[ant_name]['decim'] == 0:
                        data_len = parameters_class.parameters[ant_name]['samples']
                    else:
                        data_len = parameters_class.parameters[ant_name]['swipe_s'][0]/parameters_class.parameters[ant_name]['decim']

                elif parameters_class.parameters[ant_name]['wave_type'][0] == "NOISE":
                    data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / max(
                        parameters_class.parameters[ant_name]['decim'], 1)))

                elif parameters_class.parameters[ant_name]['wave_type'][0] == "DIRECT":
                    data_len = parameters_class.parameters[ant_name]['samples']/max(parameters_class.parameters[ant_name]['decim'],1)
                else:
                    print_warning("No file size could be determined from DSP descriptor: \'%s\'" % str(
                        parameters_class.parameters[ant_name]['wave_type'][0]))
                    data_len = 0

            data_shape_max = (n_chan, data_len)
            rx_group.create_dataset("data", data_shape_max, dtype=np.complex64, maxshape=(None, None),
                                    chunks=True)  # , compression = H5PY_compression
            rx_group.create_dataset("errors", (0, 0), dtype=np.dtype(np.int64),
                                    maxshape=(None, None))  # , compression = H5PY_compression

            if trigger is not None:
                trigger_ds = rx_group.create_dataset("trigger", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,),chunks=True)
                trigger_name = str(trigger.__class__.__name__)
                trigger_ds.attrs.create("trigger_fcn", data = trigger_name)

                trigger.dataset_init(rx_group)

            for param_name in parameters_class.parameters[ant_name]:
                rx_group.attrs.create(name=param_name, data=parameters_class.parameters[ant_name][param_name])

        return rx_names

    else:
        print_error("Cannot initialize H5 file without checked parameters.self_check() failed.")
        return []


def is_VNA_analyzed(filename, usrp_number = 0):
    '''
    Check if the VNA file has been preanalyzed. Basically checks the presence of the VNA group inside the file.
    :param filename: The file to check.
    :param usrp_number: usrp server number.
    :return: boolean results of the check.
    '''
    filename = format_filename(filename)
    f = bound_open(filename)
    try:
        grp = f["VNA_%d"%(usrp_number)]
        if grp['frequency'] is not None: pass
        if grp['S21'] is not None: pass
        ret = True
    except KeyError:
        ret = False
    f.close()
    return ret


def get_VNA_data(filename, calibrated = True, usrp_number = 0):
    '''
    Get the frequency and S21 data in a preanalyzed vna file.
    :param filename: the name of the HDF5 file containing the data.
    :param calibrated: if True returns the S21 data in linear ratio units (Vrms(in)/Vrms(out)). if False returns S21 in ADC units.
    :param usrp_number: usrp server number.
    :return: frequency and S21 axis.

    TO DO:
        - Calibrarion for frontend A could be different from frontend B. This could lead to a wrong calibration.
    '''
    usrp_number = int(usrp_number)
    if is_VNA_analyzed(filename):
        filename = format_filename(filename)
        f = bound_open(filename)
    else:
        err_msg = "Cannot get VNA data from file \'%s\' as it is not analyzed." % filename
        print_error(err_msg)
        raise ValueError(err_msg)
    if not calibrated:
        ret =  np.asarray(f["VNA_%d"%(usrp_number)]['frequency']), np.asarray(f["VNA_%d"%(usrp_number)]['S21'])
    else:
        ret =  np.asarray(f["VNA_%d"%(usrp_number)]['frequency']), np.asarray(f["VNA_%d"%(usrp_number)]['S21'])* f['VNA_%d'%(usrp_number)].attrs.get('calibration')[0]

    f.close()
    return ret

def get_dynamic_VNA_data(filename, calibrated = True, usrp_number = 0):
    '''
    Get the dynamic frequency and S21 data in a preanalyzed vna file.
    :param filename: the name of the HDF5 file containing the data.
    :param calibrated: if True returns the S21 data in linear ratio units (Vrms(in)/Vrms(out)). if False returns S21 in ADC units.
    :param usrp_number: usrp server number.
    :return: frequency and S21 axis.

    TO DO:
        - Calibrarion for frontend A could be different from frontend B. This could lead to a wrong calibration.
    '''
    usrp_number = int(usrp_number)
    if is_VNA_dynamic_analyzed(filename):
        filename = format_filename(filename)
        f = bound_open(filename)
    else:
        err_msg = "Cannot get VNA data from file \'%s\' as it is not analyzed." % filename
        print_error(err_msg)
        raise ValueError(err_msg)
    if not calibrated:
        ret =  np.asarray(f["VNA_dynamic_%d"%(usrp_number)]['frequency']), np.asarray(f["VNA_dynamic_%d"%(usrp_number)]['S21'])
    else:
        ret =  np.asarray(f["VNA_dynamic_%d"%(usrp_number)]['frequency']), np.asarray(f["VNA_dynamic_%d"%(usrp_number)]['S21'])* f['VNA_dynamic_%d'%(usrp_number)].attrs.get('calibration')[0]

    f.close()
    return ret


def get_init_peaks(filename, verbose = False):
    '''
    Get initialized peaks froma a VNA file.

    Arguments:
        - filename: the name of the file containing the peaks.
        - verbose: print some debug line.

    Return:
        - Numpy array containing the frequency of each ninitialized peak in MHz.
    '''

    file = bound_open(filename)

    try:
        inits = file["Resonators"].attrs.get("tones_init")
    except ValueError:
        inits = np.asarray([])
        if(verbose): print_debug("get_init_peaks() did not find any initialized peak")
    except KeyError:
        inits = np.asarray([])
        if(verbose): print_debug("get_init_peaks() did not find any initialized peak")
    file.close()

    return np.asarray(inits)


def is_VNA_analyzed(filename, usrp_number = 0):
    '''
    Check if the VNA file has been preanalyzed. Basically checks the presence of the VNA group inside the file.
    :param filename: The file to check.
    :param usrp_number: usrp server number.
    :return: boolean results of the check.
    '''
    filename = format_filename(filename)
    f = bound_open(filename)
    try:
        grp = f["VNA_%d"%(usrp_number)]
        if grp['frequency'] is not None: pass
        if grp['S21'] is not None: pass
        ret = True
    except KeyError:
        ret = False
    f.close()
    return ret

def is_VNA_dynamic_analyzed(filename, usrp_number = 0):
    '''
    Check if the VNA file has been preanalyzed as a dynamic VNA. Basically checks the presence of the VNA_dynamic group inside the file.
    :param filename: The file to check.
    :param usrp_number: usrp server number.
    :return: boolean results of the check.
    '''
    filename = format_filename(filename)
    f = bound_open(filename)
    try:
        grp = f["VNA_dynamic_%d"%(usrp_number)]
        if grp['frequency'] is not None: pass
        if grp['S21'] is not None: pass
        ret = True
    except KeyError:
        ret = False
    f.close()
    return ret
