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
                    except TypeError:
                        print_warning("\'freq\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['freq'] = [self.parameters[ant_key]['freq']]
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
                # case in which it is OFF:
                else:
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

        print_debug("check function is not complete yet. In case something goes unexpected, double check parameters.")
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
        - front end code (A or B) or None.

    Returns:
        - boolean representing the result of the check or true is assigned by default.
    '''
    if Front_end == None:
        Front_end = "A"
        return True

    if (Front_end != "A") or (Front_end != "B"):
        print_error("Front end \"" + str(Front_end) + "\" not recognised in VNA scan setting.")
        return False

    return True


def Param_to_H5(H5fp, parameters_class, tag):
    '''
    Generate the internal structure of a H5 file correstonding to the parameters given.

    Arguments:
        - H5fp: already opened H5 file with write permissions
        - parameters: an initialized global_parameter object containing the informations used to drive the GPU server.

    Returns:
        - A list of names of H5 groups where to write incoming data.

    Note:
        This function is ment to be used inside the Packets_to_file() function for data collection.
    '''
    if parameters_class.self_check():
        rx_names = parameters_class.get_active_rx_param()
        tx_names = parameters_class.get_active_tx_param()
        usrp_group = H5fp.create_group("raw_data" + str(int(parameters_class.parameters['device'])))
        if tag != None:
            usrp_group.attrs.create(name="tag", data=str(tag))
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

            if parameters_class.parameters[ant_name]['wave_type'][0] == "TONES":
                data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / (
                            parameters_class.parameters[ant_name]['fft_tones'] * max(
                        parameters_class.parameters[ant_name]['decim'], 1))))
            elif parameters_class.parameters[ant_name]['wave_type'][0] == "CHIRP":
                if parameters_class.parameters[ant_name]['decim'] < 1:
                    data_len = parameters_class.parameters[ant_name]['samples']
                else:
                    data_len = 0
            elif parameters_class.parameters[ant_name]['wave_type'][0] == "NOISE":
                data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / max(
                    parameters_class.parameters[ant_name]['decim'], 1)))
            else:
                print_warning("No file size could be determined from DSP descriptor: \'%s\'" % str(
                    parameters_class.parameters[ant_name]['wave_type'][0]))
                data_len = 0

            data_shape_max = (n_chan, data_len)
            data_shape = (0, 0)
            print "dataset initial length is %d" % data_len
            rx_group.create_dataset("data", data_shape_max, dtype=np.complex64, maxshape=(None, None),
                                    chunks=True)  # , compression = H5PY_compression
            rx_group.create_dataset("errors", (0, 0), dtype=np.dtype(np.int64),
                                    maxshape=(None, None))  # , compression = H5PY_compression
            for param_name in parameters_class.parameters[ant_name]:
                rx_group.attrs.create(name=param_name, data=parameters_class.parameters[ant_name][param_name])

        return rx_names

    else:
        print_error("Cannot initialize H5 file without checked parameters.self_check() failed.")
        return []
