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
from USRP_parameters import *
from USRP_data_analysis import *


def Single_VNA(start_f, last_f, measure_t, n_points, tx_gain, Rate = None, decimation = True, RF = None, Front_end = None, Device = None, filename = None, Multitone_compensation = None, Iterations = 1, verbose = False):

    '''
    Perform a VNA scan.
    
    Arguments:
        - start_f: frequency in Hz where to start scanning (absolute if RF is not given, relative to RF otherwise).
        - last_f: frequency in Hz where to stop scanning (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - n_points: number of points to use in the VNA scan.
        - tx_gain: transmission amplifier gain.
        - Rate: Optional parameter to control the scan rate. Default is calculate from start_f an last_f args.
        - decimation: if True the decimation of the signal will occour on-server. Default is True.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - Multitone_compensation: integer representing the number of tones: compensate the amplitude of the signal to match a future multitones accuisition.
        - Iterations: by default a single VNA scan pass is performed.
        - verbose: if True outputs on terminal some diagnostic info. deafult is False.
        
    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''        
    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE
    
    if measure_t <= 0:
        print_error("Cannot execute a VNA measure with "+str(measure_t)+"s duration.")
        return ""
    if n_points <= 0:
        print_error("Cannot execute a VNA measure with "+str(n_points)+" points.")
        return ""
    if RF == None:
        delta_f = np.abs(start_f - last_f)
        RF = delta_f/2.
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
        decimation = 2
    else:
        decimation = 0
        
    number_of_samples = Rate* measure_t
        
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
    vna_command.set("A_RX2","delay", 1)
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
    vna_command.set("A_RX2","decim", decimation)#THIS only activate the decimation.
    
    if vna_command.self_check():
        if(verbose): print "VNA command succesfully checked"
        Async_send(vna_command.to_json())
        
    else:
        print_warning("Something went wrong with the setting of VNA command.")
        return ""
    
        
    return ""
    
def Single_VNA_analysis(filename):
    '''
    Open a H5 file containing data collected with the function single_VNA() and analyze them as a VNA scan. Write the results in a corresponding VNA# group in the root of the H% file.
    
    Arguments:
        - filename: string containing the name of the H5 file
    
    Return:
        - boolean representing the success of the operation
    '''    
    return False
    
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


def Measure_line_delay(rate, LO_freq):
    '''
    Measure the line delay around a given frequency.
    '''
    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE

    # check connection

    # configure command

    # send command

    # write it to file

def Analyze_line_delay(filename):
    '''
    Analyze the file delay from a tagged file.

    :param filename: the name of the file containing the delay data.
    :return: the delay in seconds.
    '''

def load_delay_from_file(filename):
    '''
    Recover delay information from a delay file
    :param filename:
    :return:
    '''
    
    
