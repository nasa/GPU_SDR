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
import glob

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
import matplotlib.patches as mpatches

#needed to print the data acquisition process
import progressbar

#import submodules
from USRP_low_level import *
from USRP_connections import *
from USRP_plotting import *
from USRP_files import *
from USRP_data_analysis import *


#: This variable contains total line delay at given frequencies.
LINE_DELAY = {}

def measure_line_delay(rate, LO_freq, RF_frontend, USRP_num = 0, tx_gain = 0, rx_gain = 0, output_filename = None, compensate = False, duration = 0.01, **kwargs):
    '''
    Measure the line delay around a given frequency. Save the data to file.
    :param rate: USRP sampling rate in Sps: changing the sampling rate affect how the stock firmware process data.
    :param LO_freq: the LO frequency used during the delay measurement in Hz.
    :param RF_frontend: character 'A' or 'B' to define which front end to use for line delay calculation.
    :param USRP_num: Server USRP number to use for the measure. Default is 0.
    :param tx_gain: Transmission gain in dB to use for the measure. Default is 0.
    :param rx_gain: Receiver gain in dB to use for the measure. Default is 0.
    :param output_filename: optional parameter to set the output filename.
    :param kwargs: keyword arguments will be used to store additional attributes in the HDF5 file in the raw data group.
    :param compensate: if True the initial delay is sourced from the internal variable LINE_DELAY.
    :parm duration: duration of the measurement.
    :return A string containing the filename where the data have been saved.

    '''

    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY

    print_debug("Measuring line delay...")

    # check input parameters

    USRP_num = int(np.abs(USRP_num))

    if output_filename is None:
        output_filename = "USRP_Delay_"+get_timestamp()
    else:
        output_filename = str(output_filename)

    rate = int(np.abs(rate))

    LO_freq = int(np.abs(LO_freq))

    tx_gain = int(np.abs(tx_gain))
    rx_gain = int(np.abs(rx_gain))

    RF_frontend = str(RF_frontend)
    if RF_frontend != 'A' and RF_frontend != 'B':
        print_error("Cannot find forntend %s")
        return ""

    if RF_frontend == 'A':
        TX_frontend = "A_TXRX"
        RX_frontend = "A_RX2"
    else:
        TX_frontend = "B_TXRX"
        RX_frontend = "B_RX2"

    print_debug(
        "Saving on file: %s\nUsing front-end: \'%s\'\nUsing sample rate: %.2f MHz\nUsing LO frequency: %.2f"
        %(output_filename+".h5", RF_frontend, rate/1.e6, LO_freq/1.e6)
    )

    if tx_gain != 0:
        print_debug("Using transmission gain: %d"%tx_gain)
    if rx_gain != 0:
        print_debug("Using receiver gain: %d"%rx_gain)

    # configure command

    # This measure take the undecimated data rate that will most probably exceed the LAN/Disk rate.
    # Using more than few seconds could cause errors.
    measure_t = duration
    n_points = (rate * measure_t)
    number_of_samples = (rate * measure_t)

    start_f = int(np.floor(rate/2))-1
    last_f = -start_f

    # This is not lock-in decimation. Lock-in decimation only happens if decimation parm == 1.
    gpu_decim = 200

    # Delay compensation.
    if compensate:
        try:
            compensation = LINE_DELAY[str(int(rate/1e6))]*1e-9
            print_debug("Measuring a delay with a delay base of %d ns"%int(compensation*1e9))
        except KeyError:
            print_debug("Could not find already measured delay for rate %d Msps"%int(rate/1e6))
            compensation = 0
    else:
        compensation = 0

    delay_command = global_parameter()

    delay_command.set(TX_frontend, "mode", "TX")
    delay_command.set(TX_frontend, "buffer_len", 1000000)
    delay_command.set(TX_frontend, "gain", tx_gain)
    delay_command.set(TX_frontend, "delay", 1)
    delay_command.set(TX_frontend, "samples", number_of_samples)
    delay_command.set(TX_frontend, "rate", rate)
    delay_command.set(TX_frontend, "bw", 2 * rate)

    delay_command.set(TX_frontend, "wave_type", ["CHIRP"])
    delay_command.set(TX_frontend, "ampl", [1.])
    delay_command.set(TX_frontend, "freq", [start_f])
    delay_command.set(TX_frontend, "chirp_f", [last_f])
    delay_command.set(TX_frontend, "swipe_s", [n_points])
    delay_command.set(TX_frontend, "chirp_t", [measure_t])
    delay_command.set(TX_frontend, "rf", LO_freq)

    delay_command.set(RX_frontend, "mode", "RX")
    delay_command.set(RX_frontend, "buffer_len", 1e6)
    delay_command.set(RX_frontend, "gain", rx_gain)
    delay_command.set(RX_frontend, "delay", 1+compensation)
    delay_command.set(RX_frontend, "samples", number_of_samples)
    delay_command.set(RX_frontend, "rate", rate)
    delay_command.set(RX_frontend, "bw", 2 * rate)

    delay_command.set(RX_frontend, "wave_type", ["CHIRP"])
    delay_command.set(RX_frontend, "freq", [start_f])
    delay_command.set(RX_frontend, "chirp_f", [last_f])
    delay_command.set(RX_frontend, "swipe_s", [n_points])
    delay_command.set(RX_frontend, "chirp_t", [measure_t])
    delay_command.set(RX_frontend, "rf", LO_freq)
    delay_command.set(RX_frontend, "decim", gpu_decim)

    # send command

    if delay_command.self_check():
        Async_send(delay_command.to_json())

        #delay_command.pprint()

    else:
        print_error("Something went wrong with the line delay command.")
        return ""

    # write it to file

    Packets_to_file(
        parameters=delay_command,
        timeout=None,
        filename=output_filename,
        dpc_expected=number_of_samples/gpu_decim,
        meas_type = "delay", **kwargs
    )

    print_debug("Line delay acquisition terminated.")

    return output_filename

def write_delay_to_file(filename, delay):
    '''
    Wrapper around h5py functions to write the delay on file.

    :param filename: Name of the file containing the delay data.
    :param delay: Delay from analysis function is seconds.
    '''

    try:
        filename = format_filename(filename)
    except:
        print_error("Cannot interpret filename while opening a H5 file in write_delay_to_file function")
        raise ValueError("Cannot interpret filename while opening a H5 file in write_delay_to_file function")

    info = get_rx_info(filename, ant=None)

    try:
        filename = format_filename(filename)
        f = h5py.File(filename, 'r+')
    except IOError as msg:
        print_error("Cannot open "+str(filename)+" file in write_delay_to_file function"+ str(msg))
        raise ValueError("Cannot open "+str(filename)+" file in write_delay_to_file function: "+ str(msg))

    try:
        d_grp = f.create_group("delay_info")
    except ValueError:
        d_grp = f["delay_info"]

    rate = int(info['rate']/1e6)
    delay_ns = int(delay*1e9)
    print_debug("Storing a delay of %d ns for a rate of %d Msps"%(delay_ns, rate))

    d_grp.attrs.__setitem__(str(rate), delay_ns)

    f.close()

def analyze_line_delay(filename, diagnostic_plots = False):
    '''
    Analyze the file delay from a tagged file.

    :param filename: the name of the file containing the delay data.
    :param diagnostic_plots: saves two diagnostic plots in png.
    :return: the delay in seconds.
    '''

    print_debug("Analyzing line delay info form file: \'%s\' ..."%(filename))


    decimation = 2
    zz = signal.decimate(openH5file(filename)[0], decimation, ftype="fir")
    info = get_rx_info(filename, ant=None)
    decimation *= info['decim']
    freq, Pxx = signal.welch(zz.real, nperseg=len(zz), fs=int(info['rate'] / float(decimation)), detrend='linear',
                             scaling='density')

    if diagnostic_plots:
        diag_fig = pl.figure()
        pl.plot(zz.real, label="real")
        pl.plot(zz.imag, label="imag")
        pl.plot(np.abs(zz), label="abs")
        pl.title("Delay acquisition diagnostic.\n total decimation: %d"%decimation)
        pl.xlabel("Samples")
        pl.ylabel("ADCu")
        pl.legend()
        pl.grid()
        pl.savefig("Delay_diagnostic.png")
        pl.close(diag_fig)

        diag_fig = pl.figure()
        pl.plot(np.angle(zz), label="angle")
        pl.title("Delay acquisition diagnostic.\n total decimation: %d" % decimation)
        pl.xlabel("Samples")
        pl.ylabel("ADCu")
        pl.legend()
        pl.grid()
        pl.savefig("Delay_diagnostic_phase.png")
        pl.close(diag_fig)

        diag_fig = pl.figure()
        pl.title("Delay acquisition diagnostic.")
        Pxx = 20 * np.log10(Pxx)
        pl.xlabel("Frequency [Hz]")
        pl.ylabel("ADC dB")
        pl.semilogx(freq, Pxx, label="real")
        pl.legend()
        pl.grid()
        pl.savefig("Delay_diagnostic_FFT.png")
        pl.close(diag_fig)


    coeff = float(info['chirp_t'][0]) / float(np.abs(info['freq'][0] - info['chirp_f'][0]))

    print_debug("Coefficient is: "+str(coeff))

    print_debug("Max low freq found: "+str(freq[Pxx.argmax()]))

    delay = freq[Pxx.argmax()] * coeff

    delay = int(delay * 2e8) / 2.e8

    print_debug("Delay found %d ns"%int(delay*1e9))

    return delay

def load_delay_from_file(filename):
    '''
    Recover delay information from a delay file. Write this information in the internal dictionary LINE_DELAY.

    :param filename: Filename of the delay measure already analyzed.

    :return: rate in Msps, delay in ns

    Example:
        >>> # Once connected to the GPU server:
        >>> filename = measure_line_delay(...)
        >>> delay = analyze_line_delay(filename)
        >>> write_delay_to_file(filename, delay)
        >>> # Read back the result:
        >>> delay, rate = load_delay_from_file(filename)
    '''

    global LINE_DELAY

    try:
        filename = format_filename(filename)
    except:
        print_error("Cannot interpret filename while opening a H5 file in load_delay_from_file function")
        raise ValueError("Cannot interpret filename while opening a H5 file in load_delay_from_file function")

    f = bound_open(filename)

    try:
        rate = f["delay_info"].attrs.keys()[0]
        delay = f["delay_info"].attrs.get(rate)
    except ValueError:
        err_msg = "Cannot find any delay info in file "+str(filename)
        print_error(err_msg)
        raise ValueError(err_msg)

    try:
        rate = int(rate)
    except ValueError:
        err_msg = "Cannot convert rate stored on file to integer in load_delay_from_file function"
        print_error(err_msg)
        raise ValueError(err_msg)

    LINE_DELAY[str(rate)] = delay

    return rate, delay

def set_line_delay(rate, delay_ns):
    '''
    Manually set the LINE_DELAY internal variable.

    :param rate: reference rate for the delay in Msps.
    :param delay_ns: delay to set in ns.

    '''

    global LINE_DELAY

    try:
        if LINE_DELAY[str(rate/1e6)]: pass
        print_warning("Overwriting line delay for rate %d Msps"%rate)
    except KeyError:
        pass

    LINE_DELAY[str(rate)] = delay_ns

    print_debug("Line delay for rate %d Msps is set to %d ns"%(rate,delay_ns))

def load_delay_from_folder(foldername):
    '''
    Load line delay structure from folder matching the the acquisition rate with the right delay in the LINE_DELAY variable.
    
    Note:
    * The line delay is NOT matched with frequency but only with the sps rate of the USRP.

    :param foldername: folder containing the delay files. Delay files are detected using the filename string "USRP_Delay*.h5".
    '''

    try:
        os.chdir(foldername)
    except OSError:
        errmsg = "cannot find/access folder %s"%foldername
        print_error(errmsg)
        raise ValueError(errmsg)
    except WindowsError:
        errmsg = "cannot find/access folder %s" % foldername
        print_error(errmsg)
        raise ValueError(errmsg)

    filenames = glob.glob("USRP_Delay*.h5")

    for f in filenames:
        try:
            rate,delay = load_delay_from_file(f)
            print_debug("File %s, rate %d Msps, delay %d ns"%(f, rate, delay))
        except ValueError:
            print_debug("File %s - No delay info found")

    os.chdir("..")
    return
