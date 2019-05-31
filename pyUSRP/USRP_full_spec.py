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
import warnings
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

def get_NODSP_tones(tones, measure_t, rate, amplitudes = None, RF = None, tx_gain = 0, output_filename = None, Front_end = None,
              Device = None, delay = None, **kwargs):
    '''
    Perform a noise acquisition using fixed tone technique without demodulating the output result.
    It does not perform any dignal processing operation on the data.

    Arguments:
        - tones: list of tones frequencies in Hz (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - amplitudes: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - tx_gain: gain to use in the transmission side.
        - output_filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - delay: delay between TX and RX processes. Default is taken from the INTERNAL_DELAY variable.
        - kwargs:
            * verbose: additional prints. Default is False.
            * push_queue: queue for post writing samples.

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

    number_of_samples = rate * measure_t

    expected_samples = number_of_samples
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
    noise_command.set(TX_frontend, "fft_tones", 1e8)

    noise_command.set(RX_frontend, "mode", "RX")
    #noise_command.set(RX_frontend, 'tuning_mode', 0)
    noise_command.set(RX_frontend, "buffer_len", 1e6)
    noise_command.set(RX_frontend, "gain", 0)
    noise_command.set(RX_frontend, "delay", 1 + delay)
    noise_command.set(RX_frontend, "samples", number_of_samples)
    noise_command.set(RX_frontend, "rate", rate)
    noise_command.set(RX_frontend, "bw", 2 * rate)

    noise_command.set(RX_frontend, "wave_type", ["NODSP",])
    noise_command.set(RX_frontend, "freq", tones)
    noise_command.set(RX_frontend, "rf", RF)


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
        meas_type="Raw_data",
        push_queue = push_queue,
        **kwargs
    )

    print_debug("Noise acquisition terminated.")

    return output_filename

def Get_full_spec(tones, channels, measure_t, rate, RF = None, Front_end = None, amplitudes = None, tx_gain=0, decimation = None, pf_average = 4, output_filename = None, delay = None, **kwargs):
    '''
    Full spectrum version of the Get_noise() function.

    :param tones: tones to put in output.
    :param channels: number of channels to read. The minimum will be determined by the number and spacing of tones.
    :param measure_t: acquisition duration.
    :param rate: acquisition rate.
    :param RF: central rf frequency.
    :param tx_gain: gain of the transmit amplifier. Default is 0.
    :param decimation: decimation on the PFB spectra. DEfault is None equivalent to 0 and 1.
    :param pf_average: internal parameter of the pfb. default is 4.
    :param output_filename: output filename. Default is USRP_PFB_ + datetime.
    :param Front_end: the front_end to use for the acquisition.

    Keyword arguments:
        ::param verbose print some debug line

    :return String containing the name of the saved file.
    '''
    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE, LINE_DELAY, USRP_power

    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    if output_filename is None:
        output_filename = "USRP_PFB_"+get_timestamp()
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

    print("Tone [MHz]\tPower [dBm]\tOffset [MHz]")
    for i in range(len(tones)):
        print("%.1f\t%.2f\t%.1f" % ((RF + tones[i]) / 1e6, USRP_power + 20 * np.log10(amplitudes[i]), tones[i] / 1e6))
        if tones[i] > rate / 2:
            print_error("Out of bandwidth tone!")
            raise ValueError("Out of bandwidth tone requested. %.2f MHz / %.2f MHz (Nyq)" %(tones[i]/1e6, rate / 2e6) )

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

    if channels < min_required_fft:
        print_warning("Cannot use a channels factor of %d as the minimum required number of bin in the PFB is %d" % (decimation, min_required_fft))
        final_fft_bins = min_required_fft
    else:
        final_fft_bins = int(channels)


    if final_fft_bins<10:
        print_warning("Less that 10 pfb bins cause a bottleneck in the GPU: too many instruction to fetch. Setting channels to 10")
        final_fft_bins = 10

    print_debug("Using %d PFB channels"%final_fft_bins)

    if decimation == 1 or decimation == 0:
        decimation = None

    if decimation is not None:
        rx_number_of_samples = rate * final_fft_bins / decimation
    else:
        rx_number_of_samples = rate * final_fft_bins

    tx_number_of_samples = rate * measure_t

    noise_command = global_parameter()

    noise_command.set(TX_frontend, "mode", "TX")
    noise_command.set(TX_frontend, "buffer_len", 1e6)
    noise_command.set(TX_frontend, "gain", tx_gain)
    noise_command.set(TX_frontend, "delay", 1)
    noise_command.set(TX_frontend, "samples", tx_number_of_samples)
    noise_command.set(TX_frontend, "rate", rate)
    noise_command.set(TX_frontend, "bw", 2 * rate)

    noise_command.set(TX_frontend, "wave_type", ["TONES" for x in tones])
    noise_command.set(TX_frontend, "ampl", amplitudes)
    noise_command.set(TX_frontend, "freq", tones)
    noise_command.set(TX_frontend, "rf", RF)

    # This parameter does not have an effect (except suppress a warning from the server)
    noise_command.set(TX_frontend, "fft_tones", 100)

    noise_command.set(RX_frontend, "mode", "RX")
    noise_command.set(RX_frontend, "buffer_len", 1e6)
    noise_command.set(RX_frontend, "gain", 0)
    noise_command.set(RX_frontend, "delay", 1 + delay)
    noise_command.set(RX_frontend, "samples", tx_number_of_samples)
    noise_command.set(RX_frontend, "rate", rate)
    noise_command.set(RX_frontend, "bw", 2 * rate)

    noise_command.set(RX_frontend, "wave_type", ["NOISE",])
    noise_command.set(RX_frontend, "freq", tones)
    noise_command.set(RX_frontend, "rf", RF)
    noise_command.set(RX_frontend, "fft_tones", final_fft_bins)
    noise_command.set(RX_frontend, "pf_average", pf_average)


    if decimation is not None:
        noise_command.set(RX_frontend, "decim", decimation)
    else:
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
        dpc_expected=rx_number_of_samples,
        meas_type="PFB", **kwargs
    )

    print_debug("Full spec noise acquisition terminated.")

    return output_filename


def plot_pfb(filename, decimation=None, low_pass=None, backend='matplotlib', output_filename=None, start_time=None,
                 end_time=None, auto_open=True, **kwargs):
    '''
    Plot the output of a PFB acquisition as an heatmap.

    :retrurn Name of the file.
    '''
    final_filename = ""
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    ant = parameters.get_active_rx_param()
    try:
        usrp_number = kwargs['usrp_number']
    except KeyError:
        usrp_number = None
    if len(ant) > 1:
        print_error("multiple RX devices not yet supported")
        return

    if parameters.get(ant[0], 'wave_type')[0] != "NOISE":
        print_warning("The file selected does not have the PFB acquisition tag. Errors may occour")

    fft_tones = parameters.get(ant[0], 'fft_tones')
    rate = parameters.get(ant[0], 'rate')
    channel_width = rate / fft_tones
    decimation = parameters.get(ant[0], 'decim')
    integ_time = fft_tones * max(decimation, 1) / rate
    rf = parameters.get(ant[0], 'rf')
    if start_time is not None:
        start_time *= effective_rate
    else:
        start_time = 0
    if end_time is not None:
        end_time *= effective_rate

    try:
        front_end = kwargs['front_end']
    except KeyError:
        front_end = None
    samples, errors = openH5file(
        filename,
        ch_list=None,
        start_sample=start_time,
        last_sample=end_time,
        usrp_number=usrp_number,
        front_end=front_end,
        verbose=False,
        error_coord=True
    )
    if output_filename is None:
        output_filename = "PFB_waterfall_"+get_timestamp()

    y_label = np.arange(len(samples[0]) / fft_tones) / (rate / (fft_tones * max(1, decimation)))
    x_label = (rf + (np.arange(fft_tones) - fft_tones / 2) * (rate / fft_tones)) / 1e6
    title = "PFB acquisition form file %s" % filename
    subtitle = "Channel width %.2f kHz; Frame integration time: %.2e s" % (channel_width / 1.e3, integ_time)
    with warnings.catch_warnings():
        # it's very likely to do some division by 0 in the log10
        warnings.simplefilter("ignore")
        z = 20 * np.log10(np.abs(samples[0]))
    try:
        z_shaped = np.roll(np.reshape(z, (len(z) / fft_tones, fft_tones)), -fft_tones / 2, axis=1)
    except ValueError as msg:
        print_warning("Error while plotting pfb spectra: " + str(msg))
        cut = len(z) - len(z) / fft_tones * fft_tones
        z = z[:-cut]
        print_debug("Cutting last data (%d samples) to fit" % cut)
        # z_shaped = np.roll(np.reshape(z,(len(z)/fft_tones,fft_tones)),fft_tones/2,axis = 1)
        z_shaped = np.roll(np.reshape(z, (len(z) / fft_tones, fft_tones)), -fft_tones / 2, axis=1)

    # pl.plot(z_shaped.T, alpha = 0.1, color = "k")
    # pl.show()

    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=2, ncols=1,)# sharex=True)
        try:
            fig.set_size_inches(kwargs['size'][0], kwargs['size'][1])
        except KeyError:
            fig.set_size_inches(10, 16)
        ax[0].set_xlabel("Channel [MHz]")
        ax[0].set_ylabel("Time [s]")
        ax[0].set_title(title + "\n" + subtitle)
        ax[0].set_xlim((min(x_label), max(x_label)))
        imag = ax[0].imshow(z_shaped, aspect='auto', interpolation='nearest',
                            extent=[min(x_label), max(x_label), min(y_label), max(y_label)])
        # fig.colorbar(imag)#,ax=ax[0]
        for zz in z_shaped[::100]:
            ax[1].plot(x_label, zz, color='k', alpha=0.1)
        ax[1].set_xlabel("Channel [MHz]")
        ax[1].set_ylabel("Power [dBm]")
        ax[1].set_xlim((min(x_label), max(x_label)))
        # ax[1].set_title("Trace stack")

        final_filename = output_filename + '.png'
        fig.savefig(final_filename)

        pl.close()

    if backend == 'plotly':
        data = [
            go.Heatmap(
                z=z_shaped,
                x=x_label,
                y=y_label,
                colorscale='Viridis',
            )
        ]

        layout = go.Layout(
            title=title + "<br>" + subtitle,
            xaxis=dict(title="Channel [MHz]"),
            yaxis=dict(title="Time [s]")
        )

        fig = go.Figure(data=data, layout=layout)
        final_filename = output_filename + ".html"
        plotly.offline.plot(fig, filename=final_filename, auto_open=auto_open)

    return final_filename
