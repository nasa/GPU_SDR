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
from USRP_low_level import *
from USRP_files import *


def get_color(N):
    '''
    Get a color for the list above without exceeding bounds. Usefull to change the overall color scheme.

    Arguments:
        N: index identifying stuff that has to have the same color.

    Return:
        string containing the color name.
    '''
    N = int(N)
    return COLORS[N % len(COLORS)]


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
        - output_filename: string: if given the function saves the plot (in png for matplotlib backend and html for plotly backend) with the given name.
        - kwargs: usrp_number and front_end can be passed to the openH5file() function. tx_front_end can be passed to manually determine the tx frontend to calculate the readout power. add_info could be a list of the same length og filenames containing additional leggend informations.
    '''
    filenames = to_list_of_str(filenames)

    add_info_labels = None
    try:
        add_info_labels = kwargs['add_info']
    except KeyError:
        pass

    plot_title = 'USRP Noise spectra from '
    if len(filenames) < 2:
        plot_title += "file: " + filenames[0] + "."
    else:
        plot_title += "multiple files."

    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=1, ncols=1)
        try:
            fig.set_size_inches(kwargs['size'][0], kwargs['size'][1])
        except KeyError:
            pass
        ax.set_xlabel("Frequency [Hz]")

    elif backend == 'plotly':
        fig = tools.make_subplots(rows=1, cols=1)
        fig['layout']['xaxis1'].update(title="Frequency [Hz]", type='log')

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
            label = "%.2f MHz" % (info['tones'][i] / 1e6)
            R = 10 * np.log10(real[i])
            I = 10 * np.log10(imag[i])
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
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=7)
        ax.grid(True)
        if output_filename is not None:
            fig.savefig(output_filename + '.png')
        if auto_open:
            pl.show()
        else:
            pl.close()

    elif backend == 'plotly':
        if title_info is not None:
            plot_title += "<br>" + title_info

        fig['layout'].update(title=plot_title)
        if output_filename is None:
            output_filename = "raw_data_plot"

        plotly.offline.plot(fig, filename=output_filename + ".html", auto_open=auto_open)


def plot_raw_data(filenames, decimation=None, low_pass=None, backend='matplotlib', output_filename=None,
                  channel_list=None, mode='IQ', start_time=None, end_time=None, auto_open=True, **kwargs):
    '''
    Plot raw data group from given H5 files.

    Arguments:
        - a list of strings containing the files to plot.
        - decimation: eventually deciamte the signal before plotting.
        - low pass: floating point number controlling the cut-off frequency of a low pass filter that is eventually applied to the data.
        - backend: [string] choose the return type of the plot. Allowed backends for now are:
            * matplotlib: creates a matplotlib figure, plots in non-blocking mode and return the matplotlib figure object. kwargs in this case accept:
                - size: size of the plot in the form of a tuple (inches,inches). Default is matplotlib default.
            * plotly: plot using plotly and webgl interface, returns the html code descibing the plot. kwargs in this case accept:
                - size: size of the plot. Default is plotly default.
            * bokeh: use bokeh to generate an interactive html file containing the IQ plane and the magnitude/phase timestream.

        - output_filename: string: if given the function saves the plot (in png for matplotlib backend and html for plotly backend) with the given name.
        - channel_list: select only al list of channels to plot.
        - mode: [string] how to print the IQ signals. Allowed modes are:
            * IQ: default. Just plot the IQ signal with no processing.
            * PM: phase and magnitude. The fase will be unwrapped and the offset will be removed.
        - start_time: time where to start plotting. Default is 0.
        - end_time: time where to stop plotting. Default is end of the measure.
        - auto_open: open the plot in default system browser if plotly backend is selected (non-blocking) or open the matplotlib figure (blocking). Default is True.
        - kwargs:
            * usrp_number and front_end can be passed to the openH5file() function.
            * size: the size of matplotlib figure.
            * add_info: list of strings as long as the file list to add info to the legend.

    Returns:
        - the complete name of the saved file or None in case no file is saved.

    Note:
        - Possible errors are signaled on the plot.
    '''
    try:
        add_info_labels = kwargs['add_info']
    except KeyError:
        add_info_labels = None
    plot_title = 'USRP raw data acquisition. '
    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)
        try:
            fig.set_size_inches(kwargs['size'][0], kwargs['size'][1])
        except KeyError:
            pass
        ax[1].set_xlabel("Time [s]")
        if mode == 'IQ':
            ax[0].set_ylabel("I [fp ADC]")
            ax[1].set_ylabel("Q [fp ADC]")
        elif mode == 'PM':
            ax[0].set_ylabel("Magnitude [abs(ADC)]")
            ax[1].set_ylabel("Phase [Rad]")

    elif backend == 'plotly':
        if mode == 'IQ':
            fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('I timestream', 'Q timestream'),
                                      shared_xaxes=True)
            fig['layout']['yaxis1'].update(title='I [fp ADC]')
            fig['layout']['yaxis2'].update(title='Q [fp ADC]')
        elif mode == 'PM':
            fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Magnitude', 'Phase'), shared_xaxes=True)
            fig['layout']['yaxis1'].update(title='Magnitude [abs(ADC)]')
            fig['layout']['yaxis2'].update(title='Phase [Rad]')

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

        samples, errors = openH5file(
            filename,
            ch_list=channel_list,
            start_sample=file_start_time,
            last_sample=file_end_time,
            usrp_number=usrp_number,
            front_end=front_end,
            verbose=False,
            error_coord=True
        )

        print_debug("plot_raw_data() found %d channels each long %d samples" % (len(samples), len(samples[0])))
        if channel_list == None:
            ch_list = range(len(samples))
        else:
            if max(channel_list) > len(samples):
                print_warning(
                    "Channel list selected in plot_raw_data() is bigger than avaliable channels. plotting all available channels")
                ch_list = range(len(samples))
            else:
                ch_list = channel_list

        # prepare samples TODO
        for i in ch_list:

            if mode == 'IQ':
                Y1 = samples[i].real
                Y2 = samples[i].imag
            elif mode == 'PM':
                Y1 = np.abs(samples[i])
                Y2 = np.angle(samples[i])

            if decimation is not None and decimation > 1:
                decimation = int(np.abs(decimation))
                Y1 = signal.decimate(Y1, decimation, ftype='fir')
                Y2 = signal.decimate(Y2, decimation, ftype='fir')
            else:
                decimation = 1.

            X = np.arange(len(Y1)) / float(effective_rate / decimation) + file_start_time

            if effective_rate / 1e6 > 1:
                rate_tag = 'DAQ rate: %.2f Msps' % (effective_rate / 1e6)
            else:
                rate_tag = 'DAQ rate: %.2f ksps' % (effective_rate / 1e3)

            if freq is None:
                label = "Channel %d" % i
            else:
                label = "Channel %.2f MHz" % (freq[i] / 1.e6)

            if backend == 'matplotlib':
                if add_info_labels is not None:
                    label += "\n" + add_info_labels[file_count]
                ax[0].plot(X, Y1, color=get_color(i + file_count), label=label)
                ax[1].plot(X, Y2, color=get_color(i + file_count))
            elif backend == 'plotly':
                if add_info_labels is not None:
                    label += "<br>" + add_info_labels[file_count]
                fig.append_trace(go.Scatter(
                    x=X,
                    y=Y1,
                    name=label,
                    legendgroup="group" + str(i) + "file" + str(file_count),
                    line=dict(color=get_color(i + file_count)),
                    mode='lines'
                ), 1, 1)
                fig.append_trace(go.Scatter(
                    x=X,
                    y=Y2,
                    # name = "channel %d"%i,
                    showlegend=False,
                    legendgroup="group" + str(i) + "file" + str(file_count),
                    line=dict(color=get_color(i + file_count)),
                    mode='lines'
                ), 2, 1)
        file_count += 1
    if backend == 'matplotlib':
        for error in errors:
            err_start_coord = (error[0] - decimation / 2) / float(effective_rate) + file_start_time
            err_end_coord = (error[1] + decimation / 2) / float(effective_rate) + file_start_time
            ax[0].axvspan(err_start_coord, err_end_coord, facecolor='yellow', alpha=0.4)
            ax[1].axvspan(err_start_coord, err_end_coord, facecolor='yellow', alpha=0.4)
        fig.suptitle(plot_title + "\n" + rate_tag)
        handles, labels = ax[0].get_legend_handles_labels()
        if len(errors) > 0:
            yellow_patch = mpatches.Patch(color='yellow', label='ERRORS')
            handles.append(yellow_patch)
            labels.append('ERRORS')
        fig.legend(handles, labels, loc=7)
        ax[0].grid(True)
        ax[1].grid(True)
        if output_filename is not None:
            fig.savefig(output_filename + '.png')
        if auto_open:
            pl.show()
        else:
            pl.close()

    if backend == 'plotly':

        fig['layout'].update(title=plot_title + "<br>" + rate_tag)

        if output_filename is None:
            output_filename = "PFB_waterfall"
        plotly.offline.plot(fig, filename=output_filename + ".html", auto_open=auto_open)


def plot_all_pfb(filename, decimation=None, low_pass=None, backend='matplotlib', output_filename=None, start_time=None,
                 end_time=None, auto_open=True, **kwargs):
    '''
    Plot the output of a PFB acquisition as an heatmap.
    '''
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

    y_label = np.arange(len(samples[0]) / fft_tones) / (rate / (fft_tones * max(1, decimation)))
    x_label = (rf + (np.arange(fft_tones) - fft_tones / 2) * (rate / fft_tones)) / 1e6
    title = "PFB acquisition form file %s" % filename
    subtitle = "Channel width %.2f kHz; Frame integration time: %.2e s" % (channel_width / 1.e3, integ_time)
    z = 20 * np.log10(np.abs(samples[0]))
    try:
        z_shaped = np.roll(np.reshape(z, (len(z) / fft_tones, fft_tones)), fft_tones / 2, axis=1)
    except ValueError as msg:
        print_warning("Error while plotting pfb spectra: " + str(msg))
        cut = len(z) - len(z) / fft_tones * fft_tones
        z = z[:-cut]
        print_debug("Cutting last data (%d samples) to fit" % cut)
        # z_shaped = np.roll(np.reshape(z,(len(z)/fft_tones,fft_tones)),fft_tones/2,axis = 1)
        z_shaped = np.roll(np.reshape(z, (len(z) / fft_tones, fft_tones)), fft_tones / 2, axis=1)

    # pl.plot(z_shaped.T, alpha = 0.1, color = "k")
    # pl.show()

    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)
        try:
            fig.set_size_inches(kwargs['size'][0], kwargs['size'][1])
        except KeyError:
            pass
        ax[0].set_xlabel("Channel [MHz]")
        ax[0].set_ylabel("Time [s]")
        ax[0].set_title(title + "\n" + subtitle)
        imag = ax[0].imshow(z_shaped, aspect='auto', interpolation='nearest',
                            extent=[min(x_label), max(x_label), min(y_label), max(y_label)])
        # fig.colorbar(imag)#,ax=ax[0]
        for zz in z_shaped[::100]:
            ax[1].plot(x_label, zz, color='k', alpha=0.1)
        ax[1].set_xlabel("Channel [MHz]")
        ax[1].set_ylabel("Power [dBm]")
        # ax[1].set_title("Trace stack")

        if output_filename is not None:
            fig.savefig(output_filename + '.png')
        if auto_open:
            pl.show()
        else:
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
        if output_filename is None:
            output_filename = "PFB_waterfall"
        plotly.offline.plot(fig, filename=output_filename + ".html", auto_open=auto_open)