.. pyUSRP documentation master file, created by
   sphinx-quickstart on Fri Jan 25 16:56:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyUSRP's documentation!
==================================

This API has been developed with frequency multiplexed cryogenics detector in mind. Hard coded internal parameters may be different for different applications.

Quickstart guide
================
In order to use this library an instance of the GPU server (link) has to be already running and connected to a USRP device (see link).
First of all import the pyUSRP module::

  import pyUSRP as u

Than connect to the server::

  u.connect()

.. note::
  By default the library will connect to a GPU server that is running on the local machine. If the server is running on an other machine provide the address as a sting argument.

Once the client app is connected to the server launch any measure function provided with the distribution. Let's take a VNA scan between port TX/RX and RX2 on frontend A::

  vna_filename = u.Single_VNA(<your arguments>)

The function will block until the measurement is performed by the server and will create a local HDF5 file. The filename is than returned. In order to visualize the result of the VNA launch then the function::

  u.VNA_analysis(vna_filename)

That will read the raw data in the file and create the group VNA containing the frequency and S21 datasets. Finally for plotting launch::

  u.plot_VNA(vna_filename)

and optionally use the backend argument to choose between plotly and matplotlib. The matplotlib backend will save to disk a PNG file with the plot.

Examples
========
This section explain the use and purpose of the examples contained in the script folder of this distribution. For more information about the programs arguments launch each of the scripts with the -h argument.
The scope of each of this examples is to wrap library functions and complete a measurement or a simple analysis task.

.. note::
  All the programs that gather data have to interact with the GPU server via network.

.. note::
  These programs are intended as demonstration of the features of the system. They do NOT provide every option for functionality implemented.

get_VNA.py
----------
Save the S21 function raw data to disk. If multiple gain arguments or LO frequency arguments are given, produces multiple scans. By default saves the data to the scripts/data/ folder.

analyze_VNA.py
--------------
Analyze all the USRP_VNA* files present in the scripts/data/ folder. Overlay all the analyzed VNA files in a single plot.

fit_VNA.py
----------
Fit the pre-analyzed VNA file with kinetic inductance resonator function and saves the fits results in the resonator group inside the same file.
Two ways of initializing the peaks have been implemented: one that estimates the number and position of the peaks using a threshold on the derivative of S21; the other that given a number of peaks scan the S21 function for the best candidates.

After fitting the program plot the resonators fits and the S21 with the resonator tags.

get_noise.py
------------
Multitone noise acquisition. This program acquires multitone data. The tones can be initialized from a previously fitted VNA scan (using just the filename) or manually.
Two different acquisition modes have been implemented: the direct demodulation mode and the PFB mode.

analyze_noise.py
----------------
Calculates and save in the file the dBc/dBm spectrum of a multitone noise acquisition.

freqts_plot.py
--------------
Translate tones power fluctuations into quality factor and resonant frequency fluctuation using VNA fit data.

get_line_delay.py
-----------------
Estimate the loop line delay between two ports using a short chirped signal.

get_noise_full.py
-----------------
Use the USRP as spectrum analyzer and saves data to disk

.. note::
  The client/server data rate can easily exceed the network buffer causing the client or the server to freeze if the connection between the does not have enough bandwidth for certain parameters.

plot_spectrogram.py
-------------------
Plot the full spectrum noise acquisition.


swipe_parameters.py
-------------------
This code example is the most similar to our acquisition routine.
Acquire a VNA scan, analyze and fit it. Use the fit result to take a multitone noise acquisition and change some parameter (in this case the TX gain)


Useful functions
================

Connections
-----------
.. autosummary::
  USRP_connections.Connect
  USRP_connections.Disconnect

Launching measures
------------------
.. autosummary::
  USRP_VNA.Single_VNA
  USRP_noise.Get_noise
  USRP_delay.measure_line_delay
  USRP_full_spec.get_NODSP_tones
  USRP_noise.Get_full_spec

Getting data from files
-----------------------
.. autosummary::
  USRP_files.openH5file
  USRP_files.get_rx_info
  USRP_files.get_tx_info
  USRP_files.get_noise
  USRP_files.get_trigger_info
  USRP_files.get_readout_power
  USRP_files.global_parameter.retrive_prop_from_file
  USRP_files.get_VNA_data
  USRP_files.get_dynamic_VNA_data
  USRP_files.get_init_peaks
  USRP_fitting.get_tones
  USRP_fitting.get_best_readout
  USRP_fitting.get_fit_param
  USRP_fitting.get_fit_data
  USRP_delay.load_delay_from_file
  USRP_delay.load_delay_from_folder
  USRP_full_spec.Get_full_spec
  USRP_noise.get_frequency_timestreams

Move data between files
-----------------------
.. autosummary::
  USRP_fitting.initialize_from_VNA
  USRP_delay.write_delay_to_file
  USRP_noise.copy_resonator_group



Analyze data
------------
.. autosummary::
  USRP_fitting.vna_fit
  USRP_fitting.initialize_peaks
  USRP_fitting.extimate_peak_number
  USRP_delay.analyze_line_delay
  USRP_noise.calculate_noise
  USRP_VNA.VNA_timestream_analysis
  USRP_VNA.VNA_analysis

Plotting data
-------------
.. autosummary::
  USRP_fitting.plot_resonators
  USRP_full_spec.plot_pfb
  USRP_noise.plot_noise_spec
  USRP_noise.plot_frequency_timestreams
  USRP_noise.diagnostic_VNA_noise
  USRP_plotting.plot_raw_data
  USRP_VNA.VNA_timestream_plot
  USRP_VNA.plot_VNA

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

The "Connections" module
------------------------

*This module consist groups the functions and classes that make the data exchange with the GPU USRP server.*

.. automodule:: USRP_connections
    :members:

The "Data analysis" module
--------------------------

*This module contains all the functions used to analyze the data take with this software. This module is very application specific: all the functions needed for general purpose analysis (like ffts) are in their respective submodule (i.e. noise)*

.. automodule:: USRP_data_analysis
    :members:

The "Delay" module
------------------

*MAny applications, like the VNA here implemented, requires the knowledge of the line delay. With line delay it is intended the delay introduced by the loop connecting a TX port of the USRP to an active RX port of the USRP.*

.. automodule:: USRP_delay
    :members:

The "Files" module
------------------

*This module containd the classes and the functions used to interact and create the HDF5 files containing the data coming from the USRP GPU system*

.. automodule:: USRP_files
    :members:

The "Fitting" module
--------------------

*This is a homebrew module to find, fit, store and plot the cryogenic resonators S21 profiles*

.. automodule:: USRP_fitting
    :members:

The "Noise" module
------------------

*Contains the functions needed to acquire, analyze and plot single tones noise data from the resonators.*

.. automodule:: USRP_noise
    :members:

The "Low level" module
----------------------

*Has all the boring functions needed to the other modules to work correctly*

.. automodule:: USRP_low_level
    :members:

The "Plotting" module
----------------------

*Contains a few settings to modify the general look of the plotting plus some raw data plotting function.*

.. automodule:: USRP_plotting
    :members:

The "VNA" module
----------------

*Has the functions needed to acquire, analyze and plot VNAs acquisitions*

.. automodule:: USRP_VNA
    :members:
