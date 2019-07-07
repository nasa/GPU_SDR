.. pyUSRP documentation master file, created by
   sphinx-quickstart on Fri Jan 25 16:56:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyUSRP's documentation!
==================================

This API has been developed with frequency multiplexed cryogenics detector in mind. Hard coded internal parameters may be different for diffewrent applications.

Quickstart guide
================
In order to use this library an instance of the GPU server (link) has to be already running and connected to a USRP device (see link).
First of all import the pyUSRP module::

  import pyUSRP as u

Than connect to the server::

  u.connect()

<div class="alert alert-warning" role="alert">
  By default the library will connect to a GPU server that is running on the local machine. If the server is running on an other machine provide the address as a sting argument.
</div>

Once the client app is connected to the server launch any measure function provided with the distribution. Let's take a VNA scan between port TX/RX and RX2 on frontend A::

  vna_filename = u.Single_VNA()
  
The function will block until the measurement is performed by the server and will create a local HDF5 file. The filename is than returned.

Examples
========
This section explain the use and purpose of the examples contained in the script folder of this distribution.

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

*This module contains all the functions used to analyze the data take with this software. This module is very application specific so if you are not using Kinetic Inductance Detectors it may not well suite your needs. If KIDs are used, particularly if Thermal KIDs are used, the function in this module will cast the data to meaningful physical values.*

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
