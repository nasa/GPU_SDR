System build
============

Hardware requirements
---------------------

There are hardware requiremets depending on the bandwidth the user intend to use.

  - __CPU__: The UHD libraries use the CPU to convert the data acquired with the SDR into buffers available to the GPU server. If the user is planning to use an overall bandwidth > 50 Msps consider using at leas a 7th gen i7 processor or equivalent. If the user is planning to use a bandwidth > 100 Msps the latest i9 CPU or equivalent is strongly suggested.

  _Suggested item from our tests: i9900k: supports 200 Msps full duplex streaming on two different channel using an usrp x300 while an i7-6800k support max 100Msps on a single channel_

  - __GPU__: The functionalities used in the kernels require CUDA >= 9.0. The hardware could be whatever support that cuda version. Take in account that the time required to process one packet must be inferior to the time needed to acquire it otherwise the real time feature of the GPU server is no longer in place. Often the execution time of a kernel well scales with the hardware (GPU) used.

  _Suggested item from our tests: RTX 2070 or GTX1080Ti. We had great benefit from using memory faster than GDDR5_

  - __RAM__: It' suggested to have at least 8GB available for the GPU server alone. Since the buffer on which UHD operates is stored in the RAM a fast clock frequency could improve the available bandwidth.

  _Suggested item from our tests: max out the CPU supported frequency and use at least 16GB overall_

  - __NIC__: The data transport from the USRP to the host PC usually relies on 10 GB Ethernet. In our tests we had a x300 device connected via two 10 GBe cables to our PCIe network interface card.

  _Suggested item from our tests:

  - __PCIe extension__: Ettus Research also sells a PCIe card that directly interfaces with the USRPs. Our experience with it has been punctuated by driver compatibility problem we could not directly solve as the code is not open-source.

  _Suggested item from our tests: if you plan to develop this software on windows or use the USRP with other software on Windows this is good investment_

OS requirements
---------------------

The system has been developed and tested on Ubuntu 16 and 18 running the generic kernel. The GPU server has been also tested on RHEL 7 and Mac OsX and the python API has been also tested on Windows 10 and Mac OsX.

The only reason preventing the GPU server to be compiled in a windows environment is the reliance on the pthread library (used to control threads affinity and limit context switching)

Low-latency and Real-Time linux kernel have been tested (without improvement):
  * __Real Time__: The kernel broke some component of the CUDA driver on both versions 9.1 and 10.0. There was no chance to test the whole server performance however we performed some test using the rate_benchmark tool provided by Ettus Research and the performances resulted equal or inferior to the generic kernel.

  * __Low Latency__: The full software requirements were working correctly with the low latency kernel. Nothing changed in performance in respect to the generic kernel.


Installation
============

This section covers the necessary steps to compile and run the USRP_server. Every instruction here is thought to be executed on a clean system install that does not contain any relevant data. We do not take any responsibility for system corruption or data loss.

Pre-requisites
--------------
This is just a list of software and libraries needed to compile the GPU server, the next section will detail every step.
  * gcc and g++ with version >=6.0 and <8.0
  * make
  * CUDA driver and toolkit >9.0
  * C++ boost libraries ver > 1.66
  * HDF5 C++ libraries
  * UHD libraries ver >=3.11

General guide
-------------

The commands in this guide have been tested on Ubuntu 18.04. The description of the steps is applicable to every linux-based system.

  1. Make sure your system is using the correct version of gcc and g++. You can check what version will be used by default by typing:
  '''
  gcc --version
  '''
  If the version of gcc and/or g++ is not in the range [6.0, 8.0) you need to install and configure both programs.


Ubuntu 18.04
------------
