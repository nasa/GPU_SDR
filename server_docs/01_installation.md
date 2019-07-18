System build
============

Hardware requirements
---------------------

There are hardware requiremets depending on the bandwidth the user intend to use.

  - __CPU__: The UHD libraries use the CPU to convert the data acquired with the SDR into buffers available to the GPU server. If the user is planning to use an overall bandwidth > 50 Msps consider using at leas a 7th gen i7 processor or equivalent. If the user is planning to use a bandwidth > 100 Msps the latest i9 CPU or equivalent is strongly suggested.

  _Suggested item from our tests: i9900k: supports 200 Msps full duplex streaming on two different channel using an usrp x300 while an i7-6800k support max 100Msps on a single channel_

  - __GPU__: The functionalities used in the kernels require CUDA >= 9.0. The hardware could be whatever support that cuda version. Take in account that the time required to process one packet must be inferior to the time needed to acquire it otherwise the real time feature of the GPU server is no longer in place. Often the execution time of a kernel well scales with the hardware (GPU) used.

  _Suggested item from our tests: RTX 2080Ti or RTX4000. We had great benefit from using memory faster than GDDR5_

  - __RAM__: It' suggested to have at least 8GB available for the GPU server alone. Since the buffer on which UHD operates is stored in the RAM a fast clock frequency could improve the available bandwidth.

  _Suggested item from our tests: max out the CPU supported frequency and use at least 16GB overall_

  - __NIC__: The data transport from the USRP to the host PC usually relies on 10 GB Ethernet. In our tests we had a x300 device connected via two 10 GBe cables to our PCIe network interface card.

  _Suggested item from our tests:

  - __PCIe extension__: Ettus Research also sells a PCIe card that directly interfaces with the USRPs. Our experience with it has been punctuated by driver compatibility problem we could not directly solve as the code is not open-source.

  _Suggested item from our tests: if you plan to develop this software on windows or use the USRP with other software on Windows this is good investment_

OS requirements
---------------------

The system has been developed and tested on Ubuntu 16 and 18 running the generic kernel. The GPU server has been also tested on RHEL 7 and Mac OsX and the python API has been also tested on Windows 10 and Mac OsX.

The only reason preventing the GPU server to be compiled in a windows environment is the reliance on the pthread library (used to control threads affinity and limit context switching on time critical processes)

Low-latency and Real-Time linux kernel have been tested (without improvement):
  * __Real Time__: The kernel broke some component of the CUDA driver on both versions 9.1 and 10.0. There was no chance to test the whole server performance however we performed some test using the rate_benchmark tool provided by Ettus Research and the performances resulted equal or inferior to the generic kernel.

  * __Low Latency__: The full software requirements were working correctly with the low latency kernel. Nothing changed in performance in respect to the generic kernel.


Installation
============

This section covers the necessary steps to compile and run the USRP_server. Every instruction here is thought to be executed on a clean system install that does not contain any relevant data. We do not take any responsibility for system corruption or data loss.

Pre-requisites
--------------
In order to proceed with the installation on Ubuntu the following programs are needed:
  * make
  * cmake
  * python 2.7
  * pip
  * python modules: requests mako and numpy

This guide will not cover the installation of these programs that can be obtained via packet manager. On Ubuntu:
\code{.sh}
$ apt-get update
$ apt-get -y install make cmake build-essential
$ apt-get -y install net-tools python-pip python-dev build-essential
$ sudo -H pip install --upgrade pip
$ sudo -H pip install requests mako numpy
\endcode

Requisites overview
-------------------
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
  \code{.sh}
  $ gcc --version
  $ g++ --version
  \endcode
  If the version of gcc and/or g++ is not in the range [6.0, 8.0) or if one of those programs is missing, you need to install and configure both programs.

    * First you need to download and install the right compiler. In this phase many Os help a lot by providing packets managers. In the case of Ubuntu 18.04 the following command install gcc 6.0:
    \code{.sh}
    $ sudo apt-get update
    $ sudo apt-get install gcc-6
    \endcode
    The same syntax is valid for g++.

    * To make sure that every installation uses the same compiler we'll make it default. On Ubuntu the commands are:
    \code{.sh}
    $ sudo update-alternatives --remove-all gcc
    $ sudo update-alternatives --remove-all g++
    $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
    $ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
    \endcode

  2. Install boost C++ library: the requirements on boost are ver > 1.66. This requirement mainly come from CUDA installation; UHD libraries will complain about possible missing requirements in configuration phase however this was not an issue during our tests. All our tests and development relied on boost 1.68. Although there are ways to install the boost libraries via packet manager we preferred not to use them to better control versioning and eventual uninstalling of the same. This guide will show the manual install. First download and unpack the boost libraries:
  go to https://www.boost.org/users/history/version_1_68_0.html and download  the boost_1_68_0.tar.gz.
  Unpack it with your preferred method and open a terminal in the folder just created. In the terminal type:
  \code{.sh}
  ./bootstrap.sh
  ./b2 -j <number of core of your machine>
  sudo ./b2 install
  sudo ldconfig
  \endcode
  Warnings are normal during the compile phase.
  In case you want to uninstall tor change the version of boost look for files named after \code{.sh}ibboost*\endcode in the \code{.sh}\usr\local\\endcode folder and delete them.

    \warning After installing CUDA some file named after \code{.sh} libboost*\endcode  will be present in the CUDA install path. Deletion of those files will require reinstallation of CUDA.

  3. Install the UHD libraries. This step could be performed using the packet manager however the version usually installed via apt-get is the 3.9 which is quite outdated even being the LTS branch of the distribution.
  The requirements on the UHD libraries are ver >=3.11. Before version 3.11 we had unstable performances and the firmware of the x300 hanged frequently upon tuning requests. 3.11 version has been for a long time our standard version and does not present problem of sorts except a very occasional hang of the x300 firmware upon tuning requests. The version 3.14 has better performances however presents some bug when an arbitrary delay is set on transmission of channels >0.

    * download or clone the libraries:
      \code{.sh}
      $ git clone https://github.com/EttusResearch/uhd.git
      \endcode
    * Switch to the branch of your choice (we suggest 3.14):
      \code{.sh}
      TBD
      \endcode
    * create the build folder:
      \code{.sh}
      $ mkdir build
      $ cd build
      \endcode
    * Configure the installation using cmake:
      \code{.sh}
      $ cmake ..
      \endcode

      \warning if you are installing UHD 3.11 you have to disble the octoclock feature for compiling to end successfully. the command in that case will be
      \code{.sh}
      $ cmake .. -DENABLE_OCTOCLOCK=false
      \endcode

      \note{} If you plan to use a USB device you have to manually enable that feature and get the \code{.sh}usblib\endcode from the packet manager. Check the output of Cmake to check if the feature has been effectively enabled.}

      \warning If you plan to use the DPDK (DPDK is only available starting from UHD ver 3.14) driver you must enable it with the command:
      \code{.sh}
      $ cmake .. -DPDK=true
      \endcode
      requisites and instructions for DPDK (move data transmission processes to user space, bind them to specific cores and supposedly improve performance) are not currently covered in this guide.

    * Build the library and install it in the system using make. Warnings are normal during the compile phase. For Ubuntu the commands are (staying in the previously created folder):
      \code{.sh}
      $ make -j <cores on your machine>
      $ sudo make install
      \endcode

  4. Install CUDA. You can install CUDA before the UHD libraries, it makes no difference. CUDA can also be installed via Packet manager after adding the right repository. In this guid we'll cover only the manual install for Ubuntu running Unity desktop environment.
    * Disable the nouveau driver: this is a generic graphic driver loaded by Ubuntu that will disrupt CUDA installation. To do it use the following commands:
    \code{.sh}
    $ sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    $ sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    \endcode
    Do not restart the system.

    * Download the appropriate runtime installer for your system: it can be easily obtained on the Nvidia website. For Ubuntu and CUDA 10.1 you can obtain it via command line:
    \code{.sh}
    $ wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
    \endcode
    Remember the location where you downloaded/saved the file.
    TIP: since sh does not support tabbing, rename the file to something simpler:
    \code{.sh}
    $ mv cuda_10.1.168_418.67_linux.run cuda_101.run
    \endcode

    * Install CUDA pre-requisites via packet-manager:
    \code{.sh}
    $ sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
    \endcode

    * Restart the system. After restart the graphic environment will have a very low resolution (if at all), just go in a tty terminal by pressing \code{.sh}Ctrl-Alt-F2\endcode and log in;

    * Kill the graphic environment manager. In Ubuntu with Uniti desktop manager you can do it with the command:
    \code{.sh}
    $ sudo service lightdm stop
    \endcode

    * install CUDA. Move to the cuda_101.run location and run it:
    \code{.sh}
    sudo sh cuda_101.run
    \endcode
    The screen may freeze for a while than the installation process will prompt you for various option. Keep all the default options. After the process finishes, restart your system, login and check that the Nvidia graphics card manager show the right card(s). If you cannot find the program to check, ensure that the Nvidia driver is up and running using:
    \code{.sh}
    $ sudo modprobe | grep Nv
    \endcode
    At least one entry should be listed. If not the CUDA installation process did not end successfully and you should investigate using the official Nvidia CUDA installation guide on the web.

    * Permanently add Nvidia toolkit libraries and compiler path as enviromental variable. You can do that with the commands:
    \code{.sh}
    $ sudo echo 'PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}' >> ~/.profile
    $ sudo echo 'LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile
    \endcode
    Check that you can use the toolkit by re-logging in the system and launching \code{.sh}nvcc --version\endcode

  5. Install HDF5 libraries. The HDF5 libraries are used for the server offline HDF5 local file write. They can be installed from source, freely available on the web or using the packet manager:
  \code{.sh}
  $ sudo apt-get install libhdf5-serial-dev
  \endcode
  This step is interchangable with whatever version of HDF5 libraries you want (OpenMPI, Parallel...)


  6. Compile the GPU server. Head to the main folder of this distribution and compile the server with make:
  \code{.sh}
  $ make -j <number of cores on your machine>
  \endcode
  At this point try to launch it with:
  \code{.sh}
  $ sudo ./server
  \endcode
  If the server hangs on \code{.sh}Looking for USRP...\endcode don't worry: this only means you have to configure your network or usb device. If you are using PCIe cable refer to the apposite section of this documentation.
