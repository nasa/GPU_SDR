Overview
========

An overview of the working principle is reported in the figure below with some system requirements.
<div class="right_float">
\image html general_plot.png "General scheme:"Block representation of the readout system. This software runs on the "Readout server" in the figure.
</div>
The goal of this project is to provide a tool to read frequency multiplexed that is flexible enough to service a variety of technologies and, at the same time, easy enouh to develop so that innovative algorithms can be tested without having to deal with an FPGA firmware. Most of the computational operation required by the readout operations are offloaded on the GPU: the firmware present on the USRP device is the stock firmware without any modification. This firmware is used only to communicate data and metadata with the USRP device and does not perform (almost) any operation. Once the data are streamed on the host the server upload them on the GPU memory where they will be analysed. Thanks to the massive parallelization of operation on the GPU device the analysis operations are completed before the next packet of data is uploaded on the GPU. The flexibility of this system relies in the range of operation that can be performed on the data using a Nvidia GPU: thanks to the CUDA language a new signal processing operation can be implemented and tested in a timescale much shorter that the FPGA firmware equivalent. Instructions on how to implement a new algorrithm on the server are contained in the apposite section. Once the data have been analyzed on the GPU they are streamed to a client application via a TCP socket. The client provvided within the system consists of a python library that saves to disk the results and/or plot the data in real time.

Workflow
--------

Once the server is launced, it stastes looking for USRP devices connected to the system. Once at least one device has been found the server starts accepting connections. Two different TCP connection on two different ports (defined at compile time in USRP_server_settings.cpp in variables #TCP_SYNC_PORT and #TCP_ASYNC_PORT) and same socket are accepted and necessary for the system to work. One is a connection used for exchanging messages between server and client, the other is a connection dedicated to data exchange between server and client. Once the connections have been established the server expect a JSON string describing a command: the command describes settings for the USRP hardware and settings for the signal processing/generation needed. Once the JSON command string has been received and validated (settings have to be reasonable) the hardware configuration is applied using the Ettus UHD C++ api to set the USRP FPGA registers and the GPU kernels and meory are initialized accordigly to the requirements in the JSON command. Many threads are spawned in this phase: two different threads for the transmission handling and a minimum of three threads for the receiver side.
The measurement can terminate for different reasons: number of samples in input/output requested has been reached, the client has disconnected or the USRP devices encountered an error.
When the mesuremnt terminates, all the GPU memory is released, all the queues between threads are clean and every thread (except for memoyr managers and main) is joined; the servers goes in IDLE state waiting for a new command.

Datapath
--------

Data are moved around the server using the lock-free queue implementation of boost libraries. Those queues (defined in USRP_server_settings.cpp) are queues of pointers to different kind of struct, these structs serve different purposes but relies on the same structure: a pointer to the memory and some metadata usefull to read the memory.

<div class="right_float">
\image html Memory_model.png "RX datapath scheme:"Block representation of the datapath in the receiver side.
</div>

<div class="left_float">
\image html memory_tx.png "TX datapath scheme:"Block representation of the datapath in the transmission side.
</div>

Memory
----------------
<div class="left_float">
\image html Memory_allocator.png Block diagram of the memory manager class used in the server.
</div>
