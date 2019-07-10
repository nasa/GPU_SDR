Overview
========

An overview of the working principle is reported in the figure below with some system requirements.

<p>
 <img src="general_plot.png" alt="General scheme" style="width:70%">
 <center><em>General scheme:Block representation of the readout system. This software runs on the "Readout server" in the figure.</em></center>
<p>

The goal of this project is to provide a tool to read frequency multiplexed that is flexible enough to service a variety of technologies and, at the same time, easy enouh to develop so that innovative algorithms can be tested without having to deal with an FPGA firmware. Most of the computational operation required by the readout operations are offloaded on the GPU: the firmware present on the USRP device is the stock firmware without any modification. This firmware is used only to communicate data and metadata with the USRP device and does not perform (almost) any operation. Once the data are streamed on the host the server upload them on the GPU memory where they will be analysed. Thanks to the massive parallelization of operation on the GPU device the analysis operations are completed before the next packet of data is uploaded on the GPU. The flexibility of this system relies in the range of operation that can be performed on the data using a Nvidia GPU: thanks to the CUDA language a new signal processing operation can be implemented and tested in a timescale much shorter that the FPGA firmware equivalent. Instructions on how to implement a new algorrithm on the server are contained in the apposite section. Once the data have been analyzed on the GPU they are streamed to a client application via a TCP socket. The client provvided within the system consists of a python library that saves to disk the results and/or plot the data in real time.

Workflow
--------

Once the server is launced, it stastes looking for USRP devices connected to the system. Once at least one device has been found the server starts accepting connections. Two different TCP connection on two different ports (defined at compile time in USRP_server_settings.cpp in variables #TCP_SYNC_PORT and #TCP_ASYNC_PORT) and same socket are accepted and necessary for the system to work. One is a connection used for exchanging messages between server and client, the other is a connection dedicated to data exchange between server and client. Once the connections have been established the server expect a JSON string describing a command: the command describes settings for the USRP hardware and settings for the signal processing/generation needed. Once the JSON command string has been received and validated (settings have to be reasonable) the hardware configuration is applied using the Ettus UHD C++ api to set the USRP FPGA registers and the GPU kernels and meory are initialized accordigly to the requirements in the JSON command. Many threads are spawned in this phase: two different threads for the transmission handling and a minimum of three threads for the receiver side.
The measurement can terminate for different reasons: number of samples in input/output requested has been reached, the client has disconnected or the USRP devices encountered an error.
When the mesuremnt terminates, all the GPU memory is released, all the queues between threads are clean and every thread (except for memoyr managers and main) is joined; the servers goes in IDLE state waiting for a new command.

Datapath
--------

Data are moved around the server using the lock-free queue implementation of boost libraries. Those queues (defined in USRP_server_settings.cpp) are queues of pointers to different kind of struct, these structs serve different purposes but relies on the same structure: a pointer to the memory and some metadata usefull to read the memory.

<p>
 <img src="Memory_model.png" alt="Memory_model.png" style="width:70%">
 <center><em>RX datapath scheme:"Block representation of the datapath in the receiver side.</em></center>
<p>


<p>
 <img src="memory_tx.png" alt="memory_tx.png" style="width:70%">
 <center><em>TX datapath scheme:"Block representation of the datapath in the transmission side</em></center>
<p>


Memory
----------------
All the memory used during a measurement is allocated at the beginning of the same except in case the system strongly (segfault without) needs more memory. The way buffer memory is managed is using a custom memory allocator coded in the file #USRP_server_memory_management.hpp. Note that most of the code has been moved to the .hpp because of the templating approach. The custom memory allocator is just an interface between the CUDA pinning mechanism and a couple of lock-free queues for passing and recycling pointers. They way it works can be summarized in few steps:
* Upon initialization the memory allocator initializes a user-defined number of buffer and pin them to the CUDA driver page file (see CUDA pinned memory for more information)
* When a buffer is needed the #preallocator::get() method will return a buffer.
* Once the buffer has been used, it is returned to the pool using the #preallocator::trash() method.
* A call to the #preallocator::close() method will free the memory pool.
Internally pulling and pushing from the memory pool (which is a lock-free queue) is handled by a couple of low-priority, unbinded threads which ensure the operation is completed without blocking the main application. It may seem redundant to use a lock-free queue and two threads to achieve the immediate return of the memory invocation but lock-free stuff generally come with the price of a failure possibility which in this context will cause memory leaks and loss of stability.
<p>
 <img src="Memory_allocator.png" alt="Memory_allocator.png" style="width:70%">
 <center><em>Block diagram of the memory manager class used in the server.</em></center>
<p>
