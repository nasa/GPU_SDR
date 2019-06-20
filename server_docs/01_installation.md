Installation
============

This section covers the necessary steps to compile and run the USRP_server.

Hardware requirements
---------------------

There are hardware requiremets depending on the bandwidth the user intend to use.
  -CPU: The UHD libraries use the CPU to convert the data acquired with the SDR into buffers available to the GPU server. If the user is planning to use an overall bandwidth > 50 Msps consider using at leas a 7th gen i7 processor or equivalent. If the user is planning to use an  bandwidth > 100 Msps
  -GPU: >=gtx1050

Software requirements
---------------------

Software requirements can variate with the OS actually.

Ubuntu 18.04
------------

The only working system testes until yet.
