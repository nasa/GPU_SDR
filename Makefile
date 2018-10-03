CC=gcc
NVCC = nvcc
NVLINKFLAGS =  -arch=sm_61 -std=c++11 -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu -lpthread -L/usr/local/cuda-9.1/lib64 -lcuda -lcublas  -luhd -lboost_system  -lboost_chrono  -lboost_thread  -I/usr/local/cuda-9.1/include -lboost_program_options  -lboost_system -luhd -lz -lsz -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a 
NVCUFFTLINK = -lcufft_static -lculibos -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a 

#loopback: loopback_test.cu
#	$(NVCC) $(NVLINKFLAGS) loopback_test.cu -o loopback $(CFLAGS) $(NVCUFFTLINK)

usrp_server: usrp_server.cu
	$(NVCC) $(NVLINKFLAGS) usrp_server.cu -o usrp_server $(CFLAGS) $(NVCUFFTLINK)
	
clean:
	rm -f usrp_server loopback
