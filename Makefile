CC=gcc
NVCC = nvcc
NVLINKFLAGS =  -arch=sm_61 -std=c++11 -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include  -I/usr/local/cuda-9.1/include -Xptxas -O3
NVCUFFTLINK = -lcufft_static -lculibos -lboost_program_options -lpthread -lboost_system -luhd -lz -lsz -lcuda -L/usr/local/cuda-9.1/lib64 -lcublas  -luhd -lboost_system  -lboost_chrono  -lboost_thread -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a 

usrp_server: usrp_server.cu
	$(NVCC) $(NVLINKFLAGS) usrp_server.cu -o usrp_server $(CFLAGS) $(NVCUFFTLINK)
	
clean:
	rm -f usrp_server
	
	
#requires hdf5-dev
