CC=g++

CC_DEF_FLAGS =-std=c++11 -O3
CC+=$(CC_DEF_FLAGS)

CINCLUDE =-I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
CLINK = -lz -lsz -lboost_system -lboost_program_options -lpthread -lboost_chrono -lboost_thread -luhd -lhdf5_hl_cpp -lhdf5_cpp -lcuda -lcudart


NVCC = nvcc

NV_DEF_FLAGS = -std=c++11 -arch=sm_61
NVCC+=$(NV_DEF_FLAGS)

NVINCLUDE = -I/usr/local/cuda/include/ 
NVLINK = -lcufft_static -lculibos -lcuda -lcublas -lcudart

OBJ = usrp_server.o USRP_file_writer.o USRP_server_link_threads.o USRP_hardware_manager.o USRP_demodulator.o USRP_server_network.o USRP_server_memory_management.o USRP_server_diagnostic.o USRP_JSON_interpreter.o USRP_buffer_generator.o USRP_server_console_print.o

SRC = usrp_server.cpp USRP_server_console_print.cpp USRP_file_writer.cpp USRP_server_link_threads.cpp USRP_hardware_manager.cpp USRP_demodulator.cpp USRP_server_network.cpp USRP_server_memory_management.cpp USRP_server_diagnostic.cpp USRP_JSON_interpreter.cpp USRP_buffer_generator.cpp

usrp_server: kernels.o
	$(CC) $(CLINK) $(CFLAGS) $(CINCLUDE) $(NVINCLUDE) -L/usr/local/cuda/lib64 -o usrp_server $(SRC) kernels.o

usrp_server.o:
	$(CC) $(CLINK) $(CINCLUDE) $(NVINCLUDE) -c usrp_server.cpp -o usrp_server.o
	
kernels.o:
	$(NVCC) $(NVLINK) $(NVINCLUDE) -dc -o kernels.o kernels.cu

USRP_server_console_print.o:
	$(CC) -c USRP_server_console_print.cpp -o USRP_server_console_print.o 

USRP_buffer_generator.o:
	$(CC) $(CLINK) $(NVINCLUDE)  -c USRP_buffer_generator.cpp -o USRP_buffer_generator.o

USRP_demodulator.o:
	$(CC) $(CLINK) $(NVINCLUDE)  -c USRP_demodulator.cpp -o USRP_demodulator.o

USRP_file_writer.o:
	$(CC) $(CLINK) $(CINCLUDE) $(NVINCLUDE)  -c USRP_file_writer.cpp -o USRP_file_writer.o

USRP_server_diagnostic.o:
	$(CC) $(CLINK) $(NVINCLUDE) $(CINCLUDE)  -c USRP_server_diagnostic.cpp -o USRP_server_diagnostic.o

USRP_JSON_interpreter.o:
	$(CC) $(CLINK) $(NVINCLUDE) $(CINCLUDE)  -c USRP_JSON_interpreter.cpp -o USRP_JSON_interpreter.o
	
USRP_server_memory_management.o:
	$(CC)  $(CLINK) $(NVINCLUDE) $(CINCLUDE) -c USRP_server_memory_management.cpp -o USRP_server_memory_management.o

USRP_hardware_manager.o:
	$(CC)  $(CLINK) $(NVINCLUDE) $(CINCLUDE) -c USRP_hardware_manager.cpp -o USRP_hardware_manager.o

USRP_server_network.o:
	$(CC)  $(CLINK) $(NVINCLUDE) $(CINCLUDE) -c USRP_server_network.cpp -o USRP_server_network.o

USRP_server_link_threads.o:
	$(CC) $(CLINK) $(NVINCLUDE) $(CINCLUDE)  -c USRP_server_link_threads.cpp -o USRP_server_link_threads.o

clean:
	rm -f usrp_server
	rm -f *.o
