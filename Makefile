CC=g++

CC_DEF_FLAGS =-std=c++11 -O3
CC+=$(CC_DEF_FLAGS)

HDF5_PATH_INC = /usr/include/hdf5/serial

CINCLUDE =-I$(HDF5_PATH_INC)
CLINK = -lz -lsz -ldl 
CLINK += -lpthread -lboost_system -lboost_program_options -lboost_chrono -lboost_thread 

CLINK += -luhd 
CLINK += -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_serial 

NVCC = nvcc

NV_DEF_FLAGS = -std=c++11 -arch=sm_61
NVCC+=$(NV_DEF_FLAGS)

NVINCLUDE = -I/usr/local/cuda/include/ 
NVLINK = -lcudart -lcuda  -lcufft -lcublas -lculibos

OBJ =  USRP_server_settings.o  USRP_server_diagnostic.o USRP_server_memory_management.o USRP_server_console_print.o kernels.o USRP_server_link_threads.o USRP_hardware_manager.o USRP_demodulator.o USRP_server_network.o  USRP_JSON_interpreter.o USRP_buffer_generator.o  USRP_file_writer.o  

SRC = usrp_server.cpp USRP_server_console_print.cpp USRP_file_writer.cpp USRP_server_link_threads.cpp USRP_hardware_manager.cpp USRP_demodulator.cpp USRP_server_network.cpp USRP_server_memory_management.cpp USRP_server_diagnostic.cpp USRP_JSON_interpreter.cpp USRP_buffer_generator.cpp USRP_server_settings.cpp

HEAD = USRP_server_link_threads.hpp USRP_hardware_manager.hpp USRP_file_writer.hpp USRP_server_console_print.hpp USRP_server_network.hpp USRP_buffer_generator.hpp USRP_demodulator.hpp USRP_server_diagnostic.hpp USRP_server_settings.hpp USRP_server_memory_management.hpp USRP_JSON_interpreter.hpp

usrp_server:  $(OBJ) usrp_server.o
	$(info Linking all using nvcc...)
	@$(NVCC) $(CFLAGS) -o usrp_server usrp_server.o  $(OBJ) $(CLINK) $(NVLINK)

usrp_server.o: $(HEAD) usrp_server.cpp
	$(info Compiling usrp_server.cpp ...)
	@$(CC) $(CINCLUDE) $(NVINCLUDE)  -c usrp_server.cpp -o usrp_server.o

kernels.o: kernels.cuh kernels.cu
	$(info Compiling kernels.cu ...)
	@$(NVCC) $(NVINCLUDE) -dc -o kernels.o kernels.cu $(NVLINK)
    
USRP_server_console_print.o: $(HEAD) USRP_server_console_print.cpp
	$(info Compiling USRP_server_console_print.cpp ...)
	@$(CC)  -c USRP_server_console_print.cpp -o USRP_server_console_print.o 

USRP_buffer_generator.o: $(HEAD) USRP_buffer_generator.cpp
	$(info Compiling USRP_buffer_generator.cpp ...)
	@$(CC) $(CINCLUDE) $(NVINCLUDE) -c USRP_buffer_generator.cpp -o USRP_buffer_generator.o

USRP_demodulator.o: $(HEAD) USRP_demodulator.cpp
	$(info Compiling USRP_demodulator.cpp ...)
	@$(CC)  $(CINCLUDE) $(NVINCLUDE)  -c USRP_demodulator.cpp -o USRP_demodulator.o

USRP_file_writer.o: $(HEAD) USRP_file_writer.cpp
	$(info Compiling USRP_file_writer.cpp ...)
	@$(CC) -std=c++11  $(CINCLUDE) $(NVINCLUDE)  -c USRP_file_writer.cpp -o USRP_file_writer.o

USRP_server_diagnostic.o: $(HEAD) USRP_server_diagnostic.cpp
	$(info Compiling USRP_server_diagnostic.cpp ...)
	@$(CC) $(NVINCLUDE) $(CINCLUDE)  -c USRP_server_diagnostic.cpp -o USRP_server_diagnostic.o

USRP_JSON_interpreter.o: $(HEAD) USRP_JSON_interpreter.cpp
	$(info Compiling USRP_JSON_interpreter.cpp ...)
	@$(CC) $(NVINCLUDE) $(CINCLUDE)  -c USRP_JSON_interpreter.cpp -o USRP_JSON_interpreter.o
	
USRP_server_memory_management.o: $(HEAD) USRP_server_memory_management.cpp
	$(info Compiling USRP_server_memory_management.cpp ...)
	@$(CC)  $(NVINCLUDE) $(CINCLUDE)  -c USRP_server_memory_management.cpp -o USRP_server_memory_management.o

USRP_hardware_manager.o: $(HEAD) USRP_hardware_manager.cpp
	$(info Compiling USRP_hardware_manager.cpp ...)
	@$(CC)   $(NVINCLUDE) $(CINCLUDE)  -c USRP_hardware_manager.cpp -o USRP_hardware_manager.o

USRP_server_network.o: $(HEAD) USRP_server_network.cpp
	$(info Compiling USRP_server_network.cpp ...)
	@$(CC)   $(NVINCLUDE) $(CINCLUDE) -c USRP_server_network.cpp -o USRP_server_network.o

USRP_server_link_threads.o: $(HEAD) USRP_server_link_threads.cpp
	$(info Compiling USRP_server_link_threads.cpp ...)
	@$(CC) $(NVINCLUDE) $(CINCLUDE) -c USRP_server_link_threads.cpp -o USRP_server_link_threads.o
	
USRP_server_settings.o: $(HEAD) USRP_server_settings.cpp
	$(info Compiling USRP_server_settings.cpp ...)
	@$(CC) $(NVINCLUDE) $(CINCLUDE)  -c USRP_server_settings.cpp -o USRP_server_settings.o

clean:
	rm -f usrp_server
	rm -f *.o
