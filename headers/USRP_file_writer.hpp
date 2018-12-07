#pragma once
#ifndef FILE_WRITER_INCLUDED
#define FILE_WRITER_INCLUDED
#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_server_memory_management.hpp"
#include "USRP_server_network.hpp"
#include <ctime>
#include "H5Cpp.h"
#include "H5CompType.h"

using namespace H5;
class H5_file_writer{
    public:
    
        rx_queue* stream_queue;
        
        //the initialization needs a rx queue to get packets and a memory to dispose of them
        //NOTE: the file writer should always be the last element of the chain unless using a very fast storage support
        H5_file_writer(rx_queue* init_queue, preallocator<float2>* init_memory);

        H5_file_writer(Sync_server *streaming_server);
        
        void start(usrp_param* global_params);
        
        bool stop(bool force = false);
        
        void close();
        
        //in case of pointers update in the TXRX class method set() those functions have to be called
        //this is because memory size adjustments on rx_output preallocator
        void update_pointers(rx_queue* init_queue, preallocator<float2>* init_memory);
        
        void update_pointers(Sync_server *streaming_server);
    
    private:
        
        //datatype used by HDF5 api
        CompType *complex_data_type;//(sizeof(float2));
        
        //pointer to the h5 file
        H5File *file;
        
        //pointer to the raw_data group (single USRP writer)
        Group *group;
        
        //pointers to possible groups for representing USRP X300 data
        Group *A_TXRX;
        Group *B_TXRX;
        Group *A_RX2;
        Group *B_RX2;
        
        //dataspace for H5 file writing
        DataSpace *dataspace;
        
        //rank of the raw_data's datasets and dimensions
        int dspace_rank;
        hsize_t *dimsf; 
        
        //pointer to the memory recycler
        preallocator<float2>* memory;
        
        //pointer to the thread
        boost::thread* binary_writer;
        
        std::atomic<bool> writing_in_progress;
        
        void write_properties(Group *measure_group, param* parameters_group);

        std::string get_name();
        
        void clean_queue();
        
        //force the joining of the thread
        std::atomic<bool> force_close;
        
        void write_files();
};
#endif
