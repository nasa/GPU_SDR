#pragma once
#ifndef SYNC_CLASS_INCLUDED
#define SYNC_CLASS_INCLUDED
#include "USRP_server_settings.hpp"
#include "USRP_buffer_generator.hpp"
#include "USRP_server_memory_management.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_hardware_manager.hpp"
#include "USRP_demodulator.hpp"
#include "USRP_buffer_generator.hpp"
#include "USRP_file_writer.hpp"
#include "USRP_server_network.hpp"
#include "kernels.cuh"

class TXRX{

    public:
    
        //pointer to the streaming queue, initialized with the class. output of rx dsp process
        rx_queue* stream_queue;
    
        //pointer to the output memory allocator
        preallocator<float2>* rx_output_memory;
        
        //diagnostic info on output
        bool diagnostic;
        
        bool file_writing;
        
        bool tcp_streaming;
        
        //the initialization method requires an already initialized hardware manager class and an already initialized streaming queue (output of analysis)
        TXRX(server_settings* settings, hardware_manager* init_hardware, bool diagnostic_init = false);

        //launches the setting functions for the required signals, antennas...
        void set(usrp_param* global_param);

        //start the threads
        void start(usrp_param* global_param);
        
        //check if the streamer can take a new command and clean the threads for it.
        //in case the force option is true, force close the threads and cleans the queues
        // NOTE: with force option true this call is blocking
        bool stop(bool force = false);
        
    private:
        
        
        //thread for loading a packet from the buffer generator into the transmit thread
        //assuming a single TX generator and a single TX loader
        void tx_single_link(
            preallocator<float2>* memory, //the custom memory allocator to use in case of dynamically denerated buffer
            TX_buffer_generator* generator, //source of the buffer
            tx_queue* queue_tx, //holds the pointer to the queue
            size_t total_samples, //how many sample to produce and push
            bool dynamic, //true if the preallocation requires dynamic memory
            int preallocated // how many samples have been preallocate
        );
        
        //thread for taking a packet from the receive queue and pushing it into the analysis queue
        void rx_single_link(
            preallocator<float2>* input_memory,
            preallocator<float2>* output_memory,
            RX_buffer_demodulator* demodulator,
            hardware_manager* rx_thread,
            size_t max_samples,
            rx_queue* stream_q    //pointer to the queue to transport the buffer wrapper structure from the analysis to the streaming thread
        );
        
        //pointer to current tx parameters.
        param* current_tx_param;
        param* current_rx_param;
        
        //status of the workers
        std::atomic<bool> RX_status, TX_status;
        
        //pointer to the worker threads
        boost::thread* RX_worker;
        boost::thread* TX_worker;
        
        //pointer to the preallocator to the rx memory. will be initialized during class init
        preallocator<float2>* rx_memory;
        
        //to avoid reallocation of memory in case two measures shares the same buffer length
        int rx_buffer_len;
        
        //pointer to the preallocator to the tx memory. will be initialized during class init
        preallocator<float2>* tx_memory;
        
        //keep track of the preallocated packets in the tx queue.
        int preallocated;
        
        //to avoid reallocation of memory in case two measures shares the same buffer length
        int tx_buffer_len;

        int output_memory_size;
        
        //pointer to the hardware class
        hardware_manager* hardware;
        
        //pointers to demodulator and signal generators classes
        TX_buffer_generator* tx_gen;
        RX_buffer_demodulator* rx_dem;

        //how to know if the measure is in progress
        threading_condition *rx_conditional_waiting;
        threading_condition *tx_conditional_waiting;
        
        //internal streaming and writing classes
        Sync_server* TCP_streamer;
        
        H5_file_writer* H5_writer;
};

#endif
