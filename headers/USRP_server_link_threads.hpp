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

        //! @brief Pointer to the TCP streaming queue, initialized with the class. Output of every frontend rx dsp processes.
        rx_queue* stream_queue;

        //! @brief Pointer to the output memory allocator of all frontends
        preallocator<float2>* rx_output_memory;

        //! @brief Enables diagnostic info on output.
        bool diagnostic;

        //! @brief Enables server local file writing.
        bool file_writing;

        //! @brief Enables server tcp streaming.
        bool tcp_streaming;

        //! @brief Initialization method requires an already initialized hardware manager class and an already initialized streaming queue (output of analysis)
        TXRX(server_settings* settings, hardware_manager* init_hardware, bool diagnostic_init = false);

        //! @brief Launches the setting functions for the required signals, antennas...
        void set(usrp_param* global_param);

        //! @brief Start the threads.
        void start(usrp_param* global_param);

        //! @ brief Check if the streamer can take a new command and clean the threads for it.
        //! in case the force option is true, force close the threads and cleans the queues
        //! NOTE: with force option true this call is blocking as it calls the join() method of boost's thread.
        bool stop(bool force = false);

    private:


        //! @brief Control number used to set the affinity or rx threads on the same core.
        std::vector<int> rx_thread_n;

        //! @brief Control number used to set the affinity or tx threads on the same core.
        std::vector<int> tx_thread_n;

        //! @brief Control number used to avoid tx and rx threads to have same core affinity.
        int thread_counter;

        //! @brief This function is ment to be run as thread for generating a packet with the buffer generator and load it into the transmit thread.
        //! Assuming a single TX generator and a single TX loader
        void tx_single_link(
            preallocator<float2>* memory, //the custom memory allocator to use in case of dynamically denerated buffer
            TX_buffer_generator* generator, //source of the buffer
            tx_queue* queue_tx, //holds the pointer to the queue
            size_t total_samples, //how many sample to produce and push
            bool dynamic, //true if the preallocation requires dynamic memory
            int preallocated, // how many samples have been preallocate
            char front_end
        );

        //thread for taking a packet from the receive queue and pushing it into the analysis queue
        void rx_single_link(
            preallocator<float2>* input_memory,
            preallocator<float2>* output_memory,
            RX_buffer_demodulator* demodulator,
            hardware_manager* rx_thread,
            size_t max_samples,
            rx_queue* stream_q,    //pointer to the queue to transport the buffer wrapper structure from the analysis to the streaming thread
            char front_end
        );

        //pointer to current tx parameters.
        param* A_current_tx_param;
        param* A_current_rx_param;
        param* B_current_tx_param;
        param* B_current_rx_param;

        //status of the workers
        std::atomic<bool> RX_status, TX_status;

        //pointer to the worker threads
        boost::thread* A_RX_worker;
        boost::thread* A_TX_worker;
        boost::thread* B_RX_worker;
        boost::thread* B_TX_worker;

        //! @brief Pointer to the preallocator to the rx memory of frontend A. will be initialized during class init.
        preallocator<float2>* A_rx_memory;

        //! @brief Pointer to the preallocator to the rx memory of frontend B. will be initialized during class init.
        preallocator<float2>* B_rx_memory;

        //! @brief Bookeeping to avoid reallocation of memory in case two measures shares the same buffer length
        size_t A_rx_buffer_len;

        //! @brief Bookeeping to avoid reallocation of memory in case two measures shares the same buffer length
        size_t B_rx_buffer_len;

        //! @brief Pointer to the preallocator to the tx memory (frontend A).
        preallocator<float2>* A_tx_memory;

        //! @brief Pointer to the preallocator to the tx memory (frontend B).
        preallocator<float2>* B_tx_memory;

        //! @brief Keep track of the preallocated packets in the tx queue (frontend A).
        size_t A_preallocated;

        //! @brief Keep track of the preallocated packets in the tx queue (frontend ).
        size_t B_preallocated;

        //! @brief  Bookeeping to avoid reallocation of memory in case two measures shares the same buffer length (frontend A).
        size_t A_tx_buffer_len;

        //! @brief  Bookeeping to avoid reallocation of memory in case two measures shares the same buffer length (frontend B).
        size_t B_tx_buffer_len;

        size_t output_memory_size;

        //pointer to the hardware class
        hardware_manager* hardware;

        //pointers to demodulator and signal generators classes
        TX_buffer_generator* A_tx_gen;
        RX_buffer_demodulator* A_rx_dem;
        TX_buffer_generator* B_tx_gen;
        RX_buffer_demodulator* B_rx_dem;

        //how to know if the measure is in progress
        threading_condition *rx_conditional_waiting;
        threading_condition *tx_conditional_waiting;

        //internal streaming and writing classes
        Sync_server* TCP_streamer;

        H5_file_writer* H5_writer;

        //temporary storage of memory size multiplier for analysis
        size_t mem_mult_tmp;
};

#endif
