#pragma once
#ifndef USRP_HW_MANAGER
#define USRP_HW_MANAGER

#include "USRP_server_diagnostic.hpp"
#include "USRP_server_settings.hpp"
#include "USRP_server_memory_management.hpp"
#include <uhd/types/time_spec.hpp>

class hardware_manager{
    public:
        
        //internally stored usrp number
        int this_usrp_number;
        
        //determine if the hardware has to be replaced by a software loop
        bool sw_loop;
        
        //address of the device controlled by this instance
        uhd::usrp::multi_usrp::sptr main_usrp;
        
        //the initializer of the class can be used to select which usrp is controlled by the class
        //Default call suppose only one USRP is connected
        hardware_manager(server_settings* settings, bool sw_loop_init, int usrp_number = 0);
        
        //this function should be used to set the USRP device with user parameters
        //TODO catch exceptions and return a boolean
        bool preset_usrp(usrp_param* requested_config);

        //apply the global parameter configuration
        // NOTE: only changes parameters when needed
        
        //those queues can be accessed to retrive or stream data from each frontend.
        rx_queue* RX_queue;
        tx_queue* TX_queue;
        
        bool check_rx_status(bool verbose = false);
        
        bool check_tx_status(bool verbose = false);
        
        void start_tx(
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory = NULL    //if the thread is transmitting a buffer that requires dynamical allocation than a pointer to  custo memory manager class has to be passed
        
        );
        
        void start_rx(
            int buffer_len,                         //length of the buffer. MUST be the same of the preallocator initialization
            long int num_samples,                   //how many sample to receive
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory            //custom memory preallocator
        
        );
        
        void close_tx();
        
        void close_rx();
        
        int clean_tx_queue(preallocator<float2>* memory);
        
        int clean_rx_queue(preallocator<float2>* memory);
        
    private:

        void apply(usrp_param* requested_config);
        
        bool check_tuning();
        
        //the next variables will be called by the tx and rx functions to stream and receive packets.
        //they are managed inside this class as I suspect the frequent reinitialization of the streaming causes
        //the DAC sync error.
       
        uhd::rx_streamer::sptr rx_stream;
        uhd::tx_streamer::sptr tx_stream;

        void set_streams();

        //pointer to rx thread and boolean chk variable
        std::atomic<bool> rx_thread_operation;
        boost::thread* rx_thread;
        
        //pointer to tx thread and boolean chk variable
        std::atomic<bool> tx_thread_operation;
        boost::thread* tx_thread;
        
        //queue for sharing the error event code with RX thread
        error_queue* tx_error_queue;
        
        //kind of device to look for
        uhd::device_addr_t hint;
        
        //array of usrp addresses
        uhd::device_addrs_t dev_addrs;
        
        //last configuration of the usrp device
        usrp_param config;
        
        //pointer to the software loop queue
        tx_queue* sw_loop_queue;
        
        //channel vector. is 1 unit long because each channel has its own streamer
        //used in set_streams() method.
        std::vector<size_t> channel_num;
        
        
        //port-already-connected map: in initialization no port is connected and no thrad is started
        ant_mode A_TXRX_chk;
        ant_mode B_RX2_chk;
        ant_mode B_TXRX_chk;
        ant_mode A_RX2_chk;
        
        //clear the stream before setting a new configuration
        //TODO this can potentially cause a memory leak every time the channel configuration is changed.
        void clear_streams();
        

        char front_end_code0;
        char front_end_code1;
        
        //apply the single antenna configuration
        // NOTE: only changes parameters when needed
        //returns eventual diagnostic messages
        std::string apply_antenna_config(param *parameters, param *old_parameters, size_t chan);
        
        //check if there are more than 1 tx/rx channel
        bool check_double_txrx(ant_mode TXRX);
       
        
        //check if the selected mode has to be tuned
        bool check_global_mode_presence(ant_mode mode, size_t chan);
        
        void software_tx_thread(
            param *current_settings,                //some parameters are useful also in sw
            preallocator<float2>* memory            //custom memory preallocator
            );
        
        void single_tx_thread(
            param *current_settings,                //(managed internally to the class) user parameter to use for rx setting
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory            //custom memory preallocator
        );
        
        //ment to be in a thread. receive messages asyncronously on metadata
        void async_stream();
        
        void software_rx_thread(
            param *current_settings,
            preallocator<float2>* memory,
            rx_queue* Rx_queue
        );
            
        
        void single_rx_thread(
            param *current_settings,                //(managed internally) user parameter to use for rx setting

            rx_queue* Rx_queue,                     //(managed internally)queue to use for pushing
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory            //custom memory preallocator
            
        );
        
        //used to sync TX and rRX streaming time
        void sync_time();
};

#endif
