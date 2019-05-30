#pragma once
#ifndef USRP_HW_MANAGER
#define USRP_HW_MANAGER

#include "USRP_server_diagnostic.hpp"
#include "USRP_server_settings.hpp"
#include "USRP_server_memory_management.hpp"
#include <uhd/types/time_spec.hpp>


//! @brief Manages the hardware I/O of one usrp unit.
class hardware_manager{
    public:

        //internally stored usrp number
        size_t this_usrp_number;

        //determine if the hardware has to be replaced by a software loop
        bool sw_loop;

        //address of the device controlled by this instance
        uhd::usrp::multi_usrp::sptr main_usrp;

        //! @brief The initializer of the class can be used to select which usrp is controlled by the class
        //! Default call suppose only one USRP is connected
        hardware_manager(server_settings* settings, bool sw_loop_init, size_t usrp_number = 0);

        //! @brief Set the USRP device with user parameters
        //! @todo TODO catch exceptions and return a boolean
        bool preset_usrp(usrp_param* requested_config);

        //! @brief Queue accessed to retrive data from A frontend.
        rx_queue* A_RX_queue;

        //! @brief Queue accessed to stream data from A frontend.
        tx_queue* A_TX_queue;

        //! @brief Queue accessed to retrive data from B frontend.
        rx_queue* B_RX_queue;

        //! @brief Queue accessed to stream data from B frontend.
        tx_queue* B_TX_queue;

        //! @brief Check the status of every rx operations.
        //! Returns the status of A or B.
        bool check_rx_status(bool verbose = false);

        //! @brief Check the status of A rx operations.
        bool check_A_rx_status(bool verbose = false);

        //! @brief Check the status of B rx operations.
        bool check_B_rx_status(bool verbose = false);

        //! @brief Check the status of every tx operations.
        //! Returns the status of A or B.
        bool check_tx_status(bool verbose = false);

        //! @brief Check the status of A tx operations.
        bool check_A_tx_status(bool verbose = false);

        //! @brief Check the status of B tx operations.
        bool check_B_tx_status(bool verbose = false);

        //! @brief Start a transmission thread.
        //! The threads started by this function do two things: pop a packet from the respective queue; stram the packet via UHD interface.
        //! Each streamer is handled by an independent thread.
        //! If the source queue is empty a warning is printed on the console and an error is pushed in the erorr queue.
        void start_tx(
            threading_condition* wait_condition,    //before joining wait for that condition
            int thread_op,                          //core affinity of the process
            param *current_settings,                //representative of the paramenters (must match A or B frontend description)
            char front_end,                          //must be "A" or "B"
            preallocator<float2>* memory = NULL    //if the thread is transmitting a buffer that requires dynamical allocation than a pointer to  custo memory manager class has to be passed.

        );

        //! @ brief start a rx thread.
        void start_rx(
            int buffer_len,                         //length of the buffer. MUST be the same of the preallocator initialization
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory,           //custom memory preallocator
            int thread_op,                          //core affinity number
            param *current_settings,                //representative of the paramenters (must match A or B frontend description)
            char front_end                          //must be "A" or "B"

        );

        //! @brief Close all the tx streamer threads.
        void close_tx();

        //! @brief Close all the rx streamer threads.
        void close_rx();

        //! @brief Release the memory associated with pointers holded by a tx queue using the respective memory allocator.
        int clean_tx_queue(tx_queue* TX_queue,preallocator<float2>* memory);

        //! @brief Release the memory associated with pointers holded by a rx queue using the respective memory allocator.
        int clean_rx_queue(rx_queue* RX_queue, preallocator<float2>* memory);

        std::atomic<bool> B_rx_thread_operation;
        std::atomic<bool> A_rx_thread_operation;
    private:

        //! @brief Describe the state of the TX settling time for the A front_end.
        bool tx_loaded_cmd_A;

        //! @brief Describe the state of the TX settling time for the B front_end.
        bool tx_loaded_cmd_B;

        //! @brief Describe the state of the RX settling time for the A front_end.
        bool rx_loaded_cmd_A;

        //! @brief Describe the state of the RX settling time for the B front_end.
        bool rx_loaded_cmd_B;

        void apply(usrp_param* requested_config);

        bool check_tuning();

        //the next variables will be called by the tx and rx functions to stream and receive packets.
        //they are managed inside this class as I suspect the frequent reinitialization of the streaming causes
        //the DAC sync error.
        uhd::rx_streamer::sptr A_rx_stream;
        uhd::tx_streamer::sptr A_tx_stream;
        uhd::rx_streamer::sptr B_rx_stream;
        uhd::tx_streamer::sptr B_tx_stream;

        void set_streams();

        void flush_rx_streamer(uhd::rx_streamer::sptr &rx_streamer);


        boost::thread* A_rx_thread;

        boost::thread* B_rx_thread;

        //pointer to tx thread and boolean chk variable
        std::atomic<bool> A_tx_thread_operation;
        boost::thread* A_tx_thread;
        std::atomic<bool> B_tx_thread_operation;
        boost::thread* B_tx_thread;

        //queue for sharing the error event code with RX thread
        error_queue* A_tx_error_queue;
        error_queue* B_tx_error_queue;

        //kind of device to look for
        uhd::device_addr_t hint;

        //array of usrp addresses
        uhd::device_addrs_t dev_addrs;

        //last configuration of the usrp device
        usrp_param config;

        //pointer to the software loop queue
        tx_queue* A_sw_loop_queue;
        tx_queue* B_sw_loop_queue;

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
            preallocator<float2>* memory,            //custom memory preallocator
            tx_queue* TX_queue,
            tx_queue* sw_loop_queue,
            char front_end
        );

        void single_tx_thread(
            param *current_settings,                //(managed internally to the class) user parameter to use for rx setting
            threading_condition* wait_condition,    //before joining wait for that condition
            tx_queue* TX_queue,
            uhd::tx_streamer::sptr &tx_stream,       //asscociated usrp stream
            preallocator<float2>* memory,           //custom memory preallocator
            char front_end
        );

        //ment to be in a thread. receive messages asyncronously on metadata
        void async_stream(uhd::tx_streamer::sptr &tx_stream, char fron_tend);

        void software_rx_thread(
            param *current_settings,
            preallocator<float2>* memory,
            rx_queue* Rx_queue,
            tx_queue* sw_loop_queue,
            char front_end
        );


        void single_rx_thread(
            param *current_settings,                //(managed internally) user parameter to use for rx setting

            rx_queue* Rx_queue,                     //(managed internally)queue to use for pushing
            threading_condition* wait_condition,    //before joining wait for that condition
            preallocator<float2>* memory,           //custom memory preallocator
            uhd::rx_streamer::sptr &rx_stream,      //associated usrp streamer
            char front_end

        );
};

#endif
