#pragma once
#ifndef USRP_NET_INCLUDED
#define USRP_NET_INCLUDED

#include "USRP_server_diagnostic.hpp"
#include "USRP_server_memory_management.hpp"
#include "USRP_server_settings.hpp"
#include "USRP_JSON_interpreter.hpp"
#include <boost/asio.hpp>
using boost::asio::ip::tcp;
using boost::asio::ip::address;

#define MSG_LENGHT 1e4  //Lenght og the buffer string of the server, should not influent since is in a loop

extern std::atomic<bool> reconnect_data; //when th async server detects a disconnection of the API make the sync thread reconnect
extern std::atomic<bool> reconnect_async; // same thing used whe the exception is caught on the sync thread

class Sync_server{

    public:

        //true if the udp socket is connected
        std::atomic<bool> NET_IS_CONNECTED;

        //true when the server is streaming data
        std::atomic<bool> NET_IS_STREAMING;

        bool NEED_RECONNECT = false;

        bool verbose;
        bool passthrough;
        rx_queue* stream_queue;
        rx_queue* out_queue;
        preallocator<float2>* memory;

        Sync_server(rx_queue* init_stream_queue, preallocator<float2>* init_memory,bool init_passthrough = false);

        //update pointers in case of memory swapping in TXRX class
        void update_pointers(rx_queue* init_stream_queue, preallocator<float2>* init_memory);

        void connect(int init_tcp_port);

        void reconnect(int init_tcp_port);

        bool start(param* current_settings);

        //gracefully stop streaming or check streaming status
        bool stop(bool force = false);

        bool check_status();

        //Clean the queue using the associated memory preallocator. NOTE: does not close the preallocator.
        int clear_stream_queue( rx_queue *q, preallocator<float2>* memory);

    private:

        boost::thread* TCP_worker;
        boost::thread* reconnect_thread;
        boost::thread* virtual_pinger_thread;
        std::atomic<bool> virtual_pinger_online;

        boost::asio::io_service *io_service = nullptr;
        boost::asio::socket_base::reuse_address *option;
        tcp::acceptor *acceptor;
        tcp::socket *socket;

        //periodically check the status of the async thread to determine if there is needing to reconnect
        void virtual_pinger();
        //size_t ilen = 0;
        //This function serialize a net_buffer struct into a boost buffer.
        void format_net_buffer(RX_wrapper input_packet, char* __restrict__ output_buffer);

        //this variable force the joining of the thread
        std::atomic<bool> force_close;

        //THIS FUNCTION IS INTENDED TO BE LUNCHED AS A SEPARATE THREAD
        void tcp_streamer(param* current_settings);

};

//typedef boost::lockfree::queue< char* > async_queue;
typedef boost::lockfree::queue< std::string* > async_queue;

char* format_error();

char* format_status();

//allocates memory for char*, print the message in it and returns the pointer
char* format_parameter(usrp_param *parameters, bool response);

usrp_param json_2_parameters(std::string message);

//this stuff is encoding the server action with a int
enum servr_action { START, STOP, FORCE_STOP, RESET_USRP, STATUS_REQUEST, INFO_REQUEST, NOTHING };

//convert (and define) the action requrest code into a enumerator used by the server to decide what to do
servr_action code_2_server_action(int code);

//manage all the async communication between the server and the API
class Async_server{

    public:

        //determines if the async server is connected or not
        std::atomic<bool> ASYNC_SERVER_CONNECTED;

        bool connected();

        Async_server(bool init_verbose = false);

        bool chk_new_command();

        //blocks until it pushes the pointer in the async transmit queue
        void send_async(std::string* message);

        //return true if there is a message and points to it
        bool recv_async(usrp_param &my_parameter, bool blocking = true);

    private:

        //determine if the threads foreward messages to the std out
        bool verbose;

        //relax the busy loops. Value in ms
        int loop_delay = 50;

        //this thread will fill the command_queue
        boost::thread* TCP_async_worker_RX;
        boost::thread* TCP_async_worker_TX;

        async_queue* command_queue;
        async_queue* response_queue;

        boost::asio::io_service *io_service = nullptr;
        boost::asio::socket_base::reuse_address option;
        tcp::acceptor *acceptor;
        tcp::socket *socket;

        //connect to the async server
        void connect(int init_tcp_port);

        void Disconnect();

        void Reconnect();

        //returns the number of bytes in the next async message
        int check_header(char* init_header);

        //format the header to be coherent with msg length and initialization code
        //returns the size in byte to send
        void format_header(char* header, std::string* message);

        void rx_async(async_queue* link_command_queue);

        void tx_async(async_queue* response_queue_link);
};

#endif
