/*
THIS FILE IS PART OF THE USRP SERVER.
THIS FILE CONTAINS VARIABLES AND FUNCTIONS NEEDED TO SEND THE DATA TO THE PYTHON API
*/

#ifndef USRP_NET_INCLUDED
#define USRP_NET_INCLUDED 1

#include "USRP_server_diagnostic.cpp"
#include "USRP_server_memory_management.cpp"
#include "USRP_server_settings.hpp"

using boost::asio::ip::tcp;
using boost::asio::ip::address;

#define MSG_LENGHT 1e4  //Lenght og the buffer string of the server, should not influent since is in a loop

class Sync_server{

    public:
    
        //true if the udp socket is connected
        std::atomic<bool> NET_IS_CONNECTED;
        
        //true when the server is streaming data
        std::atomic<bool> NET_IS_STREAMING;

        bool verbose;
        bool passthrough;
        rx_queue* stream_queue;
        rx_queue* out_queue;
        preallocator<float2>* memory;
        
        /* CROSS INCLUDE PROBLEM
        //constructor for usage outside TXRX class
        Sync_server(TXRX *thread_manager,bool init_passthrough = false){
            passthrough = init_passthrough;
            stream_queue = thread_manager->stream_queue;
            if(passthrough){
                out_queue = new rx_queue(SECONDARY_STREAM_QUEUE_LENGTH);
                memory = thread_manager->rx_output_memory;
            }
            NET_IS_CONNECTED = false;
            NET_IS_STREAMING = false;
            verbose = true;
            io_service = new boost::asio::io_service;
            option = new boost::asio::socket_base::reuse_address(true);
        }
        */
        //constructor to be used inside a TXRX class
        Sync_server(rx_queue* init_stream_queue, preallocator<float2>* init_memory,bool init_passthrough = false){
            passthrough = init_passthrough;
            stream_queue = init_stream_queue;
            if(passthrough){
                out_queue = new rx_queue(SECONDARY_STREAM_QUEUE_LENGTH);
            }
            memory = init_memory;
            NET_IS_CONNECTED = false;
            NET_IS_STREAMING = false;
            verbose = true;
            io_service = new boost::asio::io_service;
            option = new boost::asio::socket_base::reuse_address(true);
        }
        
        //update pointers in case of memory swapping in TXRX class
        void update_pointers(rx_queue* init_stream_queue, preallocator<float2>* init_memory){
            memory = init_memory;
            stream_queue = init_stream_queue;
        }
        
        void connect(int init_tcp_port){
            if(verbose)std::cout<<"Waiting for TCP data connection on port: "<< init_tcp_port<<" ..."<<std::flush;
            acceptor = new tcp::acceptor(*io_service, tcp::endpoint(tcp::v4(), init_tcp_port));
            acceptor->set_option(*option);
            socket = new tcp::socket(*io_service);
            acceptor->accept(*socket);
            if(verbose)std::cout<<"Connected."<< std::endl;
            NET_IS_CONNECTED = true;
        }
        
        void start(param* current_settings){
            if (NET_IS_CONNECTED){
                TCP_worker = new boost::thread(boost::bind(&Sync_server::tcp_streamer,this, current_settings));
            }
        }
        
        //gracefully stop streaming or check streaming status
        bool stop(bool force = true){
            if(NET_IS_CONNECTED and force){
                force_close = true;
                TCP_worker->interrupt();
                TCP_worker->join();
                return NET_IS_STREAMING;
            }else if(NET_IS_CONNECTED){
                force_close = false;
                TCP_worker->interrupt();
                if(not NET_IS_STREAMING)TCP_worker->join();
                return NET_IS_STREAMING;
            }
            print_warning("Chekcing streaming status on disconnected socket");
            return false;
        }
        
        bool check_status(){
            return NET_IS_STREAMING;
        }
        
        //Clean the queue using the associated memory preallocator. NOTE: does not close the preallocator.
        int clear_stream_queue( rx_queue *q, preallocator<float2>* memory){
            int i = 0;
            RX_wrapper trash_packet;
            while(!q->empty()){
                i++;
                q->pop(trash_packet);
                memory->trash(trash_packet.buffer);
            }
            return i;
        }

    private:
        
        boost::thread* TCP_worker;
        
        boost::asio::io_service *io_service;
        boost::asio::socket_base::reuse_address *option;
        tcp::acceptor *acceptor;
        tcp::socket *socket;
        
        //This function serialize a net_buffer struct into a boost buffer.
        void format_net_buffer(RX_wrapper input_packet, char* __restrict__ output_buffer){

            //where to write in the buffer
            int offset = 0;
            
            memcpy(&output_buffer[offset], &input_packet.usrp_number, sizeof(input_packet.usrp_number));
            offset += sizeof(input_packet.usrp_number);
            
            memcpy(&output_buffer[offset], &input_packet.front_end_code, sizeof(input_packet.front_end_code));
            offset += sizeof(input_packet.front_end_code);
            
            memcpy(&output_buffer[offset], &input_packet.packet_number, sizeof(input_packet.packet_number));
            offset += sizeof(input_packet.packet_number);
            
            memcpy(&output_buffer[offset], &input_packet.length, sizeof(input_packet.length));
            offset += sizeof(input_packet.length);
            
            memcpy(&output_buffer[offset], &input_packet.errors, sizeof(input_packet.errors));
            offset += sizeof(input_packet.errors);
            
            memcpy(&output_buffer[offset], &input_packet.channels, sizeof(input_packet.channels));
            offset += sizeof(input_packet.channels);
            
            memcpy(&output_buffer[offset], input_packet.buffer, input_packet.length * 2 * sizeof(float));

        }
        
        //this variable force the joining of the thread
        std::atomic<bool> force_close;
        
        //THIS FUNCTION IS INTENDED TO BE LUNCHED AS A SEPARATE THREAD
        void tcp_streamer(param* current_settings){
            //Packet to be serialized and sent
            RX_wrapper incoming_packet;
            
            //The python API will first download the header containing the metadata
            //(including the length of the buffer). The header has fixed size.
            int header_size = sizeof(incoming_packet.usrp_number) + sizeof(incoming_packet.front_end_code);
            header_size += sizeof(incoming_packet.packet_number) + sizeof(incoming_packet.length);
            header_size += sizeof(incoming_packet.errors) + sizeof(incoming_packet.channels);


            //maximum transmission buffer size is only reached when no decimation is applied.
            //To avoid error the support memory to the transmission buffer will be oversized.
            int max_size = current_settings->buffer_len * 2 * sizeof(float) + header_size;

            //buffer for serializing the network packet
            char *fullData  = ( char *)std::malloc(max_size);

            //some error handling
            boost::system::error_code ignored_error;
            
            bool active = true;
            bool finishing = true;
            
            NET_IS_STREAMING = true;
            
            while(active and finishing){
                try{
                    boost::this_thread::interruption_point();
                    if(stream_queue->pop(incoming_packet)){
                        
                        //calculate total size to be transmitted
                        int total_size = header_size + incoming_packet.length * 2 * sizeof(float);
                        
                        //setrialize data structure in a char buffer
                        format_net_buffer(incoming_packet, fullData);
                        
                        try{
                            //send data structure         
                            boost::asio::write(*socket, boost::asio::buffer(fullData,total_size),boost::asio::transfer_all(), ignored_error);
                        }catch(std::exception &e){
                            active = false;
                            NET_IS_STREAMING = false;
                            std::cout<<e.what()<<std::endl;
                        }
                        //print_debug("Packet length is:",incoming_packet.length);
                        if(passthrough){
                            while(not out_queue->push(incoming_packet))std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }else{
                            memory->trash(incoming_packet.buffer);
                        }

                    }else{
                        //else wait for packets
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                    
                    if(not active)finishing = not stream_queue->empty();
                    
                }catch(boost::thread_interrupted &){
                    active = false;
                    if (not force_close){
                        finishing = not stream_queue->empty();
                    }else finishing = false;
                }
            }
            free(fullData);
            NET_IS_STREAMING = false;
        }
};

//typedef boost::lockfree::queue< char* > async_queue;
typedef boost::lockfree::queue< std::string* > async_queue;

char* format_error(){
    char* error = NULL;
    return error;
}

char* format_status(){
    char* error = NULL;
    return error;
}

//allocates memory for char*, print the message in it and returns the pointer
char* format_parameter(usrp_param *parameters, bool response){
    char* error = NULL;
    return error;
}

usrp_param json_2_parameters(std::string message){
    usrp_param parameters;
    return parameters;
}

//this stuff is encoding the server action with a int
enum servr_action { START, STOP, FORCE_STOP, RESET_USRP, STATUS_REQUEST, INFO_REQUEST, NOTHING };

//convert (and define) the action requrest code into a enumerator used by the server to decide what to do
servr_action code_2_server_action(int code){
    servr_action what_2_do;
    switch(code){
    
        case 0:
            what_2_do = START;
            break;
            
        case 1:
            what_2_do = STOP;
            break;
            
        case 2:
            what_2_do = FORCE_STOP;
            break;
            
        case 3:
            what_2_do = RESET_USRP;
            break;
            
        case 4:
            what_2_do = STATUS_REQUEST;
            break;
        
        case 5:
            what_2_do = INFO_REQUEST;
            break;
            
        default:
            what_2_do = NOTHING;
            break;
    }
    return what_2_do;
}



//manage all the async communication between the server and the API
class Async_server{
    
    public:
    
        Async_server(){
        
            //connect the async server
            ASYNC_SERVER_CONNECTED = false;
            connect(TCP_ASYNC_PORT);
            
            //create the message queue
            command_queue = new async_queue();
            response_queue = new async_queue();
            
            //start the server
            
        }
        
        //blocks until it pushes the pointer in the async transmit queue
        void send_async(std::string* message){
            if(ASYNC_SERVER_CONNECTED){
                while(not response_queue->push(message))std::this_thread::sleep_for(std::chrono::microseconds(25));
            }
        };
        
        //return true if there is a message and points to it
        bool recv_async(){return false;}
    

    private:
    
        //determines if the async server is connected or not
        std::atomic<bool> ASYNC_SERVER_CONNECTED;
        
        //determine if the threads foreward messages to the std out    
        bool verbose;
        
        //this thread will send the async messages in the response_queue data queue
        boost::thread* TCP_send;
        async_queue* response_queue;
        
        //this thread will fill the command_queue
        boost::thread* TCP_receive;
        async_queue* command_queue;
        
        boost::asio::io_service *io_service;
        boost::asio::socket_base::reuse_address *option;
        tcp::acceptor *acceptor;
        tcp::socket *socket;
    
        //connect to the asunc server
        void connect(int init_tcp_port){
                if(verbose)std::cout<<"Waiting for TCP async data connection on port: "<< init_tcp_port<<" ..."<<std::flush;
                acceptor = new tcp::acceptor(*io_service, tcp::endpoint(tcp::v4(), init_tcp_port));
                acceptor->set_option(*option);
                socket = new tcp::socket(*io_service);
                acceptor->accept(*socket);
                if(verbose)std::cout<<"Connected."<< std::endl;
                ASYNC_SERVER_CONNECTED = true;
            }
            
        //returns the number of bytes in the next async message
        int check_header(char* init_header){
            int* code_check = reinterpret_cast<int*>(init_header); 
            if(code_check[0] == 0){
                return code_check[1];
            }
            return 0;
        }
        
        //format the header to be coherent with msg length and initialization code
        //returns the size in byte to send
        void format_header(char* header, std::string* message){
            int *head = reinterpret_cast<int*>(header);
            head[0]=0;
            head[1]=message->length();
        }
        
        void recv_error_handler(
            const boost::system::error_code& error,
            std::size_t bytes_transferred
        ){
          if (error == boost::asio::error::eof)ASYNC_SERVER_CONNECTED = false;
        }
        
       
        void rx_async(){
        
            //preallocate space for the fixed header
            char* header_buffer;
            header_buffer = (char*)malloc(2*sizeof(int));
            
            //declare the message buffer
            char* message_buffer;
            std::string* message_string;
            
            bool active = true;
            
            boost::system::error_code error;
            
            while(active){
            
                try{
                
                    boost::this_thread::interruption_point();
                    
                    //Use an asynchronous operation so that it can be cancelled on timeout.
                    std::future<std::size_t> read_header = socket->async_receive(
                        boost::asio::buffer(header_buffer,2*sizeof(int)),
                        boost::asio::use_future
                    );

                    // If timeout occurs, then cancel the operation.
                    if (read_header.wait_for(std::chrono::milliseconds(200)) != std::future_status::timeout){
                        
                        //if (error == boost::asio::error::eof){
                        if(false){
                            ASYNC_SERVER_CONNECTED = false;
                        
                        }else{
                        
                            int size = (reinterpret_cast<int*>(header_buffer))[1];
                            
                            //allocates the message buffer
                            message_buffer = (char*)malloc(size);
                            
                            std::future<std::size_t> read_message = socket->async_receive(
                                boost::asio::buffer(message_buffer,size),
                                boost::asio::use_future
                                //&Async_server::recv_error_handler
                            );
                            
                            //if (error == boost::asio::error::eof){
                            if(false){
                                ASYNC_SERVER_CONNECTED = false;
                        
                            }else{
                            
                                if (read_message.wait_for(std::chrono::milliseconds(500)) == std::future_status::timeout){
                                    //if the operation goes in timeout cancel the buffer and print an error
                                    
                                    free(message_buffer);
                                    
                                    print_error("An async message was not fully received due to network timeout.");
                                    
                                }else{
                                    
                                    message_string = new std::string(message_buffer);
                                    
                                    //the buffer has been received, push the message in the queue
                                    while(not command_queue->push(message_string))std::this_thread::sleep_for(std::chrono::microseconds(25));
                                    
                                    free(message_buffer);
                                    
                                }
                            }
                        }
                    }
                    
                    if(not ASYNC_SERVER_CONNECTED)active = false;
                    
                }catch(boost::thread_interrupted &){
                    active = false;
                }
            }
            
            free(header_buffer);
        }
        
        void tx_async(){
        
            bool active = true;
            
            std::string* message;
            
            //allocate header space
            char* header = (char*)malloc(2*sizeof(int));
            
            //some error handling
            boost::system::error_code ignored_error;
            
            while(active){
            
                try{
                
                    boost::this_thread::interruption_point();
                    
                    if(response_queue->pop(message)){
                        
                        //format the header
                        format_header(header, message);
                        
                        //send the header
                        boost::asio::write(*socket, boost::asio::buffer(header,2*sizeof(int)),boost::asio::transfer_all(), ignored_error);
                        
                        //send message
                        boost::asio::write(*socket, boost::asio::buffer(*message),boost::asio::transfer_all(), ignored_error);
                        
                        //release the message memory
                        delete message;
                        
                    }else{
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }catch(boost::thread_interrupted &){
                    active = false;
                }
            }
            
            free(header);
        }
        
};









#endif 
