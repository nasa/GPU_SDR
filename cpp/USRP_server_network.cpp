#include "USRP_server_network.hpp"

std::atomic<bool> reconnect_data; //when th async server detects a disconnection of the API make the sync thread reconnect
std::atomic<bool> reconnect_async; // same thing used whe the exception is caught on the sync thread

Sync_server::Sync_server(rx_queue* init_stream_queue, preallocator<float2>* init_memory,bool init_passthrough){
    passthrough = init_passthrough;
    stream_queue = init_stream_queue;
    if(passthrough){
        out_queue = new rx_queue(SECONDARY_STREAM_QUEUE_LENGTH);
    }
    virtual_pinger_online = false;
    memory = init_memory;
    NET_IS_CONNECTED = false;
    NET_IS_STREAMING = false;
    verbose = true;
    io_service = new boost::asio::io_service;
    option = new boost::asio::socket_base::reuse_address(true);
}

//update pointers in case of memory swapping in TXRX class
void Sync_server::update_pointers(rx_queue* init_stream_queue, preallocator<float2>* init_memory){
    memory = init_memory;
    stream_queue = init_stream_queue;
}

void Sync_server::connect(int init_tcp_port){
    boost::asio::socket_base::reuse_address ciao(true);
    
    if(verbose)std::cout<<"Waiting for TCP data connection on port: "<< init_tcp_port<<" ..."<<std::endl;
    acceptor = new tcp::acceptor(*io_service, tcp::endpoint(tcp::v4(), init_tcp_port),true);
    acceptor->set_option(ciao);
    socket = new tcp::socket(*io_service);
    //socket->set_option(tcp::no_delay(true));
    acceptor->accept(*socket);
    if(verbose)std::cout<<"TCP data connection status update: Connected."<< std::endl;
    NET_IS_CONNECTED = true;
    NEED_RECONNECT = false;
    if(not virtual_pinger_online){
        //std::cout<<"launching virtual pinger..."<<std::endl;
        virtual_pinger_thread = new boost::thread(boost::bind(&Sync_server::virtual_pinger,this));
    }else{
        //std::cout<<"NO virtual pinger..."<<std::endl;
    }
}

void Sync_server::reconnect(int init_tcp_port){
    stop(true);
    delete acceptor;
    delete socket;
    connect(init_tcp_port);
    
}

bool Sync_server::start(param* current_settings){
    if (NEED_RECONNECT){
        print_warning("Before start streaming, data soket has to be reconnected.");
        while(NEED_RECONNECT)std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (NET_IS_CONNECTED){
        TCP_worker = new boost::thread(boost::bind(&Sync_server::tcp_streamer,this, current_settings));
    }else{
        print_error("Cannot stream data without a connected socket!");
        return false;
    }
    return true;
}

//gracefully stop streaming or check streaming status
bool Sync_server::stop(bool force){
    if(NET_IS_CONNECTED and force){
        force_close = true;
        TCP_worker->interrupt();
        TCP_worker->join();
        //print_debug("force_stopping TCP");
        return NET_IS_STREAMING;
    }else if(NET_IS_CONNECTED){
        //print_debug("stopping TCP");
        force_close = false;
        TCP_worker->interrupt();
        if(not NET_IS_STREAMING)TCP_worker->join();
        return NET_IS_STREAMING;
    }
    print_warning("Chekcing streaming status on disconnected socket");
    return false;
}

bool Sync_server::check_status(){
    return NET_IS_STREAMING;
}

//Clean the queue using the associated memory preallocator. NOTE: does not close the preallocator.
int Sync_server::clear_stream_queue( rx_queue *q, preallocator<float2>* memory){
    int i = 0;
    RX_wrapper trash_packet;
    while(!q->empty()){
        i++;
        q->pop(trash_packet);
        memory->trash(trash_packet.buffer);
    }
    return i;
}

//periodically check the status of the async thread to determine if there is needing to reconnect
void Sync_server::virtual_pinger(){
    bool active = true;
    virtual_pinger_online = true;
    while(active){
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        try{
            //std::cout<<"Testing"<<std::endl;
            if(reconnect_data){
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                reconnect_data = false; //twice to avoid data race with async
                NEED_RECONNECT = true;
                NET_IS_CONNECTED = false;
                reconnect_thread = new boost::thread(boost::bind(&Sync_server::reconnect,this,TCP_SYNC_PORT));
                reconnect_data = false;
                active = false;
            }   
        }catch(boost::thread_interrupted &){
            active = false;
        } 
    }
    virtual_pinger_online = false;
}
//size_t ilen = 0;
//This function serialize a net_buffer struct into a boost buffer.
void Sync_server::format_net_buffer(RX_wrapper input_packet, char* __restrict__ output_buffer){

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
    //ilen+=input_packet.length/input_packet.channels;
    //std::cout<<"Streaming packet: "<< input_packet.packet_number <<" acc_samp: "<< ilen<<std::endl;

}


//THIS FUNCTION IS INTENDED TO BE LUNCHED AS A SEPARATE THREAD
void Sync_server::tcp_streamer(param* current_settings){

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
    
    //stop_watch timer;
    
    while(active or finishing){
        if(reconnect_data){
            active = false;
            finishing = false;
        }
        try{
            boost::this_thread::interruption_point();
            if(stream_queue->pop(incoming_packet)){
                //calculate total size to be transmitted
                int total_size = header_size + incoming_packet.length * 2 * sizeof(float);
                
                
                
                //setrialize data structure in a char buffer
                format_net_buffer(incoming_packet, fullData);
                
                if(not NEED_RECONNECT){
                    try{
                        //send data structure         
                        //timer.start();
                        boost::asio::write(*socket, boost::asio::buffer(fullData,total_size),boost::asio::transfer_all(), ignored_error);
                        //timer.cycle();

                        
                    }catch(std::exception &e){
                        std::stringstream ss;
                        ss<<"Sync data thread: "<<e.what();
                        print_error(ss.str());
                        active = false;
                        NET_IS_STREAMING = false;
                        //reconnect_data = true;
                        std::cout<<e.what()<<std::endl;
                    }
                    if (ignored_error!=boost::system::errc::success and ignored_error){
                        std::stringstream ss;
                        ss<<std::string("Sync data thread: ")<<std::string(ignored_error.message());
                        print_error(ss.str());
                        NEED_RECONNECT = true;
                        NET_IS_CONNECTED = false;
                        //reconnect_data = true;
                        active = false;
                        //reconnect_thread = new boost::thread(boost::bind(&Sync_server::reconnect,this,TCP_SYNC_PORT));
                    }
                }
                //print_debug("Packet length is:",incoming_packet.length);
                if(passthrough){
                    while(not out_queue->push(incoming_packet))std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }else{
                    memory->trash(incoming_packet.buffer);
                }
                
                
            }else{
                //else wait for packets
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //print_warning("TCP streamer in hold");
            }
            
            if(not active)finishing = not stream_queue->empty();
            /*
            //recheck: the empty method can lead to false positives.
            if(not finishing){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout<<"DEBUG re-checking"<<std::endl;
                finishing = true; //not stream_queue->empty();
            }
            std::cout<<"DEBUG queue finished: "<<finishing<<std::endl;
            */
        }catch(boost::thread_interrupted &){
            active = false;
            if (not force_close){
                //finishing = not stream_queue->empty();
                
            }else{
                finishing = false;
                //was outside the if else statement
            }
        }
    }
    free(fullData);
    NET_IS_STREAMING = false;
    //std::cout<<"time elapsed in boost::asio::write: "<<timer.get_average()<<std::endl;
}

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



bool Async_server::connected(){
    return ASYNC_SERVER_CONNECTED;
}

Async_server::Async_server(bool init_verbose){
    reconnect_async = false;
    verbose = init_verbose;

    //connect the async server
    ASYNC_SERVER_CONNECTED = false;
    connect(TCP_ASYNC_PORT);
    
    //create the message queue
    command_queue = new async_queue(0);
    response_queue = new async_queue(0);
    
    //start the server
    TCP_async_worker_RX = new boost::thread(boost::bind(&Async_server::rx_async,this,command_queue));
    TCP_async_worker_TX = new boost::thread(boost::bind(&Async_server::tx_async,this,response_queue));
    
    
}

bool Async_server::chk_new_command(){
    print_debug("async interrupt size: ",socket->available());
    return socket->available()>0?true:false;
}

//blocks until it pushes the pointer in the async transmit queue
void Async_server::send_async(std::string* message){
    if(ASYNC_SERVER_CONNECTED){
        while(not response_queue->push(message))std::this_thread::sleep_for(std::chrono::microseconds(25));
    }else{
        print_warning("Cannot sent async message, interface disconnected.");
        delete message;
    }
};

//return true if there is a message and points to it
bool Async_server::recv_async(usrp_param &my_parameter, bool blocking){
    if(not ASYNC_SERVER_CONNECTED){
        print_warning("Async server is not connected, cannot receive messages.");
        return false;
    }
    bool res = false;
    std::string* message_string;
    if (blocking){
        while(not command_queue->pop(message_string)) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        res = string2param(*message_string, my_parameter);
        return res;
        
    }else{
        
        if(command_queue->pop(message_string)){
            //interpreter goes here
            res = string2param(*message_string, my_parameter);
            delete message_string;
        }
        return res;
    }
}



//connect to the async server
void Async_server::connect(int init_tcp_port){
        reconnect_async = false;
        if(verbose)std::cout<<"Waiting for TCP async data connection on port: "<< init_tcp_port<<" ..."<<std::endl;;
        boost::asio::socket_base::reuse_address ciao(true);
        io_service = new boost::asio::io_service();
        acceptor = new tcp::acceptor(*io_service, tcp::endpoint(tcp::v4(), init_tcp_port),true);
        acceptor->set_option(ciao);
        
        socket = new tcp::socket(*io_service);
        
        acceptor->accept(*socket);
        //socket->set_option(ciao);
        if(verbose)std::cout<<"Async TCP connection update: Connected."<< std::endl;
        ASYNC_SERVER_CONNECTED = true;
    }
    
void Async_server::Disconnect(){
    
    delete io_service;
    delete acceptor;
    delete socket;
}    
    
void Async_server::Reconnect(){
    reconnect_data = true;
    reconnect_async = false;
    ASYNC_SERVER_CONNECTED = false;
    connect(TCP_ASYNC_PORT);
    TCP_async_worker_RX = new boost::thread(boost::bind(&Async_server::rx_async,this,command_queue));
    TCP_async_worker_TX = new boost::thread(boost::bind(&Async_server::tx_async,this,response_queue));
}    

//returns the number of bytes in the next async message
int Async_server::check_header(char* init_header){
    int* code_check = reinterpret_cast<int*>(init_header); 
    if(code_check[0] == 0){
        return code_check[1];
    }
    return 0;
}

//format the header to be coherent with msg length and initialization code
//returns the size in byte to send
void Async_server::format_header(char* header, std::string* message){
    int *head = reinterpret_cast<int*>(header);
    head[0]=0;
    head[1]=message->length();
}


void Async_server::rx_async(async_queue* link_command_queue){

    //preallocate space for the fixed header
    char* header_buffer;
    header_buffer = (char*)malloc(2*sizeof(int));
    
    //declare the message buffer
    char* message_buffer;
    std::string* message_string;
    
    bool active = true;
    
    boost::system::error_code error;
    
    while(active){
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        try{
        
            boost::this_thread::interruption_point();

            *header_buffer = {0};
            boost::asio::read(
                *socket,
                boost::asio::buffer(header_buffer,2*sizeof(int)),
                boost::asio::transfer_all(),
                error
            );

            
            int size = error != boost::system::errc::success?0:(reinterpret_cast<int*>(header_buffer))[1];
            if ((reinterpret_cast<int*>(header_buffer))[0]!=0){
                print_warning("Corrupted async header detected!");
            }
            if (error == boost::system::errc::success){
                //std::cout<<"Header size is "<<size<<std::endl;
                
                //allocates the message buffer
                message_buffer = (char*)malloc(size);
                *message_buffer = {0};
                
                //std::cout<<message_buffer<<std::endl;
                boost::asio::read(
                    *socket,
                    boost::asio::buffer(message_buffer,size),
                    boost::asio::transfer_all(),
                    error
                );
                
                if (error == boost::system::errc::success){
                
                    message_string = new std::string(message_buffer,size);
                    //std::cout<<message_buffer<<std::endl;
                    //the buffer has been received, push the message in the queue
                    while(not link_command_queue->push(message_string))std::this_thread::sleep_for(std::chrono::microseconds(25));
                    free(message_buffer);
                    
                    
                }else{
                    std::stringstream ss;
                    ss<<"Async RX server side encountered a payload problem: "<<error.message()<<std::endl;
                    reconnect_data = true;
                    print_warning(ss.str());
                    active = false;
                    ASYNC_SERVER_CONNECTED = false;
                    Disconnect();
                    Reconnect();
                    
                }
            }else{
            
                    std::stringstream ss;
                    ss<<"Async RX server side encountered a header problem: "<<error.message()<<std::endl;
                    reconnect_data = true;
                    print_warning(ss.str());
                    active = false;
                    ASYNC_SERVER_CONNECTED = false;
                    Disconnect();
                    Reconnect();
            
            }
            if(reconnect_async){
                active = false;
                ASYNC_SERVER_CONNECTED = false;
                Disconnect();
                Reconnect();
            
            }
            if(not ASYNC_SERVER_CONNECTED)active = false;
            
        }catch(boost::thread_interrupted &){
            active = false;
        }
    }
    
    //free(header_buffer);
}

void Async_server::tx_async(async_queue* response_queue_link){

    bool active = true;
    
    std::string* message;
    
    //allocate header space
    char* header = (char*)malloc(2*sizeof(int));
    
    //some error handling
    boost::system::error_code ignored_error;
    
    while(active){
    
        try{
        
            boost::this_thread::interruption_point();
            
            if(response_queue_link->pop(message)){
                
                //format the header
                format_header(header, message);
                
                //send the header
                boost::asio::write(*socket, boost::asio::buffer(header,2*sizeof(int)),boost::asio::transfer_all(), ignored_error);
                
                //send message
                boost::asio::write(*socket, boost::asio::buffer(*message),boost::asio::transfer_all(), ignored_error);
                
                //check error
                if (ignored_error!=boost::system::errc::success){
                    std::stringstream ss;
                    ss<<"Async tx error: "<<ignored_error.message()<<std::endl;
                    print_warning(ss.str());
                }
                
                //release the message memory
                delete message;
                
            }else{
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }catch(boost::thread_interrupted &){
            active = false;
        }
        
        if(not ASYNC_SERVER_CONNECTED)active = false;
    }
    
    free(header);
}
