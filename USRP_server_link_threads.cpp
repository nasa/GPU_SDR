

/**********************************************************************
THIS FILE CONTAINS THE CLASSES TO WRAP THE PARAM STRUCT - UHD INTERFACE
THE METHODS IN THIS FILE WILL BE CALLED IN THE INTERPRET FUNCTIONS
***********************************************************************/
#ifndef SYNC_CLASS_INCLUDED
#define SYNC_CLASS_INCLUDED 1
#include "USRP_server_settings.hpp"
#include "USRP_buffer_generator.cpp"
#include "USRP_server_memory_management.cpp"
#include "USRP_server_diagnostic.cpp"
#include "USRP_hardware_manager.cpp"
#include "USRP_demodulator.cpp"
#include "USRP_buffer_generator.cpp"
#include "USRP_file_writer.cpp"
#include "USRP_server_network.cpp"
#include "kernels.cu"
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
        TXRX(server_settings* settings, hardware_manager* init_hardware, bool diagnostic_init = false){
            
            //set the streamin/writing options
            tcp_streaming = settings->TCP_streaming;
            file_writing = settings->FILE_writing;

            //assign the streaming queue
            stream_queue = new rx_queue(STREAM_QUEUE_LENGTH);
            
            //set diagnostic info
            diagnostic = diagnostic_init;
            
            //import the hardware pointer
            hardware = init_hardware;
            
            //memory initialization (can be changed on the fly)
            rx_buffer_len = settings->default_rx_buffer_len;
            tx_buffer_len = settings->default_tx_buffer_len;
            
            //initialize all the possible RX memory
            std::cout<<"initializing "<<(rx_buffer_len *  RX_QUEUE_LENGTH * sizeof(float2))/(1024.*1024.)<<"MB of memory... "<<std::endl;
            rx_memory = new preallocator<float2>(rx_buffer_len,RX_QUEUE_LENGTH);
            
            //TX memory do not have autorefill
            tx_memory = new preallocator<float2>(tx_buffer_len,TX_QUEUE_LENGTH,false);

            //build the streaming/ writing chain
            if(tcp_streaming and file_writing){
                TCP_streamer = new Sync_server(stream_queue, rx_output_memory, true);
                H5_writer = new H5_file_writer(TCP_streamer);
            }
            
            if(tcp_streaming and (not file_writing))TCP_streamer = new Sync_server(stream_queue, rx_output_memory, false);
            
            if((not tcp_streaming) and file_writing)H5_writer = new H5_file_writer(stream_queue, rx_output_memory);

            if(tcp_streaming)TCP_streamer->connect(TCP_SYNC_PORT);
            
            //allocation depends on parameters
            output_memory_size = 0;
            
            std::cout<<"Memory initialization done."<<std::endl;
            
            //initialize the conditional waiting classes
            rx_conditional_waiting = new threading_condition();
            tx_conditional_waiting = new threading_condition();
            
            //set the initial thread status
            RX_status = false;
            TX_status = false;
            
            //set the pointer to current parameter configuration
            current_tx_param = NULL;
            current_rx_param = NULL;
            
        }

        //launches the setting functions for the required signals, antennas...
        void set(usrp_param* global_param){
        
            
            
            std::vector<param*> modes(4);
            modes[0] = &global_param->A_TXRX;
            modes[1] = &global_param->A_RX2;
            modes[2] = &global_param->B_TXRX;
            modes[3] = &global_param->B_RX2;
            
            //the writer class takes care of single and double RX stream so it only has to be started once.(moved to start method)
            //bool writer_started = false;
            
            //the streamer class is currently setted to only receive one stream (moved to start method)
            //bool streamer_started = false;
            
            //single TX and RX cases initialization
            if(global_param->get_number(RX)<2 and global_param->get_number(TX)<2){
                for(int i = 0; i < modes.size(); i++ ){
                
                    if(modes[i]->mode != OFF){
                        if((modes[i]->burst_on != 0) and (modes[i]->burst_on == 0)){
                            print_error("one parameter has burst_off != 0 and burst_on == 0");
                            exit(-1);
                        }
                        if((modes[i]->burst_on == 0) and (modes[i]->burst_on != 0)){
                            print_error("one parameter has burst_on != 0 and burst_off == 0");
                            exit(-1);
                        }
                        if(modes[i]->burst_on != 0){
                            modes[i]->buffer_len = modes[i]->burst_on * modes[i]->rate;
                            std::cout<<"Resizing buffer length to match burst length to: "<< modes[i]->buffer_len<<" samples."<<std::endl;
                        }
                    }
                
                    switch(modes[i]->mode){
                        case OFF:
                            break;
                            
                        case RX:
                        
                            //set the current rx parameter set
                            current_rx_param = modes[i];
                            
                            //adjust the memory buffer in case of custom buffer length
                            if(modes[i]->buffer_len != rx_buffer_len){
                                std::cout<<"Adjusting RX memory buffer size to: "<< (modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::flush;
                                rx_memory->close();
                                rx_buffer_len = modes[i]->buffer_len;
                                rx_memory = new preallocator<float2>(rx_buffer_len,RX_QUEUE_LENGTH);
                                std::cout<<"\tdone."<<std::endl;
                            }
                            
                            //allocates or adjusts the output buffer memory
                            if(output_memory_size != modes[i]->buffer_len){//->get_output_buffer_size()){
                                std::cout<<"Adjusting RX output memory buffer size to: "<< (modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::flush;

                                if(output_memory_size != 0) rx_output_memory->close();
                                
                                output_memory_size = modes[i]->buffer_len;//get_output_buffer_size();
                                //print_debug("outmemory size is: ",output_memory_size);//TODO can be adjusted to use less memory
                                rx_output_memory = new preallocator<float2>(output_memory_size,RX_QUEUE_LENGTH);
                                std::cout<<"\tdone."<<std::endl;
                            }
                            
                            //update pointers
                            if(tcp_streaming)TCP_streamer->update_pointers(stream_queue, rx_output_memory);
                            if(file_writing){
                                tcp_streaming?
                                    H5_writer->update_pointers(TCP_streamer->out_queue, TCP_streamer->memory):
                                    H5_writer->update_pointers(stream_queue, rx_output_memory);
                            }
                            
                            //initialize the demodulator class
                            rx_dem = new RX_buffer_demodulator(modes[i]);
                            
                            break;
                            
                        case TX:
                        
                            //set the current tx parameter set
                            current_tx_param = modes[i];
                            
                            //adjust the memory buffer in case of custom buffer length or pint the memori to NULL
                            if((tx_memory==NULL or modes[i]->buffer_len != tx_buffer_len) and modes[i]->dynamic_buffer()){
                                std::cout<<"Adjusting TX memory buffer size to: "<< (modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::flush;
                                tx_memory->close();
                                tx_buffer_len = modes[i]->buffer_len;
                                
                                //NOTE: this queue doesn't autorefill
                                tx_memory = new preallocator<float2>(tx_buffer_len,TX_QUEUE_LENGTH,false);
                                std::cout<<"\tdone."<<std::endl;
                                
                            //in case a static memory is required
                            }else if (not modes[i]->dynamic_buffer()){
                                if(tx_memory)tx_memory->close();
                                tx_memory = NULL;
                            }
                            
                            //create the buffer generator
                            tx_gen = new TX_buffer_generator(modes[i]);
                            
                            //prefill the queue and save the number of packets
                            //WARNING: something was wrong with that func. see the TX_buffer_generator class
                            preallocated = 0;// tx_gen->prefill_queue(hardware->TX_queue, tx_memory, modes[i]);
                            break;
                    }
                }
            }else{
                print_error("Dual tx rx mode has to be implemented in the thread link library");
                exit(-1);
            }
            
            std::cout<<"\033[1;32mSetting USRP hardware:\033[0m"<<std::endl;
            hardware->preset_usrp(global_param);
            
        }

        //start the threads
        void start(usrp_param* global_param){
            
            //current_tx_param is NULL if no param struct in global_param has TX mode
            if(current_tx_param){
                if(not TX_status and not hardware->check_tx_status()){
                    
                    //start the TX worker: this thread produces the samples and push them in a queue read by the next thread
                    TX_worker = new boost::thread(boost::bind(&TXRX::tx_single_link,this,
                        tx_memory,  //memory preallocator
                        tx_gen,     //signal generator class
                        hardware->TX_queue,   //has the queue to the tx loader thread
                        current_tx_param->samples,
                        current_tx_param->dynamic_buffer(),
                        preallocated    ));
                       
                    //start the TX loader: this thrad takes samples form the other thread and stream them on the USRP
                    hardware->start_tx(
                        tx_conditional_waiting,
                        tx_memory   );
                
                }else{
                    std::stringstream ss;
                    ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on TX";
                    print_error(ss.str());
                    return;
                }
            }
            
            //current_rx_param is NULL if no param struct in global_param has RX mode
            if(current_rx_param){
                if(not RX_status and not hardware->check_rx_status()){
                    
                    //start the RX worker: takes samples from the queue of the next thread, analyze them and push the result in the streaming queue
                    RX_worker = new boost::thread(boost::bind(&TXRX::rx_single_link,this,
                        rx_memory,
                        rx_output_memory,
                        rx_dem,
                        hardware,
                        current_rx_param->samples,
                        stream_queue    ));
                        
                    //start the RX thread: interfaces with the USRP receiving samples and pushing them in a queue read by the thread launched above.
                    hardware->start_rx(
                        rx_buffer_len,
                        current_rx_param->samples,
                        rx_conditional_waiting,
                        rx_memory   );
                    
                    
                    //eventually start streaming and writing    
                    if(file_writing){
                        H5_writer->start(global_param);
                        print_debug("starting write",0) ;
                    }
                    if(tcp_streaming){
                        if(not TCP_streamer->start(current_rx_param)){
                            stop(true);
                        }
                        print_debug("starting net",0) ;
                    }
            
                }else{
                    std::stringstream ss;
                    ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on RX";
                    print_error(ss.str());
                    return;
                }
            }
        }
        //check if the streamer can take a new command and clean the threads for it.
        //in case the force option is true, force close the threads and cleans the queues
        // NOTE: with force option true this call is blocking
        bool stop(bool force = false){
        
            if(tcp_streaming)if (TCP_streamer->NEED_RECONNECT == true)force = true;
            
            bool status = true;
            if(current_rx_param){
                
                bool data_output_status;
                data_output_status = (file_writing or tcp_streaming)?true:false;
                //give the stop command to the filewriter and streamer only if the hardware and dsp are done
                bool hw_status = RX_status or hardware->check_rx_status();
                bool tcp_status = tcp_streaming?(hw_status?true:TCP_streamer->stop(force)):false;
                bool wrt_status = file_writing?(tcp_status?true:(hw_status?true:H5_writer->stop())):false;
                data_output_status = tcp_status or wrt_status;
                if(diagnostic){
                    print_debug("sw_rx is_active: ",RX_status);
                    print_debug("hw_rx is_active: ",hardware->check_rx_status());
                    print_debug("sw_tx is_active: ",TX_status);
                    print_debug("hw_tx is_active: ",hardware->check_tx_status());
                    if(file_writing)print_debug("sw_wrt is_active: ",wrt_status);
                    if(tcp_streaming)print_debug("sw_stream is_active: ",tcp_status);
                }
                if(((not RX_status) and (not hardware->check_rx_status()) and (not data_output_status)) or force){
                    
                    //close the rx interface thread
                    hardware->close_rx();
                    
                    //close the rx worker
                    RX_worker->interrupt();
                    RX_worker->join();
                    
                    //reset the parameter pointer
                    current_rx_param = NULL;
                    
                    //force close data output threads                    
                    if(file_writing)H5_writer->stop(true);
                    //if(tcp_streaming)TCP_streamer->stop(true);
                    
                    //everything's fine for rx
                    status = status and true;
                    
                //if the threads are still running
                }else{status = status and false;}
                
            }
            
            if(current_tx_param){
                if((not TX_status and not hardware->check_tx_status()) or force){
                    //close the rx interface thread
                    hardware->close_tx();
                    
                    //close the rx worker
                    TX_worker->interrupt();
                    TX_worker->join();
                    
                    //reset the parameter pointer
                    current_tx_param = NULL;
                    
                    //everything's fine for rt
                    status = status and true;
                    
                //if the threads are still running
                }else{status = status and false;}
            }
            
            return status;
        }
        
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
        ){
            //notify that the tx worker is on
            TX_status = true;
            
            //pointer to the transmission buffer
            float2* tx_vector;
            
            //number of samples "sent" in to the tx queue
            long int sent_samples = preallocated*tx_buffer_len;
            
            //thread loop controller
            bool active = true;
            printf("SAMPLES AT thread LEVEL %lu\n", total_samples);
            //main loading loop
            while((sent_samples < total_samples) and active){
                try{
                    boost::this_thread::interruption_point();
                    sent_samples+=tx_buffer_len;
                    if(dynamic)tx_vector = memory->get();
                    generator->get(&tx_vector);
                    //print_debug("pushing..",sent_samples);
                    bool insert = false;
                    while(not insert){
                        insert = queue_tx->push(tx_vector);
                        
                        //print_debug("response..",insert);
                        boost::this_thread::sleep_for(boost::chrono::milliseconds{1});
                    }
                }catch(boost::thread_interrupted &){ active = false; 

                }
            }
            
            //notify that the tx worker is off
            TX_status = false;
        }
        
        //thread for taking a packet from the receive queue and pushing it into the analysis queue
        void rx_single_link(
            preallocator<float2>* input_memory,
            preallocator<float2>* output_memory,
            RX_buffer_demodulator* demodulator,
            hardware_manager* rx_thread,
            size_t max_samples,
            rx_queue* stream_q    //pointer to the queue to transport the buffer wrapper structure from the analysis to the streaming thread
        ){
            //notify that the rx worker is on
            RX_status = true;
        
            //wrapper to the buffer
            RX_wrapper rx_buffer;
            
            //pointer to the new buffer
            float2* output_buffer;
            
            //controls thread loop
            bool active = true;
            
            //counter of samples to acquire
            size_t recv_samples = 0;
            
            //saturation warning will only be displayed once.
            bool queue_saturnation_warning = true;
            
            //analysis cycle
            while(active and (recv_samples < max_samples)){
                try{
                
                    //look for abrupt interruptions
                    boost::this_thread::interruption_point();
                    
                    //check if a new packet is in the queue
                    if(rx_thread->RX_queue->pop(rx_buffer)){
                        
                        //the number of channels will be the same for all the measure
                        rx_buffer.channels = demodulator->parameters->wave_type.size();
                    
                        //update the counter
                        recv_samples += rx_buffer.length;
                        
                        //get the output memory available
                        output_buffer = output_memory->get();
                        
                        //analyze the packet and change the valid length of the packet
                        rx_buffer.length = demodulator->process(&rx_buffer.buffer, &output_buffer);
                        
                        //recycle the input buffer
                        input_memory->trash(rx_buffer.buffer);
                        
                        //point to the right buffer in the wrapper
                        rx_buffer.buffer = output_buffer;
                        
                        //push the buffer in the output queue
                        short att = 0;
                        while (not stream_q->push(rx_buffer)){
                            att++;
                            boost::this_thread::sleep_for(boost::chrono::microseconds{5});
                            if(queue_saturnation_warning and att>10){
                                print_warning("Network streaming queue saturated");
                                queue_saturnation_warning = false;
                            }
                        }
                        
                    //if no packet is present sleep for some time, there is no criticality here
                    }else{boost::this_thread::sleep_for(boost::chrono::milliseconds{2});}
                
                //chatch the interruption of the thread
                }catch (boost::thread_interrupted &){ active = false; }
            }
            //exit operation is clean the rx queue to avoid memory leak is a interrupted measure situation
            while(not rx_thread->RX_queue->empty()){
                rx_thread->RX_queue->pop(rx_buffer);
                input_memory->trash(rx_buffer.buffer);
            }
            
            //notify that the rx worker is off
            RX_status = false;
        }
        
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
class json_2_param{

    public:
        
        //the initialization method requires
        json_2_param(){
        
        
        }

        //bool check_parameters(){}

        void get();
        
    private:
    
        //this function will read the arrays inside a json file and put them in a std vector.
        template <typename T>
        std::vector<T> as_vector(boost::property_tree::ptree const& pt,
                                 boost::property_tree::ptree::key_type const& key,
                                 boost::property_tree::ptree::key_type const& sub_key = "NULL"
                                 ){
            std::vector<T> r;
            if (sub_key == "NULL"){
                for (auto& item : pt.get_child(key)) r.push_back(item.second.get_value<T>());
            }else{
                for (auto& item : pt.get_child(key).get_child(sub_key)) r.push_back(item.second.get_value<T>());
            }
            return r;
        }

};
#endif
