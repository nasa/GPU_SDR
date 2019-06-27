#include "USRP_server_link_threads.hpp"


//the initialization method requires an already initialized hardware manager class and an already initialized streaming queue (output of analysis)
TXRX::TXRX(server_settings* settings, hardware_manager* init_hardware, bool diagnostic_init){
    BOOST_LOG_TRIVIAL(info) << "Initializing thread link class";
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
    A_rx_buffer_len = settings->default_rx_buffer_len;
    A_tx_buffer_len = settings->default_tx_buffer_len;
    B_rx_buffer_len = settings->default_rx_buffer_len;
    B_tx_buffer_len = settings->default_tx_buffer_len;

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

    //what follows is the initialization of the memory allocator pointers.
    //do NOT skip that part as the system will use the pointers to detect memory allocation status
    //and unallocated memory will result in if(pointer) to be compiled as if(true)

    //set the pointer to current parameter configuration
    A_current_tx_param = nullptr;
    A_current_rx_param = nullptr;
    B_current_tx_param = nullptr;
    B_current_rx_param = nullptr;

    //initializing memory pointers
    A_tx_memory = nullptr;
    A_rx_memory = nullptr;
    B_tx_memory = nullptr;
    B_rx_memory = nullptr;

    rx_output_memory = nullptr;

    BOOST_LOG_TRIVIAL(info) << "Thread link class initialized";
}

//launches the setting functions for the required signals, antennas...
void TXRX::set(usrp_param* global_param){

    BOOST_LOG_TRIVIAL(info) << "Setting thread link class";

    std::vector<param*> modes(4);
    modes[0] = &global_param->A_TXRX;
    modes[1] = &global_param->A_RX2;
    modes[2] = &global_param->B_TXRX;
    modes[3] = &global_param->B_RX2;

    //reset thread counting mechanism
    thread_counter = 0;
    rx_thread_n.clear();
    tx_thread_n.clear();

    for(size_t i = 0; i < modes.size(); i++ ){

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
        int this_thread_n = thread_counter*2;
        switch(modes[i]->mode){
            case OFF:
                break;

            case RX:
                std::cout<<"Allocating RF frontend "<<(char)(i<2?'A':'B')<<" RX memory buffer: "<< (modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::endl;
                if(i<2){
                    if(A_rx_buffer_len != modes[i]->buffer_len or not A_rx_memory){
                        if(A_rx_memory)A_rx_memory->close();
                        A_rx_memory = new preallocator<float2>(A_rx_buffer_len,RX_QUEUE_LENGTH,this_thread_n);
                    }else{
                        std::cout<<"(already allocated).."<<std::endl;
                    }
                    A_rx_buffer_len = modes[i]->buffer_len;

                    //initialize the demodulator class
                    A_rx_dem = new RX_buffer_demodulator(modes[i]);

                    A_current_rx_param = modes[i];

                }else{

                    if(B_rx_buffer_len != modes[i]->buffer_len or not B_rx_memory){
                        if(B_rx_memory)B_rx_memory->close();
                        B_rx_memory = new preallocator<float2>(B_rx_buffer_len,RX_QUEUE_LENGTH,this_thread_n);
                    }else{
                        std::cout<<"(already allocated).."<<std::endl;
                    }
                    B_rx_buffer_len = modes[i]->buffer_len;

                    //initialize the demodulator class
                    B_rx_dem = new RX_buffer_demodulator(modes[i]);

                    B_current_rx_param = modes[i];
                }

                rx_thread_n.push_back(this_thread_n);
                thread_counter +=1;
                mem_mult_tmp = std::max((int)(modes[i]->data_mem_mult),1);
                print_debug("memory expansion multiplier: ",mem_mult_tmp);
                if ((output_memory_size>modes[i]->buffer_len * mem_mult_tmp) or not rx_output_memory){
                    std::cout<<"Allocating RX output memory buffer: "<< (mem_mult_tmp * modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::endl;
                    if(modes[i]->buffer_len>output_memory_size and output_memory_size>0)std::cout<<" (updating buffer size)"<<std::endl;
                    if(rx_output_memory) rx_output_memory->close();
                    output_memory_size = modes[i]->buffer_len * mem_mult_tmp;
                    rx_output_memory = new preallocator<float2>(output_memory_size,RX_QUEUE_LENGTH);

                }else{
                    std::cout<<" RX output memory buffer requirements already satisfaid."<<std::endl;
                }


                //update pointers
                if(tcp_streaming)TCP_streamer->update_pointers(stream_queue, rx_output_memory);
                if(file_writing){
                    tcp_streaming?
                        H5_writer->update_pointers(TCP_streamer->out_queue, TCP_streamer->memory):
                        H5_writer->update_pointers(stream_queue, rx_output_memory);
                }

                break;

            case TX:

                //adjust the memory buffer in case of custom buffer length or pint the memori to nullptr
                std::cout<<"Allocating RF frontend "<<(char)(i<2?'A':'B')<<" TX memory buffer: "<< (modes[i]->buffer_len * sizeof(float2))/(1024.*1024.)<< " MB per buffer..."<<std::endl;
                //NOTE: this queue doesn't autorefill

                if(i<2){
                    if(modes[i]->dynamic_buffer()){
                        if(not A_tx_memory or modes[i]->buffer_len != A_tx_buffer_len){
                            if(A_tx_memory){
                                A_tx_memory->close();
                            }
                            A_tx_memory = new preallocator<float2>(modes[i]->buffer_len,TX_QUEUE_LENGTH,false,this_thread_n);
                            A_tx_buffer_len = modes[i]->buffer_len;
                        }else{
                            std::cout<<"(already allocated).."<<std::flush;
                        }
                    }else{
                        if(A_tx_memory){
                            A_tx_memory->close();
                        }
                        A_tx_memory = nullptr;
                    }

                    A_tx_gen = new TX_buffer_generator(modes[i]);

                    A_preallocated = 0;

                    A_current_tx_param = modes[i];

                }else{
                    if(modes[i]->dynamic_buffer()){
                        if(not B_tx_memory or modes[i]->buffer_len != B_tx_buffer_len){
                            B_tx_memory = new preallocator<float2>(modes[i]->buffer_len,TX_QUEUE_LENGTH,false,this_thread_n);
                            B_tx_buffer_len = modes[i]->buffer_len;
                        }else{
                            std::cout<<"(already allocated).."<<std::flush;
                        }
                    }else{
                        if(B_tx_memory){
                            B_tx_memory->close();
                        }
                        B_tx_memory = nullptr;
                    }

                    B_tx_gen = new TX_buffer_generator(modes[i]);

                    //prefill the queue and save the number of packets
                    //WARNING: something was wrong with that func. see the TX_buffer_generator class
                    B_preallocated = 0;// tx_gen->prefill_queue(hardware->TX_queue, tx_memory, modes[i]);

                    B_current_tx_param = modes[i];

                }
                tx_thread_n.push_back(this_thread_n);

                thread_counter+=1;

                break;
        }
    }


    std::cout<<"\033[1;32mSetting USRP hardware:\033[0m"<<std::endl;
    hardware->preset_usrp(global_param);

    BOOST_LOG_TRIVIAL(info) << "Thread link class set";

}

//start the threads
void TXRX::start(usrp_param* global_param){

	BOOST_LOG_TRIVIAL(info) << "Starting thread link";

    //counts the started threads
    int rx_threads = 0;
    int tx_threads = 0;

    if(not hardware->sw_loop)hardware->main_usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));

    //current_tx_param is nullptr if no param struct in global_param has TX mode
    if(global_param->A_TXRX.mode!=OFF){
        if(not hardware->check_A_tx_status()){

            //start the TX worker: this thread produces the samples and push them in a queue read by the next thread
            A_TX_worker = new boost::thread(boost::bind(&TXRX::tx_single_link,this,
                A_tx_memory,  //memory preallocator
                A_tx_gen,     //signal generator class
                hardware->A_TX_queue,   //has the queue to the tx loader thread
                global_param->A_TXRX.samples,
                global_param->A_TXRX.dynamic_buffer(),
                A_preallocated,
                'A'    ));

            SetThreadName(A_TX_worker, "A_TX_worker");

            Thread_Prioriry(*A_TX_worker, 1, 6);//tx_thread_n[tx_threads]+1);

            //start the TX loader: this thrad takes samples form the other thread and stream them on the USRP
            hardware->start_tx(
                tx_conditional_waiting,
                6,//tx_thread_n[tx_threads]+1,
                &(global_param->A_TXRX),
                'A',
                A_tx_memory
            );
            tx_threads += 1;

        }else{
            std::stringstream ss;
            ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on TX";
            print_error(ss.str());
            return;
        }
    }else A_TX_worker = nullptr;

    if(global_param->B_TXRX.mode!=OFF){
        if(not hardware->check_B_tx_status()){

            //start the TX worker: this thread produces the samples and push them in a queue read by the next thread
            B_TX_worker = new boost::thread(boost::bind(&TXRX::tx_single_link,this,
                B_tx_memory,  //memory preallocator
                B_tx_gen,     //signal generator class
                hardware->B_TX_queue,   //has the queue to the tx loader thread
                global_param->B_TXRX.samples,
                global_param->B_TXRX.dynamic_buffer(),
                B_preallocated,
                'B'    ));

            SetThreadName(B_TX_worker, "B_TX_worker");

            Thread_Prioriry(*B_TX_worker, 1, 4);//tx_thread_n[tx_threads]+1);


            //start the TX loader: this thrad takes samples form the other thread and stream them on the USRP
            hardware->start_tx(
                tx_conditional_waiting,
                4,//tx_thread_n[tx_threads]+1,
                &(global_param->B_TXRX),
                'B',
                B_tx_memory
            );
            tx_threads += 1;

        }else{
            std::stringstream ss;
            ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on TX";
            print_error(ss.str());
            return;
        }
    }else B_TX_worker = nullptr;

    //current_rx_param is nullptr if no param struct in global_param has RX mode
    if(global_param->A_RX2.mode!=OFF){
        if(not hardware->check_A_rx_status()){

            //start the RX worker: takes samples from the queue of the next thread, analyze them and push the result in the streaming queue
            A_RX_worker = new boost::thread(boost::bind(&TXRX::rx_single_link,this,
                A_rx_memory,
                rx_output_memory,
                A_rx_dem,
                hardware,
                global_param->A_RX2.samples,
                stream_queue,
                'A'    ));

            SetThreadName(A_RX_worker, "A_RX_worker");

            Thread_Prioriry(*A_RX_worker, 1, 0);//rx_thread_n[rx_threads]*2+1);

            //start the RX thread: interfaces with the USRP receiving samples and pushing them in a queue read by the thread launched above.
            hardware->start_rx(
                A_rx_buffer_len,
                rx_conditional_waiting,
                A_rx_memory,
                0,//rx_thread_n[rx_threads]*2+1,
                &(global_param->A_RX2),
                'A'
            );

            rx_threads += 1;


        }else{
            std::stringstream ss;
            ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on RX";
            print_error(ss.str());
            return;
        }
    }else A_RX_worker = nullptr;

    if(global_param->B_RX2.mode!=OFF){
        if(not hardware->check_B_rx_status()){

            //start the RX worker: takes samples from the queue of the next thread, analyze them and push the result in the streaming queue
            B_RX_worker = new boost::thread(boost::bind(&TXRX::rx_single_link,this,
                B_rx_memory,
                rx_output_memory,
                B_rx_dem,
                hardware,
                global_param->B_RX2.samples,
                stream_queue,
                'B'    ));

            SetThreadName(B_RX_worker, "B_RX_worker");

            Thread_Prioriry(*B_RX_worker, 1, 2);//rx_thread_n[rx_threads]*2+1);

            //start the RX thread: interfaces with the USRP receiving samples and pushing them in a queue read by the thread launched above.
            hardware->start_rx(
                B_rx_buffer_len,
                rx_conditional_waiting,
                B_rx_memory,
                2,//rx_thread_n[rx_threads]*2,
                &(global_param->B_RX2),
                'B'
            );

            rx_threads += 1;

            //eventually start streaming and writing


        }else{
            std::stringstream ss;
            ss << "Cannot start a new measurement on usrp "<< hardware->this_usrp_number <<" if there is already one running on RX";
            print_error(ss.str());
            return;
        }
    }else B_RX_worker = nullptr;

    if ((global_param->A_RX2.mode!=OFF) and (global_param->B_RX2.mode!=OFF)){
        if(file_writing){
            H5_writer->start(global_param);
        }
        if(tcp_streaming){
            //! @todo TODO: this should take the parameter with maximum buffer size. Probably a segm fault will happen in case of different buffer sizes.
            if(not TCP_streamer->start(&(global_param->A_RX2))){
                stop(true);
            }
        }
    }else if (global_param->B_RX2.mode!=OFF){
        if(file_writing){
            H5_writer->start(global_param);
        }
        if(tcp_streaming){
            if(not TCP_streamer->start(&(global_param->B_RX2))){
                stop(true);
            }
        }
    }else if (global_param->A_RX2.mode!=OFF){
        if(file_writing){
            H5_writer->start(global_param);
        }
        if(tcp_streaming){
            if(not TCP_streamer->start(&(global_param->A_RX2))){
                stop(true);
            }
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Thread link started";

}
//check if the streamer can take a new command and clean the threads for it.
//in case the force option is true, force close the threads and cleans the queues
// NOTE: with force option true this call is blocking
bool TXRX::stop(bool force){
    if(tcp_streaming)if (TCP_streamer->NEED_RECONNECT == true)force = true;

    bool status = true;


    if(A_current_rx_param or B_current_rx_param){

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

       hardware->check_rx_status();
        if(((not RX_status) and (not hardware->check_rx_status()) and (not data_output_status)) or force){
            hardware->close_rx();

            if (A_RX_worker){
                //close the rx worker
                A_RX_worker->interrupt();
                A_RX_worker->join();
                delete A_RX_worker;
                A_RX_worker = nullptr;
                //reset the parameter pointer
                A_current_rx_param = nullptr;
            }
            if (B_RX_worker){
                //close the rx worker
                B_RX_worker->interrupt();
                B_RX_worker->join();
                delete B_RX_worker;
                B_RX_worker = nullptr;
                //reset the parameter pointer
                B_current_rx_param = nullptr;
            }

            //force close data output threads
            if(file_writing)H5_writer->stop(true);
            //if(tcp_streaming)TCP_streamer->stop(true);
        //if the threads are still running
        }else{status = status and false;}
    }

    if(A_current_tx_param or B_current_tx_param){
    	if(diagnostic){
			print_debug("TX_status ",TX_status);
			print_debug("hardware->check_tx_status() ",hardware->check_tx_status());
		}
        if((not TX_status and not hardware->check_tx_status()) or force){
            //close the rx interface thread
            hardware->close_tx();

            //close the rx worker
            if (A_TX_worker){
                A_TX_worker->interrupt();
                A_TX_worker->join();
                delete A_TX_worker;
                A_TX_worker=nullptr;
                A_tx_gen->close();

                //reset the parameter pointer
                A_current_tx_param = nullptr;
            }
            if (B_TX_worker){
                B_TX_worker->interrupt();
                B_TX_worker->join();
                delete B_TX_worker;
                B_TX_worker = nullptr;

                B_tx_gen->close();

                //reset the parameter pointer
                B_current_tx_param = nullptr;

            }

        //if the threads are still running
        }else{status = status and false;}
    }

    //reset the thread counter
    if(status)thread_counter = 0;

    BOOST_LOG_TRIVIAL(info) << "Operations concluded? "<<status;

    return status;
}

//thread for loading a packet from the buffer generator into the transmit thread
//assuming a single TX generator and a single TX loader
void TXRX::tx_single_link(
    preallocator<float2>* memory, //the custom memory allocator to use in case of dynamically denerated buffer
    TX_buffer_generator* generator, //source of the buffer
    tx_queue* queue_tx, //holds the pointer to the queue
    size_t total_samples, //how many sample to produce and push
    bool dynamic, //true if the preallocation requires dynamic memory
    int preallocated, // how many samples have been preallocate
    char front_end
){

	std::stringstream thread_name;
    thread_name << "tx single link  "<<front_end;
    set_this_thread_name(thread_name.str());

	BOOST_LOG_TRIVIAL(info) << "Thread started";

    if(front_end!='A' and front_end!='B'){
        print_error("Frontend code not recognised in transmission link thread");
        return;
    }



    //notify that the tx worker is on
    TX_status = true;

    //pointer to the transmission buffer
    float2* tx_vector;

    size_t tx_buffer_len = front_end=='A'?A_tx_buffer_len:B_tx_buffer_len;

    //number of samples "sent" in to the tx queue
    size_t sent_samples = preallocated*tx_buffer_len;

    //thread loop controller
    bool active = true;
    //main loading loop
    while((sent_samples < total_samples) and active){
        try{
            boost::this_thread::interruption_point();
            sent_samples+=tx_buffer_len;
            if(dynamic)tx_vector = memory->get();
            generator->get(&tx_vector);
            bool insert = false;
            while(not insert){
                insert = queue_tx->push(tx_vector);
                boost::this_thread::sleep_for(boost::chrono::microseconds{1});
            }
            //std::cout<<"Pushing buffer"<<std::endl;
            boost::this_thread::sleep_for(boost::chrono::microseconds{1});
        }catch(boost::thread_interrupted &){ active = false;


        }
    }
    //notify that the tx worker is off
    TX_status = false;

    BOOST_LOG_TRIVIAL(info) << "Thread joining";

}

//thread for taking a packet from the receive queue and pushing it into the analysis queue
void TXRX::rx_single_link(
    preallocator<float2>* input_memory,
    preallocator<float2>* output_memory,
    RX_buffer_demodulator* demodulator,
    hardware_manager* rx_thread,
    size_t max_samples,
    rx_queue* stream_q,    //pointer to the queue to transport the buffer wrapper structure from the analysis to the streaming thread
    char front_end
){

	std::stringstream thread_name;
    thread_name << "rx single link  "<<front_end;
    set_this_thread_name(thread_name.str());

	BOOST_LOG_TRIVIAL(info) << "Thread started";

    if(front_end!='A' and front_end!='B'){
        print_error("Frontend code not recognised in receiver link thread");
        return;
    }

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

    rx_queue* RX_queue = front_end=='A'?rx_thread->A_RX_queue:rx_thread->B_RX_queue;

    //analysis cycle
    while(active and (recv_samples < max_samples)){
        try{

            //look for abrupt interruptions
            boost::this_thread::interruption_point();

            //check if a new packet is in the queue
            if(RX_queue->pop(rx_buffer)){

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
    RX_wrapper rx_buffer_dummy;
    while(not RX_queue->empty()){
        RX_queue->pop(rx_buffer_dummy);
        input_memory->trash(rx_buffer_dummy.buffer);
    }

    //notify that the rx worker is off
    RX_status = false;

    BOOST_LOG_TRIVIAL(info) << "Thread joining";
}
