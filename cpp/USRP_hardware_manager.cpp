#include "USRP_hardware_manager.hpp"

auto start = std::chrono::system_clock::now();

//! @brief Initializer of the class can be used to select which usrp is controlled by the class
//! Default call suppose only one USRP is connected
//! @todo TODO: the multi_usrp object has to be passed as argument to this initializer. Multiple usrp's will crash as the obj is not ts
hardware_manager::hardware_manager(server_settings* settings, bool sw_loop_init, size_t usrp_number){

	BOOST_LOG_TRIVIAL(info) << "Initializing hardware manager";


    //software loop mode exclude the hardware
    sw_loop = sw_loop_init;

    if(sw_loop)BOOST_LOG_TRIVIAL(debug) << "Software loop enabled";

    //in any case a gpu is necessary
    cudaSetDevice(settings->GPU_device_index);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, settings->GPU_device_index);


    if(not sw_loop){

        this_usrp_number = usrp_number;
;

        //recursively look for usrps
        dev_addrs = uhd::device::find(hint);
        std::cout<<"Looking for USRP x300 device number "<< usrp_number << " .." <<std::flush;
        while(dev_addrs.size()< usrp_number + 1){

            dev_addrs = uhd::device::find(hint);
            std::cout<<"."<<std::flush;
            usleep(1e6);
        }


        std::cout<<"Device found and assigned to GPU "<< props.name <<" ("<< settings->GPU_device_index <<")"<<std::endl;
        //for(size_t ii = 0; ii<dev_addrs.size(); ii++){
        //    std::cout<<dev_addrs[ii].to_pp_string()<<std::endl;
        //}
        //assign desired address


				//uhd::device_addr_t args("addr=192.168.30.2,second_addr=192.168.40.2");
				if(device_arguments.compare("noarg")!=0){
					uhd::device_addr_t args(device_arguments);
					main_usrp = uhd::usrp::multi_usrp::make(args);
					std::cout<< "Creating device with arguments: "<<device_arguments <<std::endl;
				}else{
					main_usrp = uhd::usrp::multi_usrp::make(dev_addrs[usrp_number]);
				}
        //set the clock reference
        main_usrp->set_clock_source(settings->clock_reference);

    }else{
        A_sw_loop_queue = new tx_queue(SW_LOOP_QUEUE_LENGTH);
        B_sw_loop_queue = new tx_queue(SW_LOOP_QUEUE_LENGTH);
    }

    //initialize port connection check variables
    A_TXRX_chk = OFF;
    B_RX2_chk = OFF;
    B_TXRX_chk = OFF;
    A_RX2_chk = OFF;

    //set the thread state
    A_rx_thread_operation = false;
    A_tx_thread_operation = false;
    B_rx_thread_operation = false;
    B_tx_thread_operation = false;

    //settling time for fpga register initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    //initialize transmission queues
    A_RX_queue = new rx_queue(RX_QUEUE_LENGTH);
    A_TX_queue = new tx_queue(TX_QUEUE_LENGTH);
    B_RX_queue = new rx_queue(RX_QUEUE_LENGTH);
    B_TX_queue = new tx_queue(TX_QUEUE_LENGTH);

    A_tx_error_queue = new error_queue(ERROR_QUEUE_LENGTH);
    B_tx_error_queue = new error_queue(ERROR_QUEUE_LENGTH);

    A_rx_stream = nullptr;
    A_tx_stream = nullptr;
    B_rx_stream = nullptr;
    B_tx_stream = nullptr;

    if(not sw_loop)main_usrp->set_time_now(0.);

    BOOST_LOG_TRIVIAL(info) << "Hardware manager initilaized";
}

//! @brief This function set the USRP device with user parameters.
//! It's really a wrappe raround the private methods apply(), set_streams() and check_tuning() of this class.
//! @todo TODO catch exceptions and return a boolean
bool hardware_manager::preset_usrp(usrp_param* requested_config){

	if(not sw_loop){
		BOOST_LOG_TRIVIAL(info) << "Presetting USRP";
	}else{
		BOOST_LOG_TRIVIAL(info) << "Presetting sw loop queue";
	}

    apply(requested_config);
    set_streams();
    if(not sw_loop){
        check_tuning();
    }

    BOOST_LOG_TRIVIAL(info) << "Preset done";
    return true;

}

bool hardware_manager::check_A_rx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = A_rx_thread_operation;
    if(verbose)print_debug("RX thread status: ",op);
    return op;
}

bool hardware_manager::check_B_rx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = B_rx_thread_operation;
    if(verbose)print_debug("RX thread status: ",op);
    return op;
}

bool hardware_manager::check_rx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = B_rx_thread_operation or A_rx_thread_operation;
    if(verbose)print_debug("RX thread status: ",op);
    return op;
}

bool hardware_manager::check_tx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = B_tx_thread_operation or A_tx_thread_operation;
    if(verbose)print_debug("TX thread status: ",op);
    return op;
}

bool hardware_manager::check_A_tx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = A_tx_thread_operation;
    if(verbose)print_debug("TX thread status: ",op);
    return op;
}

bool hardware_manager::check_B_tx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    bool op = B_tx_thread_operation;
    if(verbose)print_debug("TX thread status: ",op);
    return op;
}

//! @brief Start a transmission thread.
//! The threads started by this function do two things: pop a packet from the respective queue; stram the packet via UHD interface.
//! Each streamer is handled by an independent thread.
//! If the source queue is empty a warning is printed on the console and an error is pushed in the erorr queue.
void hardware_manager::start_tx(
    threading_condition* wait_condition,    //before joining wait for that condition
    int thread_op,                          //core affinity of the process
    param *current_settings,                //representative of the paramenters (must match A or B frontend description)
    char front_end,                         //must be "A" or "B"
preallocator<float2>* memory                //if the thread is transmitting a buffer that requires dynamical allocation than a pointer to  custo memory manager class has to be passed
){

	BOOST_LOG_TRIVIAL(debug) << "Starting tx threads";
    bool tx_thread_operation;

    if(front_end=='A'){
        tx_thread_operation = A_tx_thread_operation;
    }else if(front_end=='B'){
        tx_thread_operation = B_tx_thread_operation;
    }else{
        print_error("Front end code not recognised by hardware manager");
        return;
    }

    if(not tx_thread_operation){

        //start the thread
        if(not sw_loop){
            if(front_end=='A'){
                A_tx_thread = new boost::thread(boost::bind(&hardware_manager::single_tx_thread,this,
                    current_settings,
                    wait_condition,
                    A_TX_queue,
                    A_tx_stream,
                    memory,
                    'A'
                ));
            SetThreadName(A_tx_thread, "A_tx_thread");
            Thread_Prioriry(*A_tx_thread, 99, thread_op);

            }else if(front_end=='B'){
                B_tx_thread = new boost::thread(boost::bind(&hardware_manager::single_tx_thread,this,
                    current_settings,
                    wait_condition,
                    B_TX_queue,
                    B_tx_stream,
                    memory,
                    'B'
                ));

            SetThreadName(B_tx_thread, "B_tx_thread");
            Thread_Prioriry(*B_tx_thread, 99, thread_op);

            }
        }else{
            if(front_end=='A'){
                A_tx_thread = new boost::thread(boost::bind(&hardware_manager::software_tx_thread,this,current_settings,memory,A_TX_queue,A_sw_loop_queue,'A'));
                Thread_Prioriry(*A_tx_thread, 99, thread_op);
            }else if(front_end=='B'){
                B_tx_thread = new boost::thread(boost::bind(&hardware_manager::software_tx_thread,this,current_settings,memory,B_TX_queue,B_sw_loop_queue,'B'));
                Thread_Prioriry(*B_tx_thread, 99, thread_op);
            }
        }
    }else{
        std::stringstream ss;
        ss << "Cannot start TX thread, a tx thread associated with USRP "<< this_usrp_number <<" is already running";
        print_error(ss.str());
    }

    BOOST_LOG_TRIVIAL(debug) << "tx threads started";
}

//! @brief Start a receiver thread.
void hardware_manager::start_rx(
    int buffer_len,                         //length of the buffer. MUST be the same of the preallocator initialization
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory,            //custom memory preallocator
    int thread_op,
    param *current_settings, //representative of the paramenters (must match A or B frontend description)
    char front_end //must be "A" or "B"

    ){
    BOOST_LOG_TRIVIAL(debug) << "Starting rx threads";
    bool rx_thread_operation;

    if(front_end=='A'){
        rx_thread_operation = A_rx_thread_operation;
    }else if(front_end=='B'){
        rx_thread_operation = B_rx_thread_operation;
    }else{
        print_error("Front end code not recognised by hardware manager");
        return;
    }

    if(not rx_thread_operation){
        //start the thread
        if(not sw_loop){
            if(front_end=='A'){
                A_rx_thread = new boost::thread(boost::bind(&hardware_manager::single_rx_thread,this,
                    current_settings,
                    A_RX_queue,
                    wait_condition,
                    memory,
                    A_rx_stream,
                    'A'));
                Thread_Prioriry(*A_rx_thread, 99, thread_op);
                SetThreadName(A_rx_thread, "A_rx_thread");
            }else if(front_end=='B'){
                B_rx_thread = new boost::thread(boost::bind(&hardware_manager::single_rx_thread,this,
                    current_settings,
                    B_RX_queue,
                    wait_condition,
                    memory,
                    B_rx_stream,
                    'B'));
                Thread_Prioriry(*B_rx_thread, 99, thread_op);
                SetThreadName(B_rx_thread, "B_rx_thread");
            }
        }else{
            if(front_end=='A'){
                A_rx_thread = new boost::thread(boost::bind(&hardware_manager::software_rx_thread,this,current_settings,memory,A_RX_queue,A_sw_loop_queue,'A'));
                Thread_Prioriry(*A_rx_thread, 99, thread_op);
                SetThreadName(A_rx_thread, "A_rx_thread");
            }else if(front_end=='B'){
                B_rx_thread = new boost::thread(boost::bind(&hardware_manager::software_rx_thread,this,current_settings,memory,B_RX_queue,B_sw_loop_queue,'B'));
                Thread_Prioriry(*B_rx_thread, 99, thread_op);
                SetThreadName(B_rx_thread, "B_rx_thread");
            }
        }

    }else{
        std::stringstream ss;
        ss << "Cannot start RX thread, a rx threead associated with USRP "<< this_usrp_number <<" is already running";
        print_error(ss.str());
    }
    BOOST_LOG_TRIVIAL(debug) << "rx threads started";
}
//! @brief Force close the tx uploading threads if active (thread safe)
void hardware_manager::close_tx(){

    if(A_tx_thread_operation){
        A_tx_thread->interrupt();
        A_tx_thread->join();
        delete A_tx_thread;
        A_tx_thread = nullptr;
        A_tx_thread_operation = false;

    }

    if(B_tx_thread_operation){
        B_tx_thread->interrupt();
        B_tx_thread->join();
        delete B_tx_thread;
        B_tx_thread = nullptr;
        B_tx_thread_operation = false;

    }

}
//! @brief Force close the rx downloading threads if active (thread safe)
void hardware_manager::close_rx(){
    if(A_rx_thread_operation){

        A_rx_thread->interrupt();
        A_rx_thread->join();
        delete A_rx_thread;
        A_rx_thread = nullptr;
        A_rx_thread_operation = false;

    }

    if(B_rx_thread_operation){
        B_rx_thread->interrupt();
        B_rx_thread->join();
        delete B_rx_thread;
        B_rx_thread = nullptr;
        B_rx_thread_operation = false;
    }

}
//! @brief close a TX queue object preventing memory leaks.
//! Before closing the queue (as the queue used supports multiple consumers) the frontend operation MUST be terminated.
int hardware_manager::clean_tx_queue(tx_queue* TX_queue, preallocator<float2>* memory){

	BOOST_LOG_TRIVIAL(info) << "Cleaning tx queue";
    //temporary wrapper
    float2* buffer;

    //counter. Expected to be 0
    int counter = 0;

    while(not TX_queue->empty() or TX_queue->pop(buffer)){

        memory->trash(buffer);
        counter ++;
    }

    if(counter > 0){
        std::stringstream ss;
        ss << "TX queue cleaned of "<< counter <<"buffer(s)";
        print_warning(ss.str());
    }
    BOOST_LOG_TRIVIAL(info) << "tx queue cleaned of packets: "<< counter;
    return counter;
}
//! @brief close a RX queue object preventing memory leaks.
//! Before closing the queue (as the queue used supports multiple consumers) the frontend operation MUST be terminated.
int hardware_manager::clean_rx_queue(rx_queue* RX_queue, preallocator<float2>* memory){

	BOOST_LOG_TRIVIAL(info) << "Cleaning rx queue";

    //temporary wrapper
    RX_wrapper warapped_buffer;
	warapped_buffer.buffer = nullptr;
    //counter. Expected to be 0
    int counter = 0;

    //cannot execute when the rx thread is going
    while(not RX_queue->empty() and RX_queue->pop(warapped_buffer)){
        memory->trash(warapped_buffer.buffer);
        counter ++;
    }

    if(counter > 0){
        std::stringstream ss;
        ss << "RX queue cleaned of "<< counter <<"buffer(s)";
        print_warning(ss.str());
    }
    BOOST_LOG_TRIVIAL(info) << "rx queue cleaned of packets: "<< counter;
    return counter;
}


void hardware_manager::apply(usrp_param* requested_config){
	BOOST_LOG_TRIVIAL(info) << "Applying USRP configuration";
    //transfer the usrp index to the setting parameters
    requested_config->usrp_number = this_usrp_number;

    //stack of messages
    std::stringstream ss;
    ss<<std::endl;

    //apply configuration to each antenna on each subdevice
    ss<<"Hardware parameter subdevice A_TXRX: ";
    switch(requested_config->A_TXRX.mode){
        case RX:
            if(not sw_loop)main_usrp->set_rx_antenna("TX/RX",0);
            break;
        case TX:
            if(not sw_loop)main_usrp->set_tx_antenna("TX/RX",0);
            break;
        case OFF:
            ss<<"Channel is OFF";
            break;
    }
    ss<<std::endl;
    ss<<apply_antenna_config(&(requested_config->A_TXRX), &config.A_TXRX,0);

    ss<<"Hardware parameter subdevice A_RX2: ";
    switch(requested_config->A_RX2.mode){
        case RX:
            if(not sw_loop)main_usrp->set_rx_antenna("RX2",0);
            break;
        case TX:
            if(not sw_loop)main_usrp->set_tx_antenna("RX2",0);
            break;
        case OFF:
            ss<<"Channel is OFF";
            break;
    }
    ss<<std::endl;
    ss<<apply_antenna_config(&(requested_config->A_RX2), &config.A_RX2,0);

    //std::string subdev = "B:0";
    //main_usrp->set_rx_subdev_spec(subdev);

    ss<<"Hardware parameter subdevice B_TXRX: ";
    switch(requested_config->B_TXRX.mode){
        case RX:
            if(not sw_loop)main_usrp->set_rx_antenna("TX/RX",1);
            break;
        case TX:
            if(not sw_loop)main_usrp->set_tx_antenna("TX/RX",1);
            break;
        case OFF:
            ss<<"Channel is OFF";
            break;
    }
    ss<<std::endl;
    ss<<apply_antenna_config(&(requested_config->B_TXRX), &config.B_TXRX,1);

    ss<<"Hardware parameter subdevice B_RX2: ";
    switch(requested_config->B_RX2.mode){
        case RX:
            if(not sw_loop)main_usrp->set_rx_antenna("RX2",1);
            break;
        case TX:
            if(not sw_loop)main_usrp->set_tx_antenna("RX2",1);
            break;
        case OFF:
            ss<<"Channel is OFF";
            break;
    }
    ss<<std::endl;
    ss<<apply_antenna_config(&(requested_config->B_RX2), &config.B_RX2,1);

    std::cout<<ss.str();
	BOOST_LOG_TRIVIAL(info) << "USRP configuration applied";
}

bool hardware_manager::check_tuning(){
	BOOST_LOG_TRIVIAL(info) << "Checking tuning";
    //both rx and tx must be locked if selected
    bool rx = true;
    bool tx = true;

		size_t num_rx_channels = main_usrp->get_rx_num_channels();
		size_t num_tx_channels = main_usrp->get_tx_num_channels();

    for(size_t chan = 0; chan<num_rx_channels; chan++){
        //check for RX locking
        std::vector<std::string>  rx_sensor_names;
        rx_sensor_names = main_usrp->get_rx_sensor_names(chan);

        try{
            //check only if there is a channel associated with RX.
            if(check_global_mode_presence(RX,chan)){
                std::cout<<"Checking RX frontend tuning for channel "<< chan <<" ... "<<std::endl;
                if (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") != rx_sensor_names.end()) {
                    uhd::sensor_value_t lo_locked = main_usrp->get_rx_sensor("lo_locked",chan);

                    //a settling time is normal
                    int timeout_counter = 0;
                    while (not main_usrp->get_rx_sensor("lo_locked",chan).to_bool()){
                        //sleep for a short time in milliseconds
                        timeout_counter++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        if(timeout_counter>500){
                            std::stringstream ss;
                            ss<<"Cannot tune the RX frontend of channel "<<chan;
                            print_error(ss.str());
                            return false;
                        }
                    }
                    lo_locked = main_usrp->get_rx_sensor("lo_locked",chan);
                }
                rx = rx and main_usrp->get_rx_sensor("lo_locked",chan).to_bool();
            }
        }catch (uhd::lookup_error e){
            std::cout<<"None"<<std::endl;
            rx = true;
        }
			}
			for(size_t chan = 0; chan<num_tx_channels; chan++){
        //check for TX locking
        std::vector<std::string>  tx_sensor_names;
        tx_sensor_names = main_usrp->get_tx_sensor_names(chan);
        try{
            //check only if there is a channel associated with TX.
            if(check_global_mode_presence(TX,chan)){
                std::cout<<"Checking TX frontend tuning for channel "<< chan <<" ... "<<std::endl;
                if (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") != tx_sensor_names.end()) {
                    uhd::sensor_value_t lo_locked = main_usrp->get_tx_sensor("lo_locked",chan);

                    //a settling time is normal
                    int timeout_counter = 0;
                    while (not main_usrp->get_tx_sensor("lo_locked",chan).to_bool()){
                        //sleep for a short time in milliseconds
                        timeout_counter++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        if(timeout_counter>500){
                            std::stringstream ss;
                            ss<<"Cannot tune the TX frontend of channel "<<chan;
                            print_error(ss.str());
                            return false;
                        }
                    }
                    lo_locked = main_usrp->get_tx_sensor("lo_locked",chan);
                }
                tx = tx and main_usrp->get_tx_sensor("lo_locked",chan).to_bool();
            }
        }catch (uhd::lookup_error e){
            std::cout<<"None"<<std::endl;
            tx = true;
        }
    }
    BOOST_LOG_TRIVIAL(info) << "Tuning checked with results tx: "<< tx << " and rx: "<< rx ;
    return rx and tx;

}

void hardware_manager::set_streams(){

	BOOST_LOG_TRIVIAL(info) << "Presetting streams";

    //in this function config is an object representing the current paramenters.

    //if the stream configuration is different, reset the streams
    clear_streams();

    if(channel_num.size()!=1)channel_num.resize(1);

    if (config.A_TXRX.mode == RX and config.A_RX2.mode == RX){
        print_error("Currently only one receiver per front end is suppored");
        return;
    }

    if (config.A_TXRX.mode == TX and config.A_RX2.mode == TX){
        print_error("Currently only one transmitter per front end is suppored");
        return;
    }

	//declare unit to be used

	//front_end_code0 was used in a previous version. Kept for showing the scheme

    if(config.A_TXRX.mode == RX){
		BOOST_LOG_TRIVIAL(info) << "Config A_TXRX as RX";
		uhd::stream_args_t stream_args("fc32");
        front_end_code0 = 'A';
        channel_num[0] = 0;
        stream_args.channels = channel_num;
        if(not sw_loop)A_rx_stream = main_usrp->get_rx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";
    }else if(config.A_RX2.mode == RX){
		BOOST_LOG_TRIVIAL(info) << "Config A_RX2 as RX";
		uhd::stream_args_t stream_args("fc32");
        front_end_code0 = 'B';
        channel_num[0] = 0;
        stream_args.channels = channel_num;
        if(not sw_loop)A_rx_stream = main_usrp->get_rx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";
    }

    if(config.B_TXRX.mode == RX){
        BOOST_LOG_TRIVIAL(info) << "Config B_TXRX as RX";
		uhd::stream_args_t stream_args("fc32");

        front_end_code0 = 'C';
        channel_num[0] = 1;
        stream_args.channels = channel_num;
        if(not sw_loop)B_rx_stream = main_usrp->get_rx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";

    }else if(config.B_RX2.mode == RX){
        BOOST_LOG_TRIVIAL(info) << "Config B_RX2 as RX";
		uhd::stream_args_t stream_args("fc32");
        front_end_code0 = 'D';
        channel_num[0] = 1;
        stream_args.channels = channel_num;
        if(not sw_loop)B_rx_stream = main_usrp->get_rx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";
    }


    if(config.A_RX2.mode == TX or config.A_TXRX.mode == TX){
        BOOST_LOG_TRIVIAL(info) << "Config A_TXRX as TX";
		uhd::stream_args_t stream_args("fc32");
        channel_num[0] = 0;
        stream_args.channels = channel_num;
        if(not sw_loop)A_tx_stream = main_usrp->get_tx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";
    }

    if(config.B_RX2.mode == TX or config.B_TXRX.mode == TX){
		BOOST_LOG_TRIVIAL(info) << "Config B_TXRX as TX";
		uhd::stream_args_t stream_args("fc32");
        //declare unit to be used
        channel_num[0] = 1;
        stream_args.channels = channel_num;
        if(not sw_loop)B_tx_stream = main_usrp->get_tx_stream(stream_args);
		BOOST_LOG_TRIVIAL(info) << "Config done";
    }

	BOOST_LOG_TRIVIAL(info) << "Stream presetting done";
};

void hardware_manager::clear_streams(){
		BOOST_LOG_TRIVIAL(info) << "Resetting streams";
    if(A_rx_stream){
        A_rx_stream.reset();
        A_rx_stream = nullptr;
    }
    if(A_tx_stream){
        A_tx_stream.reset();
        A_tx_stream = nullptr;
    }
    if(B_rx_stream){
        B_rx_stream.reset();
        B_rx_stream = nullptr;
    }
    if(B_tx_stream){
        B_tx_stream.reset();
        B_tx_stream = nullptr;
    }
		BOOST_LOG_TRIVIAL(info) << "Streams reset";
}


std::string hardware_manager::apply_antenna_config(param *parameters, param *old_parameters, size_t chan){

	BOOST_LOG_TRIVIAL(info) << "Configuring antenna...";

    //handles the eventual output
    std::stringstream ss;

    //keep track of channel changing to dispay correct message.
    bool changed = false;

    if(parameters->mode!=OFF){

        if(old_parameters->mode == OFF or old_parameters->rate != parameters->rate){
            changed = true;

            //check if tx or rx
            if(not sw_loop){
                parameters->mode == RX ?
                    main_usrp->set_rx_rate(parameters->rate,chan):
                    main_usrp->set_tx_rate(parameters->rate,chan);
            }
            if(parameters->mode == RX){
                if(not sw_loop){
                    old_parameters->rate = main_usrp->get_rx_rate(chan);
                }else old_parameters->rate = parameters->rate;
                ss << boost::format("\tSetting RX Rate: %f Msps. ") % (parameters->rate / 1e6) << std::flush;
            }else{
                if(not sw_loop){
                    old_parameters->rate = main_usrp->get_tx_rate(chan);
                }else old_parameters->rate = parameters->rate;
                ss << boost::format("\tSetting TX Rate: %f Msps. ") % (parameters->rate / 1e6) << std::flush;
            }
            old_parameters->rate == parameters->rate?
                ss<<std::endl:
                ss<<boost::format("Effective value: %f Msps. ") % (old_parameters->rate / 1e6)<<std::endl;
            parameters->rate = old_parameters->rate;
        }
        try{
            if(old_parameters->mode == OFF or old_parameters->tone != parameters->tone or old_parameters->tuning_mode != parameters->tuning_mode){
                changed = true;


                if(parameters->mode == RX) {

                        if(not sw_loop){
                            main_usrp->get_rx_sensor("lo_locked",chan).to_bool();
                            if(not parameters->tuning_mode){

                                uhd::tune_request_t tune_request(parameters->tone);
                                tune_request.args = uhd::device_addr_t("mode_n=integer");
                                main_usrp->set_rx_freq(tune_request,chan);
                            }else{
                                uhd::tune_request_t tune_request(parameters->tone);
                                main_usrp->set_rx_freq(tune_request,chan);
                            }

                            old_parameters->tone = main_usrp->get_rx_freq(chan);
                        } else old_parameters->tone = parameters->tone;
                        old_parameters->tuning_mode = parameters->tuning_mode;
                        ss << boost::format("\tSetting RX central frequency: %f MHz. ") % (parameters->tone / 1e6);
                        if(parameters->tuning_mode){
                            ss<<" (fractional) ";
                        }else{
                            ss<<" (integer) ";
                        }
                        ss<< std::flush;


                }else{

                    if(not sw_loop){
                        main_usrp->get_tx_sensor("lo_locked",chan).to_bool();
                        if(not parameters->tuning_mode){

                            uhd::tune_request_t tune_request(parameters->tone);
                            tune_request.args = uhd::device_addr_t("mode_n=integer");
                            main_usrp->set_tx_freq(tune_request,chan);
                        }else{
                            uhd::tune_request_t tune_request(parameters->tone);
                            main_usrp->set_tx_freq(tune_request,chan);
                        }
                        old_parameters->tone = main_usrp->get_tx_freq(chan);
                    }else{
                        old_parameters->tone = parameters->tone;
                    }
                    old_parameters->tuning_mode = parameters->tuning_mode;

                    ss << boost::format("\tSetting TX central frequency: %f MHz. ") % (parameters->tone / 1e6);
                    if(parameters->tuning_mode){
                        ss<<" (fractional) ";
                    }else{
                        ss<<" (integer) ";
                    }
                    ss<< std::flush;
                }

                old_parameters->tone == parameters->tone?
                    ss<<std::endl:
                    ss<<boost::format("Effective value: %f MHz. ") % (old_parameters->tone / 1e6)<<std::endl;
                parameters->tone = old_parameters->tone;
            }
        }catch(uhd::lookup_error e){
            ss << boost::format("\tNo mixer detected\n");
        }

        if(old_parameters->mode == OFF or old_parameters->gain != parameters->gain){
            changed = true;

            //check if tx or rx
            if(not sw_loop){
                parameters->mode == RX ?
                    main_usrp->set_rx_gain(parameters->gain,chan):
                    main_usrp->set_tx_gain(parameters->gain,chan);

                if(parameters->mode == RX){
                    old_parameters->gain = main_usrp->get_rx_gain(chan);
                    ss << boost::format("\tSetting RX gain: %d dB. ") % (parameters->gain ) << std::flush;
                }else{
                    old_parameters->gain = main_usrp->get_tx_gain(chan);
                    ss << boost::format("\tSetting TX gain: %d dB. ") % (parameters->gain ) << std::flush;
                }
                old_parameters->gain == parameters->gain?
                    ss<<std::endl:
                    ss<<boost::format("Effective value: %d dB. ") % (old_parameters->gain )<<std::endl;
                parameters->gain = old_parameters->gain;
            }else old_parameters->gain = parameters->gain;
        }
        if(old_parameters->mode == OFF or old_parameters->bw != parameters->bw){
            changed = true;
            //check if tx or rx
            if(not sw_loop){
                parameters->mode == RX ?
                    main_usrp->set_rx_bandwidth(parameters->bw,chan):
                    main_usrp->set_tx_bandwidth(parameters->bw,chan);

                if(parameters->mode == RX){
                    old_parameters->bw = main_usrp->get_rx_bandwidth(chan);
                    ss << boost::format("\tSetting RX bandwidth: %f MHz. ") % (parameters->bw/ 1e6 ) << std::flush;
                }else{
                    old_parameters->bw = main_usrp->get_tx_bandwidth(chan);
                    ss << boost::format("\tSetting TX bandwidth: %f MHz. ") % (parameters->bw/ 1e6 ) << std::flush;
                }
                old_parameters->bw == parameters->bw?
                    ss<<std::endl:
                    ss<<boost::format("Effective value: %f MHz. ") % (old_parameters->gain/ 1e6 )<<std::endl;
                parameters->bw = old_parameters->bw;
            }else old_parameters->bw = parameters->bw;
        }
        if(old_parameters->mode == OFF or old_parameters->delay != parameters->delay){
            changed = true;
            old_parameters->delay = parameters->delay;
            ss << boost::format("\tSetting start streaming delay: %f msec. ") % (parameters->delay*1e3 ) << std::endl;
        }
        if((old_parameters->mode == OFF or old_parameters->burst_off != parameters->burst_off) and parameters->burst_off != 0){
            changed = true;
            old_parameters->burst_off = parameters->burst_off;
            ss << boost::format("\tSetting interval between bursts: %f msec. ") % (parameters->burst_off*1e3 ) << std::endl;
        }
        if((old_parameters->mode == OFF or old_parameters->burst_on != parameters->burst_on) and parameters->burst_on != 0){
            changed = true;
            old_parameters->burst_on = parameters->burst_on;
            ss << boost::format("\tSetting bursts duration: %f msec. ") % (parameters->burst_on*1e3 ) << std::endl;
        }
        if(old_parameters->mode == OFF or old_parameters->samples != parameters->samples){
            changed = true;
            old_parameters->samples = parameters->samples;
            ss << boost::format("\tSetting total samples to %f Ms") % (parameters->samples /1e6 ) << std::endl;
        }
        if(old_parameters->mode == OFF or old_parameters->buffer_len != parameters->buffer_len){
            changed = true;
            old_parameters->buffer_len = parameters->buffer_len;
            ss << boost::format("\tSetting buffer length to %f Ms") % (parameters->buffer_len /1e6 ) << std::endl;
        }
        if(old_parameters->mode == OFF or old_parameters->burst_off != parameters->burst_off){
            changed = true;
            old_parameters->burst_off = parameters->burst_off;
            if(old_parameters->burst_off == 0){
                ss << boost::format("\tSetting continuous acquisition mode (no bursts)")<< std::endl;
            }else{
                ss << boost::format("\tSetting samples between bursts to %f msec") % (parameters->burst_off *1e3 ) << std::endl;
            }
        }
        if(old_parameters->mode == OFF or old_parameters->burst_on != parameters->burst_on){
            changed = true;
            old_parameters->burst_on = parameters->burst_on;
            if(old_parameters->burst_on != 0){
                ss << boost::format("\tSetting bursts length to %f msec") % (parameters->burst_on *1e3 ) << std::endl;
            }
        }
        if(not changed) ss<<"\tHardware parameters were identical to last setup"<<std::endl;
    }
    //last thing to do
    old_parameters->mode = parameters->mode;

		BOOST_LOG_TRIVIAL(info) << "Antenna configured";

    return ss.str();

}

//check if there are more than 1 tx/rx channel
bool hardware_manager::check_double_txrx(ant_mode TXRX){
    int tx = 0;
    if(A_TXRX_chk == TXRX)tx++;
    if(A_RX2_chk == TXRX)tx++;
    if(B_RX2_chk == TXRX)tx++;
    if(B_TXRX_chk == TXRX)tx++;
    if(tx>1)return true;
    return false;
}


//check if the selected mode has to be tuned
bool hardware_manager::check_global_mode_presence(ant_mode mode, size_t chan){
    bool present = false;
    if(chan == 0)present = present or (config.A_TXRX.mode == mode);
    if(chan == 1)present = present or (config.B_TXRX.mode == mode);
    if(chan == 0)present = present or (config.A_RX2.mode == mode);
    if(chan == 1)present = present or (config.B_RX2.mode == mode);
    return present;
}

void hardware_manager::software_tx_thread(
    param *current_settings,                //some parameters are useful also in sw
    preallocator<float2>* memory,            //custom memory preallocator
    tx_queue* TX_queue,
    tx_queue* sw_loop_queue,
    char front_end
    ){
    std::stringstream thread_name;
    thread_name << "Software tx thread "<<front_end;
    set_this_thread_name(thread_name.str());
    BOOST_LOG_TRIVIAL(debug) << "Thread started";
    float2* tx_buffer;          //the buffer pointer
    if(front_end == 'A'){
        A_tx_thread_operation = true; //class variable to account for thread activity
    }else if(front_end == 'B'){
        B_tx_thread_operation = true;
    }else{
        print_error("Frontend code not recognized in software tx thread");
        return;
    } //class variable to account for thread activity
    bool active = true;         //local activity monitor
    size_t sent_samp = 0;     //total number of samples sent

    while(active and (sent_samp < current_settings->samples)){
        try{
            boost::this_thread::interruption_point();
            if(TX_queue->pop(tx_buffer)){

                float2* tx_buffer_copy = (float2*)malloc(sizeof(float2)*current_settings->buffer_len);

                sent_samp += current_settings->buffer_len;

                memcpy(tx_buffer_copy, tx_buffer, sizeof(float2)*current_settings->buffer_len);

                while(not sw_loop_queue->push(tx_buffer_copy))std::this_thread::sleep_for(std::chrono::microseconds(1));

                if(memory)memory->trash(tx_buffer);

            }else{
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }catch (boost::thread_interrupted &){
            active = false;

        }
    }
    if(front_end == 'A'){
        A_tx_thread_operation = false; //class variable to account for thread activity
    }else if(front_end == 'B'){
        B_tx_thread_operation = false;
    }
    BOOST_LOG_TRIVIAL(debug) << "Thread joined";
}



/*

#ifndef GET_CACHE_LINE_SIZE_H_INCLUDED
#define GET_CACHE_LINE_SIZE_H_INCLUDED

#include <stddef.h>
size_t cache_line_size();

#if defined(__gnu_linux__)

#include <stdio.h>
size_t cache_line_size() {
    FILE * p = 0;
    p = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
    unsigned int i = 0;
    if (p) {
        fscanf(p, "%d", &i);
        fclose(p);
    }
    return i;
}

#else
size_t cache_line_size() {
    return 0;
}
#endif

#endif

#include <emmintrin.h>

static inline void prefetch_range(void *addr, size_t len, size_t cache){
     char *addr_c = reinterpret_cast<char*>(addr);
     char *cp;
     char *end =  addr_c + len;

     for (cp = addr_c; cp < end; cp += cache*4){
         //__builtin_prefetch(cp);
         _mm_prefetch(cp, _MM_HINT_T1);
     }

}
inline float2* get_buffer_ready(tx_queue* __restrict__ TX_queue, size_t buffer_len_tx, size_t cache){
	float2* tx_buffer;
    while(not TX_queue->pop(tx_buffer))std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    //there is no need of the full buffer as the cache may be limited and the access pattern is very regular.
    prefetch_range(tx_buffer, buffer_len_tx, cache);
    return tx_buffer;
}
*/
void hardware_manager::single_tx_thread(
    param *current_settings,                //(managed internally to the class) user parameter to use for rx setting
    threading_condition* wait_condition,    //before joining wait for that condition
    tx_queue* TX_queue,                     //associated tx stream queue
    uhd::tx_streamer::sptr &tx_stream,       //stream to usrp
    preallocator<float2>* memory,            //custom memory preallocator
    char front_end
){

	std::stringstream thread_name;
    thread_name << "Hardware tx thread "<<front_end;
    set_this_thread_name(thread_name.str());

    uhd::set_thread_priority_safe(1.);
    bool active = true;
    size_t sent_samp = 0;       //total number of samples sent
    float2* tx_buffer;  //the buffer pointer
    //float2* tx_next_buffer;  //the buffer pointer next


    //SetThreadName(metadata_thread, "TX_metadata_thread");
    std::stringstream ss;
    ss<< "TX worker " << front_end;
    set_this_thread_name(ss.str());

    if(front_end == 'A'){
        A_tx_thread_operation = true; //class variable to account for thread activity
    }else if(front_end == 'B'){
        B_tx_thread_operation = true;
    }else{
        print_error("Frontend code not recognized in hardware tx thread");
        return;
    } //class variable to account for thread activity


    //double start_time = main_usrp->get_time_now().get_real_secs();

    uhd::tx_metadata_t metadata_tx;

    BOOST_LOG_TRIVIAL(debug) <<"Starting metadata thread";
		boost::thread* metadata_thread = nullptr;
		if(front_end == 'A'){
    	metadata_thread = new boost::thread(boost::bind(&hardware_manager::async_stream,this,tx_stream,front_end));
		}
    metadata_tx.start_of_burst = true;
    metadata_tx.end_of_burst = false;
    metadata_tx.has_time_spec  = true;
    metadata_tx.time_spec = uhd::time_spec_t(1.0+current_settings->delay);
		double timeout = 1.0+current_settings->delay + 0.1;
    //optimizations for tx loop
    size_t max_samples_tx = current_settings->samples;
    //double burst_off = current_settings->burst_off;
    size_t buffer_len_tx = current_settings->buffer_len;
    //size_t cache = cache_line_size();


    std::future<float2*> handle;
    //tx_next_buffer = get_buffer_ready(TX_queue, buffer_len_tx, cache);
    BOOST_LOG_TRIVIAL(info) <<"Starting main loop";
    while(active and (sent_samp < current_settings->samples)){
        try{

            boost::this_thread::interruption_point();

            if(sent_samp + buffer_len_tx >= max_samples_tx) metadata_tx.end_of_burst = true;

            //tx_buffer = tx_next_buffer;

            //handle = std::async(std::launch::async, get_buffer_ready, TX_queue, buffer_len_tx, cache);

            while(not TX_queue->pop(tx_buffer))std::this_thread::sleep_for(std::chrono::nanoseconds(500));

            sent_samp += tx_stream->send(tx_buffer, buffer_len_tx, metadata_tx, timeout);
						timeout = 0.1f;
            metadata_tx.start_of_burst = false;
            metadata_tx.has_time_spec = false;

			//tx_next_buffer = handle.get();

            if(memory)memory->trash(tx_buffer);

        }catch (boost::thread_interrupted &){
            active = false;
            BOOST_LOG_TRIVIAL(info) <<"Interrupt received";
        }

    }
    //clean the queue as it's le last consumer
    while(not TX_queue->empty()){
        TX_queue->pop(tx_buffer);
        if(memory)memory->trash(tx_buffer);
    }
    //something went wrong and the thread has interrupred
    if(not active and sent_samp < current_settings->samples){
        print_warning("TX thread was joined without transmitting the specified samples");
        std::cout<< "Missing "<< current_settings->samples - sent_samp<<" samples"<<std::endl;
        BOOST_LOG_TRIVIAL(info) <<"Thread was joined without transmitting "<< current_settings->samples - sent_samp<<" samples";
    }
		if(front_end == 'A'){
	    metadata_thread->interrupt();
	    metadata_thread->join();
	    delete metadata_thread;
	    metadata_thread = nullptr;
		}
    //set check the condition to false
    if(front_end == 'A'){
        A_tx_thread_operation = false; //class variable to account for thread activity
    }else if(front_end == 'B'){
        B_tx_thread_operation = false;
    }
    tx_stream.reset();

    BOOST_LOG_TRIVIAL(debug) << "Thread joined";
}

//ment to be in a thread. receive messages asyncronously on metadata
void hardware_manager::async_stream(uhd::tx_streamer::sptr &tx_stream, char fornt_end){
    bool parent_process_active = false;
    bool active = true;
    uhd::async_metadata_t async_md;
    int errors;
    while(active){
        try{

            boost::this_thread::interruption_point();

            if (fornt_end == 'A'){
                parent_process_active = A_tx_thread_operation;
            }else if (fornt_end == 'B'){
                parent_process_active = B_tx_thread_operation;
            }

            if(parent_process_active){
                errors = 0;
                if(tx_stream->recv_async_msg(async_md)){
                    errors = get_tx_error(&async_md,true);
                }
                if(errors>0 and parent_process_active){
                    if (fornt_end == 'A'){
                        A_tx_error_queue->push(1);
                    }else if (fornt_end == 'B'){
                        B_tx_error_queue->push(1);
                    }
                }

            }else{ active = false; }
        }catch (boost::thread_interrupted &e){ active = false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    }
    tx_stream.reset();
}

void hardware_manager::software_rx_thread(
    param *current_settings,
    preallocator<float2>* memory,
    rx_queue* Rx_queue,
    tx_queue* sw_loop_queue,
    char front_end
){
	std::stringstream thread_name;
    thread_name << "Software rx thread "<<front_end;
    set_this_thread_name(thread_name.str());
    BOOST_LOG_TRIVIAL(debug) << "Thread started";
		char front_end_c;
    if(front_end == 'A'){
        A_rx_thread_operation = true; //class variable to account for thread activity
				front_end_c = 'B';
    }else if(front_end == 'B'){
        B_rx_thread_operation = true;
				front_end_c = 'D';
    }else{
        print_error("Frontend code not recognized in software rx thread");
        return;
    } //class variable to account for thread activity
    bool active = true;         //control the interruption of the thread
    bool taken = true;
    RX_wrapper warapped_buffer;
    float2 *rx_buffer;
    float2 *rx_buffer_cpy;
    size_t acc_samp = 0;  //total number of samples received
    int counter = 0;
    while(active and (acc_samp < current_settings->samples)){

        try{

            boost::this_thread::interruption_point();

            if(taken)rx_buffer_cpy = memory->get();

            if(sw_loop_queue->pop(rx_buffer)){
                taken = true;
                counter++;
                memcpy(rx_buffer_cpy, rx_buffer, current_settings->buffer_len * sizeof(float2));
                free(rx_buffer);
                warapped_buffer.buffer = rx_buffer_cpy;
                warapped_buffer.packet_number = counter;
                warapped_buffer.length = current_settings->buffer_len;
                warapped_buffer.errors = 0;
                warapped_buffer.front_end_code = front_end_c;
                while(not Rx_queue->push(warapped_buffer))std::this_thread::sleep_for(std::chrono::microseconds(1));
                acc_samp += current_settings->buffer_len;
            }else{
                taken = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

            }

        }catch (boost::thread_interrupted &){ active = false;}

    }
    if(front_end == 'A'){
        A_rx_thread_operation = false; //class variable to account for thread activity
    }else if(front_end == 'B'){
        B_rx_thread_operation = false;
    }
    BOOST_LOG_TRIVIAL(debug) << "Thread joined";
}


void hardware_manager::single_rx_thread(
    param *current_settings,                //(managed internally) user parameter to use for rx setting
    rx_queue* Rx_queue,                     //(managed internally)queue to use for pushing
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory,            //custom memory preallocator
    uhd::rx_streamer::sptr &rx_stream ,     //the streamer to usrp
    char front_end                          //front end code for operation accountability

){

	std::stringstream thread_name;
  thread_name << "Hardware rx thread  "<<front_end;
  set_this_thread_name(thread_name.str());
	BOOST_LOG_TRIVIAL(debug) << "Thread started";
	char front_end_c;
    if(front_end == 'A'){
        A_rx_thread_operation = true; //class variable to account for thread activity
				front_end_c = 'B';
    }else if(front_end == 'B'){
        B_rx_thread_operation = true;
				front_end_c = 'D';
    }else{
        print_error("Frontend code not recognized in hardware rx thread");
        return;
    }
    if (not uhd::set_thread_priority_safe(+1)){
        std::stringstream ss;
        ss<<"Cannot set thread priority from tx thread"<<front_end;
        print_warning(ss.str());
    }
    bool active = true;         //control the interruption of the thread

    float2* rx_buffer;      //pointer to the receiver buffer
    size_t num_rx_samps = 0;   //number of samples received in a single loop
    size_t acc_samp = 0;  //total number of samples received
    size_t frag_count = 0;     //fragmentation count used for very long buffer
    float timeout = 0.1f;   //timeout in seconds(will be used to sync the recv calls)
    size_t counter = 0;        //internal packet number
    size_t errors = 0;         //error counter (per loop)
    size_t samples_remaining;  //internal loop samples counter
    size_t push_counter;       //number of pushing attempts
    size_t push_timer = 1;     //interval between queue pushing attempts

    //packet wrapping structure
    RX_wrapper warapped_buffer;
    warapped_buffer.usrp_number = this_usrp_number;



    //setting the start metadata
    uhd::rx_metadata_t metadata_rx;
    //metadata_rx.has_time_spec = true;                               // in this application the time specification is always set
    //metadata_rx.time_spec = uhd::time_spec_t(1.0 + current_settings->delay);               //set the transmission delay
    //metadata_rx.start_of_burst = true;

    //setting the stream command (isn't it redundant with metadata?)
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);

    //change the mode properly
    if(current_settings->burst_off == 0){
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS ;
        stream_cmd.num_samps = 0;
    }else{
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_MORE;
        stream_cmd.num_samps = current_settings->buffer_len;
    }
    stream_cmd.stream_now = current_settings->delay == 0 ? true:false;

    stream_cmd.time_spec = uhd::time_spec_t(1.0+current_settings->delay);

    //if the number of samples to receive is smaller than the buffer the first packet is also the last one
    metadata_rx.end_of_burst = current_settings->samples <= current_settings->buffer_len ? true : false;

    //needed to change metadata only once
    bool first_packet = true;

    //needed to download the error from tx queue
    bool tmp;

    //issue the stream command (ignoring the code above @todo)
    stream_cmd.stream_now = false;
    stream_cmd.num_samps = current_settings->buffer_len;
    stream_cmd.time_spec = uhd::time_spec_t(1.0+current_settings->delay);
    timeout = 1.0+current_settings->delay;
    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS ;
    rx_stream->issue_stream_cmd(stream_cmd);

		// Just setting the stop command for later
		uhd::stream_cmd_t stop_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    stop_cmd.stream_now = false;

    uhd::set_thread_priority_safe(+1);
    while(active and acc_samp < current_settings->samples){
        //std::cout<<"samples: "<<acc_samp<<"/"<< current_settings->samples<<std::endl;
        try{

            boost::this_thread::interruption_point();

            //get a new buffer
            rx_buffer = memory->get();

            //reset/increment counters
            num_rx_samps = 0;
            frag_count = 0;
            errors = 0;
            counter++;

            //issue an other command for the burst mode
            /*
            if(stream_cmd.stream_mode == uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_MORE and not first_packet){
                stream_cmd.stream_now = false;
                stream_cmd.time_spec = uhd::time_spec_t(current_settings->burst_off);
                rx_stream->issue_stream_cmd(stream_cmd);
                std::cout<<"stream command is: "<<"STREAM_MODE_NUM_SAMPS_AND_MORE"<<std::endl;
            }
            */

            //this hsould be only one cycle however there are cases in which multiple recv calls are needed
            while(num_rx_samps<current_settings->buffer_len){

                //how many samples have to be received yet
                samples_remaining = std::min(current_settings->buffer_len,current_settings->buffer_len-num_rx_samps);

                //how long to wait for new samples //TODO not adapting the timeout can produce O
                //timeout = std::max((float)samples_remaining/(float)current_settings->rate,0.2f);

                //if(first_packet)std::this_thread::sleep_for(std::chrono::nanoseconds(size_t(1.e9*current_settings->delay)));

                //receive command
                num_rx_samps += rx_stream->recv(rx_buffer + num_rx_samps, samples_remaining, metadata_rx,timeout);//,0.1f
                timeout = 0.01f;
                //interpret errors
                if(get_rx_errors(&metadata_rx, true)>0)errors++;

                //get errors from tx thread if present
                if(front_end == 'A'){
                    while(A_tx_error_queue->pop(tmp) and A_tx_thread_operation)errors++;
                }else if(front_end == 'B'){
                    while(B_tx_error_queue->pop(tmp) and B_tx_thread_operation)errors++;
                }

                //change metadata for continuous streaming or burst mode
                if(first_packet){
                    metadata_rx.start_of_burst = current_settings->samples <= current_settings->buffer_len ? true : false;
                    metadata_rx.has_time_spec = current_settings->burst_off == 0? false:true;
                    metadata_rx.time_spec = uhd::time_spec_t(current_settings->burst_off);
                    first_packet = false;
                }
                if(++frag_count>1){
                    std::cout<< "F" << frag_count<<std::flush;
                }
                //fragmentation handling (intended as: the USRP is not keeping up with the host thread)
                if(++frag_count>4){
                    std::stringstream ss;
                    ss<<"RX Fragmentation too high: "<< frag_count <<" calls to recv to reach "<<num_rx_samps<<" /"<< current_settings->buffer_len<<" samples.";
                    print_warning(ss.str());
                    if(frag_count>8){
                        print_error("RX thread got stuck: USRP is not streaming any sample.");
                        active = false;
                        //exit this loop
                        num_rx_samps = current_settings->buffer_len;
                    }
                }
            }


            //update total number of accumulated samples
            acc_samp += num_rx_samps;

            //update metadata for last packet
            if(current_settings->samples < acc_samp + current_settings->buffer_len) metadata_rx.end_of_burst = true;

            //wrap the buffer

            warapped_buffer.buffer = rx_buffer;
            warapped_buffer.packet_number = counter;
            warapped_buffer.length = num_rx_samps;
            warapped_buffer.errors = errors;
            warapped_buffer.front_end_code = front_end_c;

            //insist in pushing the buffer
            push_counter = 0;
            while(not Rx_queue->push(warapped_buffer)){
                std::this_thread::sleep_for(std::chrono::microseconds(push_timer));
                if(push_counter>1)print_warning("RX queue is experencing some delay. This may cause troubles in real time acquisition.");
                push_counter++;
                if(push_timer * push_counter > 1e3*current_settings->buffer_len/current_settings->rate){
                    print_warning("RX queue is experencing some delay. This may cause troubles in real time acquisition.");
                }
            }

        }catch (boost::thread_interrupted &){ active = false; }
    }

    rx_stream->issue_stream_cmd(stop_cmd);

    flush_rx_streamer(rx_stream); // flush the cache
    rx_stream.reset();

    //something went wrong and the thread has interrupred
    if(not active){
        print_warning("RX thread was taken down without receiving the specified samples");
    }

		// Class atomic variable to account for thread activity:
    // Result in the check function to return to false.
    if(front_end == 'A'){
        A_rx_thread_operation = false;
    }else if(front_end == 'B'){
        B_rx_thread_operation = false;
    }

    BOOST_LOG_TRIVIAL(debug) << "Thread joined";
}

void hardware_manager::flush_rx_streamer(uhd::rx_streamer::sptr &rx_streamer) {
   constexpr double timeout { 0.010 }; // 10ms
   constexpr size_t size { 1048576 };
   static float2 dummy_buffer[size];
   static uhd::rx_metadata_t dummy_meta { };
   while (rx_streamer->recv(dummy_buffer, size, dummy_meta, timeout)) {}
}
