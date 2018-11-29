#include "USRP_hardware_manager.hpp"


//the initializer of the class can be used to select which usrp is controlled by the class
//Default call suppose only one USRP is connected
hardware_manager::hardware_manager(server_settings* settings, bool sw_loop_init, int usrp_number){

    //software loop mode exclude the hardware
    sw_loop = sw_loop_init;
    
    //in any case a gpu is necessary
    cudaSetDevice(settings->GPU_device_index);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, settings->GPU_device_index);
    
    
    if(not sw_loop){
    
        this_usrp_number = usrp_number;
        hint["type"] = "x300";
        
        //recursively look for usrps
        dev_addrs = uhd::device::find(hint);
        std::cout<<"Looking for USRP x300 device number "<< usrp_number << " .." <<std::flush;
        while(dev_addrs.size()< usrp_number + 1){
            
            dev_addrs = uhd::device::find(hint);
            std::cout<<"."<<std::flush;
            usleep(1e6);
        }
        
        
        std::cout<<"Device found and assigned to GPU "<< props.name <<" ("<< settings->GPU_device_index <<")"<<std::endl;
        
        //assign desired address
        main_usrp = uhd::usrp::multi_usrp::make(dev_addrs[usrp_number]);
        
        //set the clock reference
        main_usrp->set_clock_source(settings->clock_reference);
    
    }else{
        sw_loop_queue = new tx_queue(SW_LOOP_QUEUE_LENGTH);
    }
    
    //initialize port connection check variables
    A_TXRX_chk = OFF;
    B_RX2_chk = OFF;
    B_TXRX_chk = OFF;
    A_RX2_chk = OFF;
    
    //set the thread state
    rx_thread_operation = false;
    tx_thread_operation = false;
    
    //settling time for fpga register initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    
    //initialize transmission queues
    RX_queue = new rx_queue(RX_QUEUE_LENGTH);
    TX_queue = new tx_queue(TX_QUEUE_LENGTH);
    tx_error_queue = new error_queue(ERROR_QUEUE_LENGTH);

}

//this function should be used to set the USRP device with user parameters
//TODO catch exceptions and return a boolean
bool hardware_manager::preset_usrp(usrp_param* requested_config){
    apply(requested_config);
    set_streams();
    if(not sw_loop){
        check_tuning();
    }
    return true;
    
}

bool hardware_manager::check_rx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if(verbose)print_debug("RX thread status: ",rx_thread_operation);
    return rx_thread_operation;
}

bool hardware_manager::check_tx_status(bool verbose){
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if(verbose)print_debug("TX thread status: ",tx_thread_operation);
    return tx_thread_operation;
}

void hardware_manager::start_tx(
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory    //if the thread is transmitting a buffer that requires dynamical allocation than a pointer to  custo memory manager class has to be passed

){

    if(not tx_thread_operation){
        //pointer to rx parameters
        param *current_settings;
        
        //single and double threads are different
        if(not check_double_txrx(TX)){
        
            //assign the correct struct reference to the parameter input (assuming only one is active as RX)
            if(config.A_TXRX.mode == TX) current_settings = &config.A_TXRX;
            if(config.B_TXRX.mode == TX) current_settings = &config.B_TXRX;
            if(config.A_RX2.mode == TX) current_settings = &config.A_RX2;
            if(config.B_RX2.mode == TX) current_settings = &config.B_RX2;

            //start the thread
            if(not sw_loop){
            
                tx_thread = new boost::thread(boost::bind(&hardware_manager::single_tx_thread,this,
                    current_settings,
                    wait_condition,
                    memory));
            }else{
            
                tx_thread = new boost::thread(boost::bind(&hardware_manager::software_tx_thread,this,current_settings,memory));
            
            }
                
        }else{
            print_error("Double TX not implemented yet");
        }
    }else{
        std::stringstream ss;
        ss << "Cannot start TX thread, a tx thread associated with USRP "<< this_usrp_number <<" is already running";
        print_error(ss.str());
    }
}
void hardware_manager::start_rx(
    int buffer_len,                         //length of the buffer. MUST be the same of the preallocator initialization
    long int num_samples,                   //how many sample to receive
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory            //custom memory preallocator

    ){
    if(not rx_thread_operation){
        //pointer to rx parameters
        param *current_settings;
        
        //single and double threads are different
        if(not check_double_txrx(RX)){
            //assign the correct struct reference to the parameter input (assuming only one is active as RX)
            if(config.A_TXRX.mode == RX) current_settings = &config.A_TXRX; 
            if(config.B_TXRX.mode == RX) current_settings = &config.B_TXRX; 
            if(config.A_RX2.mode == RX) current_settings = &config.A_RX2; 
            if(config.B_RX2.mode == RX) current_settings = &config.B_RX2;

            //start the thread
            if(not sw_loop){
                rx_thread = new boost::thread(boost::bind(&hardware_manager::single_rx_thread,this,
                    current_settings,
                    RX_queue,
                    wait_condition,
                    memory));
            }else{
            
                rx_thread = new boost::thread(boost::bind(&hardware_manager::software_rx_thread,this,current_settings,memory,RX_queue));
                
            }
                
        }else{
            print_error("Double RX not implemented yet");
        }
    }else{
        std::stringstream ss;
        ss << "Cannot start RX thread, a rx threead associated with USRP "<< this_usrp_number <<" is already running";
        print_error(ss.str());
    }
}
void hardware_manager::close_tx(){

    //if(tx_thread_operation){
    
        tx_thread->interrupt();
        tx_thread->join();
        //tx_thread_operation = false;
        
    //}

}

void hardware_manager::close_rx(){

    //if(rx_thread_operation){
    
        rx_thread->interrupt();
        rx_thread->join();
        //rx_thread_operation = false;

    //}
}

int hardware_manager::clean_tx_queue(preallocator<float2>* memory){

    //temporary wrapper
    float2* buffer;
    
    //counter. Expected to be 0
    int counter = 0;
    
    //cannot execute when the rx thread is going
    if(not tx_thread_operation){
        while(not TX_queue->empty() or TX_queue->pop(buffer)){
        
            memory->trash(buffer);
            counter ++;
        }
    }
    if(counter > 0){
        std::stringstream ss;
        ss << "TX queue cleaned of "<< counter <<"buffer(s)";
        print_warning(ss.str());
    } 
    return counter;
}

int hardware_manager::clean_rx_queue(preallocator<float2>* memory){

    //temporary wrapper
    RX_wrapper warapped_buffer;
    
    //counter. Expected to be 0
    int counter = 0;
    
    //cannot execute when the rx thread is going
    if(not rx_thread_operation){
        while(not RX_queue->empty() or RX_queue->pop(warapped_buffer)){
            memory->trash(warapped_buffer.buffer);
            counter ++;
        }
    }
    if(counter > 0){
        std::stringstream ss;
        ss << "RX queue cleaned of "<< counter <<"buffer(s)";
        print_warning(ss.str());
    } 
    return counter;
}


void hardware_manager::apply(usrp_param* requested_config){

    //transfer the usrp index to the setting parameters
    requested_config->usrp_number = this_usrp_number;

    //stack of messages
    std::stringstream ss;
    ss<<std::endl;
    
    //set the subdevice specification IS IT REDUNDANT WITH CHANNEL NUMBER?
    /*
    std::stringstream subdev_tx; 
    if (requested_config->A_TXRX.mode == TX or requested_config->A_RX2.mode == TX)subdev_tx<<"A:0 ";
    if (requested_config->B_TXRX.mode == TX or requested_config->B_RX2.mode == TX)subdev_tx<<"B:0" ;
    if(subdev_tx.str().compare("") != 0) main_usrp->set_tx_subdev_spec(subdev_tx.str());
    
    std::stringstream subdev_rx; 
    if (requested_config->A_TXRX.mode == RX or requested_config->A_RX2.mode == RX)subdev_tx<<"A:0 ";
    if (requested_config->B_TXRX.mode == RX or requested_config->B_RX2.mode == RX)subdev_tx<<"B:0" ;
    if(subdev_rx.str().compare("") != 0) main_usrp->set_rx_subdev_spec(subdev_rx.str());
    */
    
    
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

}

bool hardware_manager::check_tuning(){

    //both rx and tx must be locked if selected
    bool rx = true;
    bool tx = true;
    
    for(size_t chan = 0; chan<2; chan++){
        //check for RX locking
        std::vector<std::string>  rx_sensor_names;
        rx_sensor_names = main_usrp->get_rx_sensor_names(chan);
        
        //check only if there is a channel associated with RX.
        if(check_global_mode_presence(RX,chan)){
            std::cout<<"Checking RX frontend tuning... "<<std::flush;
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
                        ss<<"Cannot tune the RX frontend of channel "<<chan?"A":"B";
                        print_error(ss.str());
                        return false;
                    }
                }
                lo_locked = main_usrp->get_rx_sensor("lo_locked",chan);
            }
            rx = rx and main_usrp->get_rx_sensor("lo_locked",chan).to_bool();
            std::cout<<"Done!"<<std::endl;
        }
        
        //check for TX locking
        std::vector<std::string>  tx_sensor_names;
        tx_sensor_names = main_usrp->get_tx_sensor_names(0);
        
        //check only if there is a channel associated with TX.
        if(check_global_mode_presence(TX,chan)){
            std::cout<<"Checking TX frontend tuning... "<<std::flush;
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
                        ss<<"Cannot tune the TX frontend of channel "<<chan?"A":"B";
                        print_error(ss.str());
                        return false;
                    }
                }
                lo_locked = main_usrp->get_tx_sensor("lo_locked",chan);
            }
            tx = tx and main_usrp->get_tx_sensor("lo_locked",chan).to_bool();
            std::cout<<"Done!"<<std::endl;
        }
    }    
    return rx and tx;

}

void hardware_manager::set_streams(){


    //declare unit to be used
    uhd::stream_args_t stream_args("fc32");
    
    //check if the stream configuration is different
    if(A_TXRX_chk != config.A_TXRX.mode or
       A_RX2_chk != config.A_RX2.mode or
       B_RX2_chk != config.B_RX2.mode or
       B_TXRX_chk != config.B_TXRX.mode
       ){
       
        //if the stream configuration is different, reset the streams
        clear_streams();
        
        A_TXRX_chk = config.A_TXRX.mode;
        A_RX2_chk = config.A_RX2.mode;
        B_RX2_chk = config.B_RX2.mode;
        B_TXRX_chk = config.B_TXRX.mode;
        
        //check if the rx and tx streaming is single or double
        bool rx_duble_set = check_double_txrx(RX);
        bool tx_duble_set = check_double_txrx(TX);
        
        if(not rx_duble_set){
            //set the A frontend
           
            if(config.A_TXRX.mode == RX or config.A_RX2.mode == RX){
                front_end_code0 = (config.A_TXRX.mode == RX)?'A':'B';
                channel_num.resize(1);
                channel_num[0] = 0;
                stream_args.channels = channel_num;
                if(not sw_loop)rx_stream = main_usrp->get_rx_stream(stream_args);
            }
            if(config.B_TXRX.mode == RX or config.B_RX2.mode == RX){
                front_end_code0 = (config.B_TXRX.mode == RX)?'C':'D';
                channel_num.resize(1);
                channel_num[0] = 1;
                stream_args.channels = channel_num;
                if(not sw_loop)rx_stream = main_usrp->get_rx_stream(stream_args);
            }
        }else{
            channel_num.resize(2);
            channel_num[0] = 0;
            channel_num[1] = 1;
            stream_args.channels = channel_num;
            if(not sw_loop)rx_stream = main_usrp->get_rx_stream(stream_args);
            //antenna code to be invented here
        }
        
        if(not tx_duble_set){

            if(config.A_TXRX.mode == TX or config.A_RX2.mode == TX){
                channel_num.resize(1);
                channel_num[0] = 0;
                stream_args.channels = {0};//channel_num;
                if(not sw_loop)tx_stream = main_usrp->get_tx_stream(stream_args);
            }
            if(config.B_TXRX.mode == TX or config.B_RX2.mode == TX){
                channel_num.resize(1);
                channel_num[0] = 1;
                stream_args.channels = {1};//channel_num;
                if(not sw_loop)tx_stream = main_usrp->get_tx_stream(stream_args);
            }
        }else{
            channel_num.resize(2);
            channel_num[0] = 0;
            channel_num[1] = 1;
            stream_args.channels = {0,1};//channel_num;
            if(not sw_loop)tx_stream = main_usrp->get_tx_stream(stream_args);
        }
    }

};

void hardware_manager::clear_streams(){
    rx_stream = NULL;
    tx_stream = NULL;
}


std::string hardware_manager::apply_antenna_config(param *parameters, param *old_parameters, size_t chan){

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
        
        if(old_parameters->mode == OFF or old_parameters->tone != parameters->tone or old_parameters->tuning_mode != parameters->tuning_mode){
            changed = true;
            
            if(parameters->mode == RX) {
                if(not sw_loop){
                    if(not parameters->tuning_mode){
                        
                        uhd::tune_request_t tune_request(parameters->tone,0); 
                        tune_request.args = uhd::device_addr_t("mode_n=integer");
                        main_usrp->set_rx_freq(tune_request,chan);
                    }else{
                        uhd::tune_request_t tune_request(parameters->tone,0); 
                        main_usrp->set_rx_freq(tune_request,chan);
                        //main_usrp->set_rx_freq(parameters->tone,chan);
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
                    if(not parameters->tuning_mode){
                        
                        uhd::tune_request_t tune_request(parameters->tone,0); 
                        tune_request.args = uhd::device_addr_t("mode_n=integer");
                        main_usrp->set_tx_freq(tune_request,chan);
                    }else{
                        uhd::tune_request_t tune_request(parameters->tone,0); 
                        main_usrp->set_tx_freq(tune_request,chan);
                        //main_usrp->set_tx_freq(parameters->tone,chan);
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
    preallocator<float2>* memory            //custom memory preallocator
    ){
    float2* tx_buffer;          //the buffer pointer
    tx_thread_operation = true; //class variable to account for thread activity
    bool active = true;         //local activity monitor
    long int sent_samp = 0;     //total number of samples sent
    
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
    tx_thread_operation = false;
}

void hardware_manager::single_tx_thread(
    param *current_settings,                //(managed internally to the class) user parameter to use for rx setting
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory            //custom memory preallocator
){
    

    //the wait condition is implemented so to wait this thread to transmit the given number of samples
    //wait_condition->rearm();
    
    //needed to change metadata only once
    bool first_packet = true;
    bool active = true;
    bool started = false; //Only needed for warning purposes
    size_t sent_samp = 0;       //total number of samples sent
    float2* tx_buffer;  //the buffer pointer
    
    boost::thread* metadata_thread = new boost::thread(boost::bind(&hardware_manager::async_stream,this));
    
    //float timeout = 0.1f;   //timeout in seconds(will be used to sync the recv calls)
    tx_thread_operation = true; //class variable to account for thread activity
    
    uhd::tx_metadata_t metadata_tx;
    
    metadata_tx.start_of_burst = true;
    metadata_tx.has_time_spec  = current_settings->delay == 0 ? false:true;
    metadata_tx.time_spec = uhd::time_spec_t(current_settings->delay);
    
    //if the number of samples to receive is smaller than the buffer the first packet is also the last one
    //metadata_tx.end_of_burst = current_settings->samples <= current_settings->buffer_len ? true : false;
    metadata_tx.end_of_burst = current_settings->burst_off!=0?true:false;;
    //boost::chrono::high_resolution_clock::time_point ti ;
    //boost::chrono::high_resolution_clock::time_point tf ;
    
    sync_time(); //sync to next pps + delay
    
    while(active and (sent_samp < current_settings->samples)){
        try{
            boost::this_thread::interruption_point();
            if(TX_queue->pop(tx_buffer)){

                if(sent_samp + current_settings->buffer_len >= current_settings->samples)metadata_tx.end_of_burst   = (bool)true;
                
                if((current_settings->burst_off!=0) and (not first_packet)){
                    metadata_tx.end_of_burst   = (bool)true;
                    metadata_tx.start_of_burst = (bool)true;
                    metadata_tx.has_time_spec = (bool)true;
                    //timeout = 0.1f + current_settings->burst_off;
                    metadata_tx.time_spec = main_usrp->get_time_now() + uhd::time_spec_t(current_settings->burst_off);
                }
                
                
                tx_stream->send(tx_buffer, current_settings->buffer_len, metadata_tx,0.3f);//timeout

                
                sent_samp += current_settings->buffer_len;
                
                if(current_settings->burst_off!=0){
                    first_packet = false;

                }else if(first_packet){
                    started = true;
                    metadata_tx.start_of_burst = false;
                    metadata_tx.has_time_spec = false;
                    first_packet = false;
                    //timeout = 0.1f;

                }
                
                if(memory)memory->trash(tx_buffer);

            }else{
                std::this_thread::sleep_for(std::chrono::microseconds(1));

                if(active and started){
                    print_error("TX queue is empty, cannot transmit buffer!");
                }
                //should I push an error?
            }
        }catch (boost::thread_interrupted &){
            active = false;
        }

    }
    //clean the queue as it's le last consumer
    while(not TX_queue->empty()){
        TX_queue->pop(tx_buffer);
        if(memory)memory->trash(tx_buffer);
    }
    //something went wrong and the thread has interrupred
    if(not active){
        print_warning("TX thread was taken down without transmitting the specified samples");
    }
    metadata_thread->interrupt();
    metadata_thread->join();
    
    //set check the condition to false
    tx_thread_operation = false;
    
    //wait_condition->release();
}

//ment to be in a thread. receive messages asyncronously on metadata
void hardware_manager::async_stream(){
    bool active = true;
    uhd::async_metadata_t async_md;
    int errors;
    while(active){
        try{
            boost::this_thread::interruption_point();
            if(tx_thread_operation){
                errors = 0;
                if(tx_stream->recv_async_msg(async_md)){
                    errors = get_tx_error(&async_md,true);
                }
                if(errors>0 and tx_thread_operation){
                    tx_error_queue->push(1);
                }
                
            }else{ active = false; }
        }catch (boost::thread_interrupted &e){ active = false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
    }
}

void hardware_manager::software_rx_thread(
    param *current_settings,
    preallocator<float2>* memory,
    rx_queue* Rx_queue
){
    
    rx_thread_operation = true; //class variable to account for thread activity
    bool active = true;         //control the interruption of the thread
    bool taken = true;
    RX_wrapper warapped_buffer;
    float2 *rx_buffer;
    float2 *rx_buffer_cpy;
    long int acc_samp = 0;  //total number of samples received
    int counter = 0;
    //int ll = 0;
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
                warapped_buffer.front_end_code = front_end_code0;
                while(not Rx_queue->push(warapped_buffer))std::this_thread::sleep_for(std::chrono::microseconds(1));
                acc_samp += current_settings->buffer_len;
                //print_debug("swtx inter: ",ll);
                //ll++;
            }else{
                taken = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                
            }

        }catch (boost::thread_interrupted &){ active = false;} 
        
    }
    rx_thread_operation = false;
}
    

void hardware_manager::single_rx_thread(
    param *current_settings,                //(managed internally) user parameter to use for rx setting

    rx_queue* Rx_queue,                     //(managed internally)queue to use for pushing
    threading_condition* wait_condition,    //before joining wait for that condition
    preallocator<float2>* memory            //custom memory preallocator
    
){
    rx_thread_operation = true; //class variable to account for thread activity
    bool active = true;         //control the interruption of the thread

    float2* rx_buffer;      //pointer to the receiver buffer
    size_t num_rx_samps = 0;   //number of samples received in a single loop
    size_t acc_samp = 0;  //total number of samples received
    size_t frag_count = 0;     //fragmentation count used for very long buffer
    //float timeout = 0.1f;   //timeout in seconds(will be used to sync the recv calls)
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
    metadata_rx.has_time_spec = true;                               // in this application the time specification is always set
    metadata_rx.time_spec = uhd::time_spec_t(current_settings->delay);               //set the transmission delay
    metadata_rx.start_of_burst = true;
    
    //setting the stream command (isn't it redundant with metadata?)
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    
    //change the mode properly
    if(current_settings->burst_off == 0){
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS ;
        stream_cmd.num_samps = 0;
        std::cout<<"stream command is: "<<"STREAM_MODE_START_CONTINUOUS"<<std::endl;
    }else{
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_MORE;
        stream_cmd.num_samps = current_settings->buffer_len;
        std::cout<<"stream command is: "<<"STREAM_MODE_NUM_SAMPS_AND_MORE"<<std::endl;
    }
    stream_cmd.stream_now = current_settings->delay == 0 ? true:false;

    stream_cmd.time_spec = uhd::time_spec_t(current_settings->delay);
    
    //if the number of samples to receive is smaller than the buffer the first packet is also the last one
    metadata_rx.end_of_burst = current_settings->samples <= current_settings->buffer_len ? true : false; 
    
    //needed to change metadata only once
    bool first_packet = true;
    
    //needed to download the error from tx queue
    bool tmp;
    
    //main_usrp->set_time_now(0.);
    sync_time();
    
    //issue the stream command
    rx_stream->issue_stream_cmd(stream_cmd);
    
    //sync_time(current_settings->delay);
    //std::this_thread::sleep_for(std::chrono::microseconds(int(1.e6*current_settings->delay)));
    //main thread loop



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
            if(stream_cmd.stream_mode == uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_MORE and not first_packet){
                stream_cmd.stream_now = false;
                stream_cmd.time_spec = uhd::time_spec_t(current_settings->burst_off);
                rx_stream->issue_stream_cmd(stream_cmd);
                std::cout<<"stream command is: "<<"STREAM_MODE_NUM_SAMPS_AND_MORE I should not be here"<<std::endl;
            }
            
            //this hsould be only one cycle however there are cases in which multiple recv calls are needed
            while(num_rx_samps<current_settings->buffer_len){
                
                //how many samples have to be received yet
                samples_remaining = std::min(current_settings->buffer_len,current_settings->buffer_len-num_rx_samps);
                
                //how long to wait for new samples //TODO not adapting the timeout can produce O
                //timeout = std::max((float)samples_remaining/(float)current_settings->rate,0.2f);

                if(first_packet)std::this_thread::sleep_for(std::chrono::nanoseconds(size_t(1.e9*current_settings->delay)));

                //receive command
                num_rx_samps += rx_stream->recv(rx_buffer + num_rx_samps, samples_remaining, metadata_rx);//,0.1f
                
                //interpret errors
                if(get_rx_errors(&metadata_rx, true)>0)errors++;
                
                //get errors from tx thread if present
                while(tx_error_queue->pop(tmp) and tx_thread_operation)errors++;

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
            warapped_buffer.front_end_code = front_end_code0;
            
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

    //issue a stop command if there was a continuous streaming
    //if(stream_cmd.stream_mode == uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS){
        uhd::stream_cmd_t stop_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
        stop_cmd.stream_now = false;
        rx_stream->issue_stream_cmd(stop_cmd);
    //}
    
    //something went wrong and the thread has interrupred
    if(not active){
        print_warning("RX thread was taken down without receiving the specified samples");
    }
    
    //set the check condition to false
    rx_thread_operation = false;
    
    //wait_condition->wait();
}


//used to sync TX and rRX streaming time
void hardware_manager::sync_time(){
    
    main_usrp->set_time_next_pps(uhd::time_spec_t(0.));
    double sec_to_next_pps = 1. - (main_usrp->get_time_now().get_real_secs() - main_usrp->get_time_last_pps().get_real_secs());
    std::this_thread::sleep_for(std::chrono::microseconds(int(1.e6*sec_to_next_pps)));
}
