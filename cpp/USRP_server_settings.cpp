#include "USRP_server_settings.hpp"

int TCP_SYNC_PORT = 61360;
int TCP_ASYNC_PORT = 22001;

std::string device_arguments = "noarg";

std::string w_type_to_str(w_type enumerator){
    std::string comp_string;
    comp_string = "UNINIT";
    switch(enumerator){
        case(TONES):
            comp_string =  "TONES";
            break;
        case(CHIRP):
            comp_string = "CHIRP";
            break;
        case(NOISE):
            comp_string = "NOISE";
            break;
        case(RAMP):
            comp_string = "RAMP";
            break;
        case(NODSP):
            comp_string = "NODSP";
            break;
        case(SWONLY):
            comp_string = "SWONLY";
            break;
        case(DIRECT):
            comp_string = "DIRECT";
            break;
    }
    return comp_string;
}

w_type string_to_w_type(std::string input_string){
    w_type conv = NODSP;
    if(input_string.compare("NODSP") == 0)conv = NODSP;

    if(input_string.compare("CHIRP") == 0)conv = CHIRP;

    if(input_string.compare("NOISE") == 0)conv = NOISE;

    if(input_string.compare("TONES") == 0)conv = TONES;

    if(input_string.compare("SWONLY") == 0)conv = SWONLY;

    if(input_string.compare("DIRECT") == 0)conv = DIRECT;

    return conv;

}

std::vector<w_type> string_to_w_type_vector(std::vector<std::string> string_vector){
    std::vector<w_type> res(string_vector.size());
    for(size_t i = 0; i <string_vector.size(); i++ ){
        res[i] = string_to_w_type(string_vector[i]);
    }
    return res;
}
//state of the USRP antenna
std::string ant_mode_to_str(ant_mode enumerator){
    std::string comp_string;
    comp_string = "UNINIT";
    switch(enumerator){
        case(TX):
            comp_string =  "TX";
            break;
        case(RX):
            comp_string = "RX";
            break;
        case(OFF):
            comp_string = "OFF";
            break;
    }
    return comp_string;
}

ant_mode ant_mode_from_string(std::string str){
    if (not str.compare("OFF")) return (ant_mode)OFF;
    if (not str.compare("RX")) return (ant_mode)RX;
    if (not str.compare("TX")) return (ant_mode)TX;
    print_warning("ant_mode from parametern conversion has not been recognised. Setting to OFF");
    return (ant_mode)OFF;
}


//returns the maximum output buffer size (not all samples of that size will be always good)
//TODO something's wrong with this function
int param::get_output_buffer_size(){
    print_warning("Using a wrong extimation of buffer size");
    return std::ceil((float)buffer_len/(float)decim)*wave_type.size();
}

//the execution of this measurement, if TX, requres a dynamical memory allocation?
bool param::dynamic_buffer(){
    bool dynamic = false;
    for(size_t i = 0; i< wave_type.size(); i++)if(wave_type[i]!=TONES)dynamic = true;
    return dynamic;
}

//how mny rx or tx to set up
int usrp_param::get_number(ant_mode T){
    int num = 0;
    if(A_TXRX.mode == T)num++;
    if(A_RX2.mode == T)num++;
    if(B_TXRX.mode == T)num++;
    if(B_RX2.mode == T)num++;
    return num;
}

bool usrp_param::is_A_active(){
    return (A_TXRX.mode != OFF or A_RX2.mode != OFF);
}

bool usrp_param::is_B_active(){
    return (B_TXRX.mode != OFF or B_RX2.mode != OFF);
}

void server_settings::validate(){
    if(clock_reference.compare("internal") != 0 and clock_reference.compare("external") != 0){
        std::stringstream ss;
        ss<<"Clock selection mode \""<<clock_reference<<"\" is not valid for the usrp X300. Setting mode to \"internal\".";
        print_warning(ss.str());
        clock_reference = "internal";
    }
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0){
        print_error("No GPU found in the system. This version of the USRP server needs at least one GPU to work.");
        exit(-1);
    }
    if(GPU_device_index>num_gpus){
        std::stringstream ss;
        ss<<"GPU device index ("<< GPU_device_index <<") has been selected. However on the system only (" << num_gpus<< ") GPUs have been detected. Setting GPU index to "<< 0;
        print_warning(ss.str());
        GPU_device_index = 0;
    }
    if(default_rx_buffer_len < MIN_USEFULL_BUFFER or default_rx_buffer_len > MAX_USEFULL_BUFFER){
        print_warning("RX default buffer length selected may give troubles");
    }
    if(default_tx_buffer_len < MIN_USEFULL_BUFFER or default_tx_buffer_len > MAX_USEFULL_BUFFER){
        print_warning("TX default buffer length selected may give troubles");
    }
    if((not TCP_streaming) and (not FILE_writing)){
        print_warning("No data will be saved to disk or streamed");
    }
}

void server_settings::autoset(){
    TCP_streaming = false;
    FILE_writing = true;
    clock_reference = "internal";
    GPU_device_index = 0;
    default_rx_buffer_len = 1000000;
    default_tx_buffer_len = 1000000;
    validate();
}

std::string get_front_end_name(char code){
    switch(code){
        case('A'):
            return "A_TXRX";
        case('B'):
            return "A_RX2";
        case('C'):
            return "B_TXRX";
        case('D'):
            return "B_RX2";
        default:
            return "not_init";
    }
}

    //attemp to controll the thread scheduling on mac osx
    #if defined(__APPLE__)

    #include <mach/mach.h>
    #include <mach/mach_time.h>
    #include <pthread.h>

    #endif

void Thread_Prioriry(boost::thread& Thread, int priority, int affinity){
    #if not defined(__APPLE__)
    int SYSTEM_CORES = 7;//std::thread::hardware_concurrency();
    int retcode;
    int policy;
    struct sched_param scheme;
    cpu_set_t cpuset;
    pthread_t thread_ID = (pthread_t)Thread.native_handle();
    policy = SCHED_FIFO;
    scheme.sched_priority = priority;


    retcode = pthread_setschedparam(thread_ID, policy, &scheme);
    if(retcode != 0)print_warning("Cannot set thread scheduling policy");


    print_debug("Setting affinity to ", affinity%SYSTEM_CORES);
    CPU_ZERO(&cpuset);
    CPU_SET(affinity%SYSTEM_CORES, &cpuset);
    int rc = pthread_setaffinity_np(Thread.native_handle(),sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::stringstream ss;
        ss << "Error calling pthread_setaffinity_np: there may be some instability. Result: " << rc;
        print_error(ss.str());
    }
    #endif

    #if defined(__APPLE__)
    //TODO should find a way to controll the affinity policy on osx and windows
    #endif

}

void SetThreadName(boost::thread* thread, const char* threadName){
   auto handle = thread->native_handle();
   pthread_setname_np(handle,threadName);
}
/*
void SetThreadName( const char* threadName)
{
  prctl(PR_SET_NAME,threadName,0,0,0);
}
*/
