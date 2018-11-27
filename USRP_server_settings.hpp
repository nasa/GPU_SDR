#ifndef USRP_SERVER_SETTING_IMPORTED
#define USRP_SERVER_SETTING_IMPORTED 1
#include <iostream>
#include <fstream>
#include <csignal>
#include <memory>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sched.h>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <iostream>
#include <assert.h> 
#include <sstream>
#include <future>

#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>


#include <boost/thread/thread.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/atomic.hpp>
#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/asio/use_future.hpp>
#include <boost/array.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/timer/timer.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/exception/diagnostic_information.hpp> 
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/asio/basic_deadline_timer.hpp>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "USRP_server_diagnostic.cpp"

#ifndef WARNING_PRINTER
#define WARNING_PRINTER

void print_error(std::string text){
    std::cout<<std::endl << "\033[1;31mERROR\033[0m: "<< text<<std::endl;
}

void print_warning(std::string text){
    std::cout << "\033[40;1;33mWARNING\033[0m: "<< text<<std::endl;
}

void print_debug(std::string text, double value = std::numeric_limits<double>::quiet_NaN()){
    std::cout << "\033[40;1;34mDEBUG\033[0m: "<< text<< " " << isnan(value)?"":boost::lexical_cast<std::string>(value);
    std::cout<<std::endl;
}



#endif

//length of the TX and RX queue. for non real time application longer queue are needed to stack up data

//this two values increase the ammount of cpu RAM initially allocated. Increasing those values will result in more memory usage.
#define RX_QUEUE_LENGTH     200
#define TX_QUEUE_LENGTH     200

//increasing those values will only increase the limit of RAM that COULD be used.
#define ERROR_QUEUE_LENGTH  20000
#define STREAM_QUEUE_LENGTH 20000
#define SW_LOOP_QUEUE_LENGTH 200
#define SECONDARY_STREAM_QUEUE_LENGTH 20000 //between the stream and the filewriter (keep it long if writing files)

//cut-off frequency of the post-demodulation decimator filter (relative to Nyquist)(deprecated)
#define ADDITIONAL_FILTER_FCUT 0.2

//buffer safety lengths
#define MAX_USEFULL_BUFFER 6000000
#define MIN_USEFULL_BUFFER 50000

#define DEFAULT_BUFFER_LEN 1000000

int TCP_SYNC_PORT = 61360;
int TCP_ASYNC_PORT = 22001;

//valid for TX and RX operations, describe the signal generation/demodulation.
enum w_type { TONES, CHIRP, NOISE , RAMP, NODSP, SWONLY};

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
        case(NODSP):
            comp_string = "NODSP";
            break;
        case(SWONLY):
            comp_string = "SWONLY";
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
    
    return conv;
    
}
std::vector<w_type> string_to_w_type_vector(std::vector<std::string> string_vector){
    std::vector<w_type> res(string_vector.size());
    for(int i = 0; i <string_vector.size(); i++ ){
        res[i] = string_to_w_type(string_vector[i]);
    }
    return res;
}
//state of the USRP antenna
enum ant_mode { TX, RX, OFF };
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

//describe the hardware and software paramenter for a single antenna of the USRP.
struct param{
    
    //how to use the selected antenna
    ant_mode mode = OFF;
    
    //hardware parameters
    int rate,tone,gain,bw;
    
    //runtime hardware parameters
    size_t samples;
    float delay;
    float burst_on;  //time length of the bursts in seconds
    float burst_off; //time between bursts in seconds
    size_t buffer_len;  //length of the transport buffer (both GPU and USRP). SET to 0 for default.
    bool tuning_mode;   //0 for integer and 1 for fractional
    //software signal parameters
    std::vector<int> freq;
    std::vector<w_type> wave_type;
    std::vector<float> ampl;
    size_t decim;              //all channels have the same decimation factor
    std::vector<float> chirp_t;
    std::vector<int> chirp_f;
    std::vector<int> swipe_s;
    
    //polyphase filter bank specific
    int fft_tones; // it is an int because of size_t* incompatible with cufft calls
    size_t pf_average;
    
    //returns the maximum output buffer size (not all samples of that size will be always good)
    //TODO something's wrong with this function
    int get_output_buffer_size(){
        print_warning("Using a wrong extimation of buffer size");
        return std::ceil((float)buffer_len/(float)decim)*wave_type.size();
    }
    
    //the execution of this measurement, if TX, requres a dynamical memory allocation?
    bool dynamic_buffer(){
        bool dynamic = false;
        for(int i = 0; i< wave_type.size(); i++)if(wave_type[i]!=TONES)dynamic = true;
        return dynamic;
    }
};

//should desctibe the settings for a single USRP
//ther is a parameter struct for each antenna
struct usrp_param{

    int usrp_number;

    param A_TXRX;
    param B_TXRX;
    param A_RX2;
    param B_RX2;
    
    //how mny rx or tx to set up
    int get_number(ant_mode T){
        int num = 0;
        if(A_TXRX.mode == T)num++;
        if(A_RX2.mode == T)num++;
        if(B_TXRX.mode == T)num++;
        if(B_RX2.mode == T)num++;
        return num;
    }
    
    bool is_A_active(){
        return (A_TXRX.mode != OFF or A_RX2.mode != OFF);
    }
    
    bool is_B_active(){
        return (B_TXRX.mode != OFF or B_RX2.mode != OFF);
    }
    
};

//contains the general setting to use for the USRP
//and the general server settings
struct server_settings{

    //internal or external clock reference
    std::string clock_reference;
    
    //which gpu use for signal processing on this device
    int GPU_device_index;
    
    //defaults buffer lengths
    int default_rx_buffer_len;
    int default_tx_buffer_len;
    
    //enable TCP streaming
    bool TCP_streaming;    
    
    //enable file writing
    bool FILE_writing;
    
    void validate(){
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
    
    void autoset(){
        TCP_streaming = false;
        FILE_writing = true;
        clock_reference = "internal";
        GPU_device_index = 0;
        default_rx_buffer_len = 1000000;
        default_tx_buffer_len = 1000000;
        validate();
    }

};

//wrapping the buffer with some metadata
struct RX_wrapper{
    float2* buffer;     //pointer to data content
    int usrp_number;    //identifies the usrp
    char front_end_code;     //specify RF frontend
    int packet_number;  //packet number
    int length;         //length of the buffer
    int errors;         //how many errors occured
    int channels;
};

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

//queues used for data communication between data generation/analysis classes and hardware interface class

typedef boost::lockfree::queue< RX_wrapper, boost::lockfree::fixed_sized<(bool)true>> rx_queue;
typedef boost::lockfree::queue< float2*, boost::lockfree::fixed_sized<(bool)true>> tx_queue;
typedef boost::lockfree::queue< int, boost::lockfree::fixed_sized<(bool)true>> error_queue;

#endif
