//! @file USRP_server_settings.hpp
/* @brief Containd definitions of variables, macros and functions needed for the server basic settings and definitions.
 *
 *
 *
*/
//! @cond
#ifndef USRP_SERVER_SETTING_IMPORTED
#define USRP_SERVER_SETTING_IMPORTED

#include <sys/prctl.h>
#include <fstream>
#include <csignal>
#include <memory>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <sched.h>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <assert.h>
#include <future>

#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/exception.hpp>
#include <uhd/usrp/multi_usrp.hpp>

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
#include <boost/thread/thread.hpp>

#define BOOST_LOG_DYN_LINK 1
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;
using namespace logging::trivial;

#include "USRP_server_console_print.hpp"

#include <cuda_runtime.h>
//#include <cufft.h>

//offset for LO tuning.
#define OFFSET_F 0//100000000

//length of the TX and RX queue. for non real time application longer queue are needed to stack up data

//this two values increase the ammount of cpu RAM initially allocated. Increasing those values will result in more memory usage.

//! Describe how many buffer will be allocated for the frontend receiver queue memory pool. Increasing those values will result in more memory usage.
#define RX_QUEUE_LENGTH     100

//! Describe how many buffer will be allocated for the frontend transmitter queue memory pool. Increasing those values will result in more memory usage.
#define TX_QUEUE_LENGTH     100

//increasing those values will only increase the limit of RAM that COULD be used.
#define ERROR_QUEUE_LENGTH  1000
#define STREAM_QUEUE_LENGTH 1000

//! Describe how many buffer will be allocated for the software emoulation of the USRP. This value is only important when the software is used in software loopback mode. Increasing those values will result in more memory usage.
#define SW_LOOP_QUEUE_LENGTH 100
#define SECONDARY_STREAM_QUEUE_LENGTH 100 //between the stream and the filewriter (keep it long if writing files)

//cut-off frequency of the post-demodulation decimator filter (relative to Nyquist)(deprecated)
#define ADDITIONAL_FILTER_FCUT 0.2

//buffer safety lengths
#define MAX_USEFULL_BUFFER 6000000
#define MIN_USEFULL_BUFFER 50000

#define DEFAULT_BUFFER_LEN 1000000

//! @brief Port used for client server communication.
extern int TCP_SYNC_PORT;

//! @brief Port used for data streaming.
extern int TCP_ASYNC_PORT;

extern std::string device_arguments;

//valid for TX and RX operations, describe the signal generation/demodulation.
enum w_type { TONES, CHIRP, NOISE , RAMP, NODSP, SWONLY, DIRECT};

std::string w_type_to_str(w_type enumerator);

w_type string_to_w_type(std::string input_string);

std::vector<w_type> string_to_w_type_vector(std::vector<std::string> string_vector);

//state of the USRP antenna
enum ant_mode { TX, RX, OFF };

std::string ant_mode_to_str(ant_mode enumerator);

ant_mode ant_mode_from_string(std::string str);

//describe the hardware and software paramenter for a single antenna of the USRP.
struct param{

    //how to use the selected antenna
    ant_mode mode = OFF;

    //hardware parameters
    int rate,gain,bw;
    size_t tone;

    //runtime hardware parameters
    size_t samples;
    double delay;
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

    size_t data_mem_mult; //memory multiplier added to cope with analysis data expansion.

    //polyphase filter bank specific
    int fft_tones; // it is an int because of size_t* incompatible with cufft calls
    size_t pf_average;

    //returns the maximum output buffer size (not all samples of that size will be always good)
    //TODO something's wrong with this function
    int get_output_buffer_size();

    //the execution of this measurement, if TX, requres a dynamical memory allocation?
    bool dynamic_buffer();
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
    int get_number(ant_mode T);

    bool is_A_active();

    bool is_B_active();

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

    void validate();

    void autoset();

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

std::string get_front_end_name(char code);

//queues used for data communication between data generation/analysis classes and hardware interface class

typedef boost::lockfree::queue< RX_wrapper, boost::lockfree::fixed_sized<(bool)true>> rx_queue;
typedef boost::lockfree::queue< float2*, boost::lockfree::fixed_sized<(bool)true>> tx_queue;
typedef boost::lockfree::queue< int, boost::lockfree::fixed_sized<(bool)true>> error_queue;

//! @brief Set thread priority, scheduling policy and core affinity. Require administrative privilege. Only tested on Linux and OSX
// One of the main causes of error in the system is the RX/TX processes core switching. The switch instroduces a delay that is not well tollerated by the real-time tasks.
void Thread_Prioriry(boost::thread& Thread, int priority, int affinity);

//! @brief Set the thread name to the OS
//! Under linux one can use htop or ps to display wich thread is running and how much memory/ CPU resources are being used.
//! This function set the displayed name of the thread. It does not work with htop.
void SetThreadName(boost::thread* thread, const char* threadName);
#endif
