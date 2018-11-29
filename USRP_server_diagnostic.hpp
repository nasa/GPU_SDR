#ifndef USRP_DIAG_INCLUDED
#define USRP_DIAG_INCLUDED

#include "USRP_server_settings.hpp"
#include <uhd/types/metadata.hpp>
#include <chrono>

//print on screen error description
void interptet_rx_error(uhd::rx_metadata_t::error_code_t error);

int get_rx_errors(uhd::rx_metadata_t *metadata, bool verbose = false);

int get_tx_error(uhd::async_metadata_t *async_md, bool verbose = false);

void print_params(usrp_param my_parameter);

class stop_watch{

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> dsec;
    
    public:
        
        stop_watch();
        
        void start();
        
        void stop();
        
        void reset();
        
        double get();
        
        void store();
        
        double get_average();
        
        void cycle();
        
    private:
        
        double get_time();
        
        boost::chrono::high_resolution_clock::time_point start_t;
        
        double elapsed_time = 0;
        
        double total_time = 0;
        
        std::vector<double> acc;
        
        bool state = false;   
        
};

#endif





