#pragma once
#ifndef USRP_DIAG_INCLUDED
#define USRP_DIAG_INCLUDED

#include "USRP_server_settings.hpp"
#include <uhd/types/metadata.hpp>
#include <chrono>

//! @brief Set the htread name reported in the logging.
void set_this_thread_name(std::string thread_name);

//print on screen error description
void interptet_rx_error(uhd::rx_metadata_t::error_code_t error);

int get_rx_errors(uhd::rx_metadata_t *metadata, bool verbose = false);

//! @brief Interpret tx errors from the async usrp comunication.
int get_tx_error(uhd::async_metadata_t *async_md, bool verbose = false);

//! @brief Print parameters on the terminal in a readable way.
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

//! @brief initialize the logger for the server.
//! Creates or access the folder logs. each time the server starts, creates an enumerated log file.
void init_logger();

//! @brief Define the pointer to the logging file backend.
typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > file_sink;

//! @brief Shared pointer to the logfile writer object.
extern boost::shared_ptr< file_sink > pLogSink;

#endif
