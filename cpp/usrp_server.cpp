#include "USRP_server_console_print.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_server_settings.hpp"
#include "USRP_buffer_generator.hpp"
#include "USRP_server_memory_management.hpp"
#include "USRP_hardware_manager.hpp"
#include "USRP_demodulator.hpp"
#include "USRP_buffer_generator.hpp"
#include "USRP_server_link_threads.hpp"
#include "USRP_file_writer.hpp"
#include "USRP_server_network.hpp"

#include "kernels.cuh"
#include <boost/program_options.hpp>

namespace po = boost::program_options;


int main(int argc, char **argv){
    uhd::set_thread_priority_safe(1.);
	init_logger();
	set_this_thread_name("Main");
	logging::add_common_attributes();
    std::cout << "\033[40;1;32mUSRP GPU Server v 2.0\033[0m" << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "Server started";
    bool file_write, net_streaming, sw_loop;
    std::string clock;
    int port_async,port_sync;

    bool active = true;
    bool uhd_dbg = false;
    std::string* json_res;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "help message")


    ("fw", po::value<bool>(&file_write)->default_value(false)->implicit_value(true), "Enable local file writing")
    ("no_net", po::value<bool>(&net_streaming)->default_value(true)->implicit_value(false), "Disable network streaming")
    ("sw_loop", po::value<bool>(&sw_loop)->default_value(false)->implicit_value(true), "Bypass USRP interaction")
    ("clock", po::value<std::string>(&clock)->default_value("internal")->implicit_value("external"), "Clock selector")
    ("async", po::value<int>(&port_async)->default_value(22001), "Define ascynchronous TCP communication port")
    ("data", po::value<int>(&port_sync)->default_value(61360), "Define scynchronous TCP data streaming port")
    ("uhd_dbg", po::value<bool>(&uhd_dbg)->default_value(false), "Enable UHD degug logging on console.")
    ("args", po::value<std::string>(&device_arguments)->default_value("noarg"), "Device argument to pass (experimental use)")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")){
        std::cout << "USRP GPU server version 2.0. Consult online documentation on how to use this server." << std::endl;
        return ~0;
    }

    if(uhd_dbg)uhd::log::set_console_level(uhd::log::severity_level::trace);
    server_settings settings;
    settings.autoset();
    settings.TCP_streaming = net_streaming;
    settings.FILE_writing = file_write;

    //look for USRP
    hardware_manager usrp(&settings,sw_loop);

    //look for CUDA, initialize memory     (last arg is debug)
    //blocks until tcp data connection is on-line if TCP streamer is enabled
    TXRX thread_manager(&settings, &usrp, false);

    //look for USER
    Async_server async(true);



    while(active){
        BOOST_LOG_TRIVIAL(info) << "Main loop";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if(async.connected()){
            usrp_param global_parameters;
            bool res = async.recv_async(global_parameters);//add here the action code as argument
            res = chk_param(&global_parameters);
            BOOST_LOG_TRIVIAL(info) << "Sending response to client application";
            json_res = new std::string(res?server_ack("Message received"):server_nack("Cannot convert JSON to params"));
            async.send_async(json_res);
            if(res){
                BOOST_LOG_TRIVIAL(info) << "Implementing parameter configuration";
                print_params(global_parameters);
                thread_manager.set(&global_parameters);
                thread_manager.start(&global_parameters);
                bool done = false;
                std::cout<< "DAq in progress:" <<std::flush;
                while(not done){
                    done = thread_manager.stop();
                    if(not done){
                        std::cout<<"."<<std::flush;
                    }else{
                        std::cout<<"*Measure complete"<<std::endl;
                    }
                    boost::this_thread::sleep_for(boost::chrono::milliseconds{500});
                    //if (async.chk_new_command())done = thread_manager.stop(true); //this is not working
                }
                json_res = new std::string(server_ack("EOM: end of measurement"));
                BOOST_LOG_TRIVIAL(info) << "Measure ended";
                async.send_async(json_res);
            }else{
                BOOST_LOG_TRIVIAL(warning) << "Parameter in the received message cannot be implemented";
            }
        }
    }

    return 0;


}
