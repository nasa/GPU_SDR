#include "USRP_server_diagnostic.cpp"
#include "USRP_server_settings.hpp"
#include "USRP_buffer_generator.cpp"
#include "USRP_server_memory_management.cpp"
#include "USRP_hardware_manager.cpp"
#include "USRP_demodulator.cpp"
#include "USRP_buffer_generator.cpp"
#include "USRP_server_link_threads.cpp"
#include "USRP_file_writer.cpp"
#include "USRP_server_network.cpp"
#include "kernels.cu"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char **argv){

    std::cout << "\033[40;1;32mUSRP GPU Server v 2.0\033[0m" << std::endl;
    
    bool file_write, net_streaming, sw_loop;
    std::string clock;
    int port_async,port_sync;
    
    bool active = true;
    
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
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")){
        std::cout << "USRP GPU server version 2.0. Consult online documentation on how to use this server." << std::endl;
        return ~0;
    }

    
    server_settings settings;
    settings.autoset();
    settings.TCP_streaming = net_streaming;
    settings.FILE_writing = file_write;
    
    //look for USRP
    hardware_manager usrp(&settings,sw_loop);
    
    //look for CUDA
    TXRX thread_manager(&settings, &usrp, true);
    
    //look for USER
    Async_server async(true);
    
    while(active){
    
        usrp_param global_parameters;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        if(async.connected()){
            bool res = async.recv_async(global_parameters);
            res = chk_param(&global_parameters);
            json_res = new std::string(res?server_ack("Message received"):server_nack("Cannot convert JSON to params"));
            async.send_async(json_res);

            print_params(global_parameters);
            thread_manager.set(&global_parameters);
            thread_manager.start(&global_parameters);
            bool done = false;
            while(not done){
                done = thread_manager.stop();
                boost::this_thread::sleep_for(boost::chrono::milliseconds{500});
                //if (async.chk_new_command())done = thread_manager.stop(true); //this is not working
            }
            server_ack("EOM: end of measurement");
        }
    }
    
    return 0;
    
 
}
