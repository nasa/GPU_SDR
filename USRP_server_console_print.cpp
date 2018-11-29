#include "USRP_server_console_print.hpp"


void print_error(std::string text){
    std::cout<<std::endl << "\033[1;31mERROR\033[0m: "<< text<<std::endl;
}

void print_warning(std::string text){
    std::cout << "\033[40;1;33mWARNING\033[0m: "<< text<<std::endl;
}

void print_debug(std::string text, double value){
    std::cout << "\033[40;1;34mDEBUG\033[0m: "<< text<< " " <<value<<std::endl;
}
