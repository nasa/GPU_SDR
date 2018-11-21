#ifndef USRP_DIAG_INCLUDED
#define USRP_DIAG_INCLUDED 1

#include "USRP_server_settings.hpp"
#include <uhd/types/metadata.hpp>
#include <chrono>

#ifndef WARNING_PRINTER
#define WARNING_PRINTER

void print_error(std::string text){
    std::cout<<std::endl << "\033[1;31mERROR\033[0m: "<< text<<std::endl;
}

void print_warning(std::string text){
    std::cout << "\033[40;1;33mWARNING\033[0m: "<< text<<std::endl;
}

void print_debug(std::string text, double value){
    std::cout << "\033[40;1;34mDEBUG\033[0m: "<< text<< " " <<value<<std::endl;
}

#endif

//print on screen error description
void interptet_rx_error(uhd::rx_metadata_t::error_code_t error){

    switch(error){
    
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_NONE:
            print_warning("RX error: Interpreter called on a error free packet.");
            break;
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_TIMEOUT:
            print_warning("RX error: No packet received, implementation timed-out.");
            break;
       case uhd::rx_metadata_t::error_code_t::ERROR_CODE_LATE_COMMAND:
            print_warning("RX error: A stream command was issued in the past.");
            break;
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_BROKEN_CHAIN:
            print_warning("RX error: Expected another stream command.");
            break;
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_OVERFLOW:
            print_warning("RX error: An internal receive buffer has filled or a sequence error has been detected (see next warning).");
            break;
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_ALIGNMENT:
            print_warning("RX error: Multi-channel alignment failed.");
            break;
        case uhd::rx_metadata_t::error_code_t::ERROR_CODE_BAD_PACKET:
            print_warning("RX error: The packet could not be parsed.");
            break;

    }

}

int get_rx_errors(uhd::rx_metadata_t *metadata, bool verbose = false){
    
    //initialize error counter
    int error = 0;
    
    //maximum result will be one in this configuration
    if(metadata->error_code != uhd::rx_metadata_t::error_code_t::ERROR_CODE_NONE){
        error=1;
        if(verbose){
            interptet_rx_error(metadata->error_code);
            if(metadata->out_of_sequence)print_warning("RX error: was a sequence error.");
        }
    }
    //if(metadata->fragment_offset!=0)print_warning("RX metadata shows fragmentation of the single packet transmission. Try reducing the buffer length.");
    if(metadata->more_fragments)print_error("RX buffer mismatch!");
    
    return error;
}

int get_tx_error(uhd::async_metadata_t *async_md, bool verbose = false){
    int error = 0;
    switch(async_md->event_code){
        case 0 :
            if(verbose)print_warning("TX async metadata non init");
            break;
    
        case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR:
            if(verbose)print_warning("TX thread encountered a time error");
            error++;
            break;

        case uhd::async_metadata_t::EVENT_CODE_BURST_ACK:
            if(verbose)print_debug("TX is fine",0);
            break;

        case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW:
            if(verbose)print_warning("TX thread encountered an undeflow");
            error++;
            break;
        case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR:
            if(verbose)print_warning("TX thread encountered a sequence error");
            error++;
            break;
        case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET:
            if(verbose)print_warning("TX thread encountered an undeflow in packet");
            error++;
            break;
        case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST:
            if(verbose)print_warning("TX thread encountered an undeflow in packet");
            error++;
            break;
        default:
            if(verbose){
                std::stringstream ss;
                ss << boost::format("TX thread got unexpected event code 0x%x.\n") % async_md->event_code << std::endl;
                print_error(ss.str());
            }
            error++;
            break;
    }
    return error;

}

void print_params(usrp_param my_parameter){

    bool A_TXRX = true;
    bool B_TXRX = true;
    bool A_RX2 = true;
    bool B_RX2 = true;
    
    std::stringstream ss;
    boost::format position_formatting("%=1.1e");
    
    ss << "\033[40;1;37m                  A N A L O G    P A R A M E T E R S                    \033[0m"<<std::endl;
    ss               <<"------------------------------------------------------------------------"<<std::endl;
    ss               <<"        device on-server ID: "<< my_parameter.usrp_number << "                "<<std::endl;
    ss               <<"------------------------------------------------------------------------"<<std::endl;
    ss << " board-> \033[40;1;37m              RF A             \033[0m";
    ss        << " \033[40;1;37m              RF B             \033[0m"<<std::endl<<std::endl;
    ss << "  ant->  " << "\033[40;1;32m     TX/RX     \033[0m ";
    ss <<               "\033[40;1;32m      RX2      \033[0m ";
    ss <<               "\033[40;1;32m     TX/RX     \033[0m ";
    ss <<               "\033[40;1;32m      RX2      \033[0m "<<std::endl<<std::endl;
    ss <<" param "<<std::endl;

    ss << "\033[47;1;30m MODE  \033[0m"<<"\t       ";
    
    ss << ant_mode_to_str(my_parameter.A_TXRX.mode)<< "\t\t";
    ss << ant_mode_to_str(my_parameter.A_RX2.mode)<< "\t\t";
    ss << ant_mode_to_str(my_parameter.B_TXRX.mode)<< "\t\t";
    ss << ant_mode_to_str(my_parameter.B_RX2.mode)<< std::endl;
    
    ss << std::scientific;
    
    if(my_parameter.A_TXRX.mode == OFF) A_TXRX = false;
    if(my_parameter.B_TXRX.mode == OFF) B_TXRX = false;
    if(my_parameter.A_RX2.mode == OFF) A_RX2 = false;
    if(my_parameter.B_RX2.mode == OFF) B_RX2 = false;

    ss << "\033[47;1;30m RATE  \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.rate/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.rate/1e6)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.rate/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.rate/1e6)).str():(std::string)"  -    ") << "\t[MHz]"<<std::endl;
    
    
    ss << "\033[47;1;30m GAIN  \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.gain/1.)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.gain/1.)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.gain/1.)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.gain/1.)).str():(std::string)"  -    ") << "\t[dB]"<<std::endl;
    
    ss << "\033[47;1;30m TONE  \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.tone/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.tone/1e6)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.tone/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.tone/1e6)).str():(std::string)"  -    ") << "\t[MHz]"<<std::endl;
    
    ss << "\033[47;1;30m DELAY \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.delay*1e3)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.delay*1e3)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.delay*1e3)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.delay*1e3)).str():(std::string)"  -    ") << "\t[msec]"<<std::endl;
    
    ss << "\033[47;1;30m  BW   \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.bw/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.bw/1e6)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.bw/1e6)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.bw/1e6)).str():(std::string)"  -    ") << "\t[MHz]"<<std::endl;
    
    ss << "\033[47;1;30m #SIGs \033[0m"<<"     ";
    
    ss << (A_TXRX?(position_formatting % (my_parameter.A_TXRX.wave_type.size()/1.)).str():(std::string)"  -    ") << "          ";
    ss << (A_RX2?(position_formatting % (my_parameter.A_RX2.wave_type.size()/1.)).str():(std::string)"  -    ")   << "          ";
    ss << (B_TXRX?(position_formatting % (my_parameter.B_TXRX.wave_type.size()/1.)).str():(std::string)"  -    ") << "          ";
    ss << (B_RX2?(position_formatting % (my_parameter.B_RX2.wave_type.size()/1.)).str():(std::string)"  -    ") << "\t[#]"<<std::endl;
    
    ss << std::endl;
    ss << "\033[40;1;37m                  S I G N A L S    P A R A M E T E R S                  \033[0m"<<std::endl<<std::endl;

    if(A_TXRX){
        
        ss <<   "\033[40;1;32mRF A TX/RX                                        No. of signals:    ";
        ss << my_parameter.A_TXRX.wave_type.size()<<" \033[0m"<<std::endl<<std::endl;
        ss <<   "\033[40;1;32mRF Buffer lenght:  "<<my_parameter.A_TXRX.buffer_len;
        if(my_parameter.A_TXRX.wave_type[0] != CHIRP and my_parameter.A_TXRX.mode == RX){
            ss <<                                   "         PFB points:   "<<my_parameter.A_TXRX.fft_tones;
            ss <<                                                             "     PFB average:  "<<  my_parameter.A_TXRX.pf_average<<" \033[0m"<<std::endl<<std::endl;
        }else{
            ss <<"                                            "<<" \033[0m"<<std::endl<<std::endl;                     
        }
        
        ss << "\033[47;1;30m   No    \033[0m"<<" "<< "\033[47;1;30m   AMP   \033[0m"<<" "<<"\033[47;1;30m  FREQ   \033[0m"<<" ";
        ss << "\033[47;1;30m  2 FRQ  \033[0m"<<" "<< "\033[47;1;30m  STEPS  \033[0m"<<" "<<"\033[47;1;30m  LAPSE  \033[0m"<< "";
        ss << "\033[47;1;30m   TYPE  \033[0m"<<std::endl;
        for(int i = 0; i<my_parameter.A_TXRX.wave_type.size();i++){
        
            ss << "    "<<(position_formatting % (i)).str() <<"      ";
            ss << (position_formatting % (my_parameter.A_TXRX.ampl[i]/1.)).str() <<"   ";
            ss << (position_formatting % (my_parameter.A_TXRX.freq[i]/1e6)).str() <<"   ";
            try{
                ss << ((my_parameter.A_TXRX.wave_type[i] == CHIRP )?((position_formatting % (my_parameter.A_TXRX.chirp_f.at(i)/1e6)).str()):(std::string)("    -  "))<<"   ";
                ss << ((my_parameter.A_TXRX.wave_type[i] == CHIRP )?((position_formatting % (my_parameter.A_TXRX.swipe_s.at(i)/1.)).str()):(std::string)  "    -  ") <<"   ";
                ss << ((my_parameter.A_TXRX.wave_type[i] == CHIRP )?((position_formatting % (my_parameter.A_TXRX.chirp_t.at(i)*1e3)).str()):(std::string) "    -  ") <<"   ";
            }catch(std::exception& error){
                print_error("Cannot print parameters! description of CHIRP signals inside global param is not coehrent!");
                return ;
            }
            ss << w_type_to_str(my_parameter.A_TXRX.wave_type[i]) <<std::endl;
        }
        ss << "   [#]     [linear]   [MHZ]      [MHz]      [#]     [msec]"<<std::endl<<std::endl;
        
    }
    if(A_RX2){
        ss <<   "\033[40;1;32mRF A    RX2                                       No. of signals:    ";
        ss << my_parameter.A_RX2.wave_type.size()<<" \033[0m"<<std::endl<<std::endl;
        ss <<   "\033[40;1;32mRF Buffer lenght:  "<<my_parameter.A_RX2.buffer_len;
        if(my_parameter.A_RX2.wave_type[0] != CHIRP and my_parameter.A_RX2.mode == RX){
            ss <<                                   "      PFB points:   "<<my_parameter.A_RX2.fft_tones;
            ss <<                                                          "     PFB average:  "<<  my_parameter.A_RX2.pf_average<<" \033[0m"<<std::endl<<std::endl;
        }else{
            ss <<"                                            "<<" \033[0m"<<std::endl<<std::endl;                     
        }
        ss << "\033[47;1;30m   No    \033[0m"<<" "<< "\033[47;1;30m   AMP   \033[0m"<<" "<<"\033[47;1;30m  FREQ   \033[0m"<<" ";
        ss << "\033[47;1;30m  2 FRQ  \033[0m"<<" "<< "\033[47;1;30m  STEPS  \033[0m"<<" "<<"\033[47;1;30m  LAPSE  \033[0m"<< "";
        ss << "\033[47;1;30m   TYPE  \033[0m"<<std::endl;
        for(int i = 0; i<my_parameter.A_RX2.wave_type.size();i++){
            ss << "    "<<(position_formatting % (i)).str() <<"      ";
            ss << (position_formatting % (my_parameter.A_RX2.ampl[i]/1.)).str() <<"   ";
            ss << (position_formatting % (my_parameter.A_RX2.freq[i]/1e6)).str() <<"   ";
            try{
                ss << ((my_parameter.A_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.A_RX2.chirp_f.at(i)/1e6)).str()):(std::string)("    -  "))<<"   ";
                ss << ((my_parameter.A_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.A_RX2.swipe_s.at(i)/1.)).str()):(std::string)  "    -  ") <<"   ";
                ss << ((my_parameter.A_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.A_RX2.chirp_t.at(i)*1e3)).str()):(std::string) "    -  ") <<"   ";
            }catch(std::exception& error){
                print_error("Cannot print parameters! description of CHIRP signals inside global param is not coehrent!");
                return ;
            }
            ss << w_type_to_str(my_parameter.A_RX2.wave_type[i]) <<std::endl;
        }
        ss << "   [#]     [linear]   [MHZ]      [MHz]      [#]     [msec]"<<std::endl<<std::endl;
        
    }
    if(B_TXRX){
        ss <<   "\033[40;1;32mRF B TX/RX                                        No. of signals:    ";
        ss << my_parameter.B_TXRX.wave_type.size()<<" \033[0m"<<std::endl<<std::endl;
        ss <<   "\033[40;1;32mRF Buffer lenght:  "<<my_parameter.B_TXRX.buffer_len;
        if(my_parameter.B_TXRX.wave_type[0] != CHIRP and my_parameter.B_TXRX.mode == RX){
            ss <<                                   "         PFB points:   "<<my_parameter.B_TXRX.fft_tones;
            ss <<                                                          "        PFB average:  "<<  my_parameter.B_TXRX.pf_average<<" \033[0m"<<std::endl<<std::endl;
        }else{
            ss <<"                                            "<<" \033[0m"<<std::endl<<std::endl;                     
        }
        ss << "\033[47;1;30m   No    \033[0m"<<" "<< "\033[47;1;30m   AMP   \033[0m"<<" "<<"\033[47;1;30m  FREQ   \033[0m"<<" ";
        ss << "\033[47;1;30m  2 FRQ  \033[0m"<<" "<< "\033[47;1;30m  STEPS  \033[0m"<<" "<<"\033[47;1;30m  LAPSE  \033[0m"<< "";
        ss << "\033[47;1;30m   TYPE  \033[0m"<<std::endl;
       
        
        for(int i = 0; i<my_parameter.B_TXRX.wave_type.size();i++){
            ss << "    "<<(position_formatting % (i)).str() <<"      ";
            ss << (position_formatting % (my_parameter.B_TXRX.ampl[i]/1.)).str() <<"   ";
            ss << (position_formatting % (my_parameter.B_TXRX.freq[i]/1e6)).str() <<"   ";
            try{
                ss << ((my_parameter.B_TXRX.wave_type[i] == CHIRP )?((position_formatting % (my_parameter.B_TXRX.chirp_f.at(i)/1e6)).str()):(std::string)("    -  "))<<"   ";
                ss << ((my_parameter.B_TXRX.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.B_TXRX.swipe_s.at(i)/1.)).str()):(std::string)  "    -  ") <<"   ";
                ss << ((my_parameter.B_TXRX.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.B_TXRX.chirp_t.at(i)*1e3)).str()):(std::string) "    -  ") <<"   ";
            }catch(std::exception& error){
                print_error("Cannot print parameters! description of SWIPE signals inside global param is not coehrent!");
                return ;
            }
            ss << w_type_to_str(my_parameter.B_TXRX.wave_type[i]) <<std::endl;
        }
        ss << "   [#]     [linear]   [MHZ]      [MHz]      [#]     [msec]"<<std::endl<<std::endl;
        
    }
    if(B_RX2){
        ss <<   "\033[40;1;32mRF B    RX2                                       No. of signals:    ";
        ss << my_parameter.B_RX2.wave_type.size()<<" \033[0m"<<std::endl<<std::endl;
        ss <<   "\033[40;1;32mRF Buffer lenght:  "<<my_parameter.B_RX2.buffer_len;
        if(my_parameter.B_RX2.wave_type[0] != CHIRP and my_parameter.B_RX2.mode == RX){
            ss <<                                   "            PFB points:   "<<my_parameter.B_RX2.fft_tones;
            ss <<                                                             "        PFB average:  "<<  my_parameter.B_RX2.pf_average<<" \033[0m"<<std::endl<<std::endl;
        }else{
            ss <<"                                            "<<" \033[0m"<<std::endl<<std::endl;                     
        }
        ss << "\033[47;1;30m   No    \033[0m"<<" "<< "\033[47;1;30m   AMP   \033[0m"<<" "<<"\033[47;1;30m  FREQ   \033[0m"<<" ";
        ss << "\033[47;1;30m  2 FRQ  \033[0m"<<" "<< "\033[47;1;30m  STEPS  \033[0m"<<" "<<"\033[47;1;30m  LAPSE  \033[0m"<< "";
        ss << "\033[47;1;30m   TYPE  \033[0m"<<std::endl;
        for(int i = 0; i<my_parameter.B_RX2.wave_type.size();i++){
            ss << "    "<<(position_formatting % (i)).str() <<"      ";
            ss << (position_formatting % (my_parameter.B_RX2.ampl[i]/1.)).str() <<"   ";
            ss << (position_formatting % (my_parameter.B_RX2.freq[i]/1e6)).str() <<"   ";
            try{
                ss << ((my_parameter.B_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.B_RX2.chirp_f.at(i)/1e6)).str()):(std::string)("    -  "))<<"   ";
                ss << ((my_parameter.B_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.B_RX2.swipe_s.at(i)/1.)).str()):(std::string)  "    -  ") <<"   ";
                ss << ((my_parameter.B_RX2.wave_type[i] == CHIRP)?((position_formatting % (my_parameter.B_RX2.chirp_t.at(i)*1e3)).str()):(std::string) "    -  ") <<"   ";
            }catch(std::exception& error){
                print_error("Cannot print parameters! description of SWIPE signals inside global param is not coehrent!");
                return ;
            }
            ss << w_type_to_str(my_parameter.B_RX2.wave_type[i]) <<std::endl;
        }
        ss << "   [#]     [linear]   [MHZ]      [MHz]      [#]     [msec]"<<std::endl<<std::endl;
        
    }
    std::cout << ss.str();
    
}


class stop_watch{
    //using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> dsec;
    public:
        
        stop_watch(){
            start_t = boost::chrono::high_resolution_clock::now();
        
        }
        
        void start(){
            start_t = boost::chrono::high_resolution_clock::now();
            elapsed_time = get_time();
            state = true;
        }
        
        void stop(){
            double y = get_time();
            double x = y - elapsed_time;
            if(state){
                total_time+=x;
                state = false;
            }
        }
        
        void reset(){
            start_t = boost::chrono::high_resolution_clock::now();
            total_time = 0;
            state = false;
        }
        
        double get(){
            if(state){
                print_warning("Getting a running stopwatch value");
            }
            return total_time;
        }
        
        void store(){
            acc.push_back(total_time);
            if(state){
                print_warning("Storing a running stopwatch value");
            }
        }
        
        double get_average(){
            if(state){
                    print_warning("Getting the average of a running stopwatch");
                }
            double avg = 0;
            for(size_t i = 0; i < acc.size(); i++){
                avg += acc[i];
            }
            avg/=acc.size();
            return avg;
        }
        
        void cycle(){
            stop();
            store();
            reset();
        }
        
    private:
        
        double get_time(){
            boost::chrono::nanoseconds ns = boost::chrono::high_resolution_clock::now() - start_t;
            return 1e-9 * ns.count();
        }
        
        boost::chrono::high_resolution_clock::time_point start_t;
        
        double elapsed_time = 0;
        
        double total_time = 0;
        
        std::vector<double> acc;
        
        bool state = false;   
        
};

#endif





