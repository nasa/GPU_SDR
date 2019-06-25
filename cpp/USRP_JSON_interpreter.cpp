#include "USRP_JSON_interpreter.hpp"

//this function will read the arrays inside a json file and put them in a std vector.
template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt,
                         boost::property_tree::ptree::key_type const& key,
                         boost::property_tree::ptree::key_type const& sub_key
                         ){
    std::vector<T> r;
    if (sub_key == "NULL"){
        for (auto& item : pt.get_child(key)) r.push_back(item.second.get_value<T>());
    }else{
        for (auto& item : pt.get_child(key).get_child(sub_key)) r.push_back(item.second.get_value<T>());
    }
    return r;
}

//convert a json string into a parameter object
bool string2param(std::string data, usrp_param &my_parameter){
    short device;

    std::stringstream serial_json(data);

    boost::property_tree::ptree parameters;

    try{
        boost::property_tree::read_json(serial_json, parameters);
        device = parameters.get<int>("device");
        std::cout << "Setting parameters for USRP # " << device <<std::endl;
        my_parameter.usrp_number = device;

    }catch (boost::exception &error){
        std::cerr <<boost::diagnostic_information(error)<< std::endl;
        print_error("missing device ID or wrong JSON string");
        return false;
    }

    try{
        my_parameter.A_TXRX.mode = ant_mode_from_string( parameters.get_child("A_TXRX").get<std::string>("mode"));
        my_parameter.B_TXRX.mode = ant_mode_from_string( parameters.get_child("B_TXRX").get<std::string>("mode"));
        my_parameter.A_RX2.mode  = ant_mode_from_string( parameters.get_child("A_RX2").get<std::string>("mode"));
        my_parameter.B_RX2.mode  = ant_mode_from_string( parameters.get_child("B_RX2").get<std::string>("mode"));
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"mode\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.tone = parameters.get_child("A_TXRX").get<double>("rf");
        my_parameter.B_TXRX.tone = parameters.get_child("B_TXRX").get<double>("rf");
        my_parameter.A_RX2.tone  = parameters.get_child("A_RX2").get<double>("rf");
        my_parameter.B_RX2.tone  = parameters.get_child("B_RX2").get<double>("rf");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"rf\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.tuning_mode = parameters.get_child("A_TXRX").get<size_t>("tuning_mode");
        my_parameter.B_TXRX.tuning_mode = parameters.get_child("B_TXRX").get<size_t>("tuning_mode");
        my_parameter.A_RX2.tuning_mode  = parameters.get_child("A_RX2").get<size_t>("tuning_mode");
        my_parameter.B_RX2.tuning_mode  = parameters.get_child("B_RX2").get<size_t>("tuning_mode");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"tuning_mode\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.rate = parameters.get_child("A_TXRX").get<double>("rate");
        my_parameter.B_TXRX.rate = parameters.get_child("B_TXRX").get<double>("rate");
        my_parameter.A_RX2.rate  = parameters.get_child("A_RX2").get<double>("rate");
        my_parameter.B_RX2.rate  = parameters.get_child("B_RX2").get<double>("rate");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"rate\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.decim = parameters.get_child("A_TXRX").get<double>("decim");
        my_parameter.B_TXRX.decim = parameters.get_child("B_TXRX").get<double>("decim");
        my_parameter.A_RX2.decim  = parameters.get_child("A_RX2").get<double>("decim");
        my_parameter.B_RX2.decim  = parameters.get_child("B_RX2").get<double>("decim");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"decim\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.fft_tones = parameters.get_child("A_TXRX").get<double>("fft_tones");
        my_parameter.B_TXRX.fft_tones = parameters.get_child("B_TXRX").get<double>("fft_tones");
        my_parameter.A_RX2.fft_tones  = parameters.get_child("A_RX2").get<double>("fft_tones");
        my_parameter.B_RX2.fft_tones  = parameters.get_child("B_RX2").get<double>("fft_tones");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"fft_tones\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.pf_average = parameters.get_child("A_TXRX").get<double>("pf_average");
        my_parameter.B_TXRX.pf_average = parameters.get_child("B_TXRX").get<double>("pf_average");
        my_parameter.A_RX2.pf_average  = parameters.get_child("A_RX2").get<double>("pf_average");
        my_parameter.B_RX2.pf_average  = parameters.get_child("B_RX2").get<double>("pf_average");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"pf_average\" match the specifications!");
        return false;
    }

    try{
        my_parameter.A_TXRX.samples = parameters.get_child("A_TXRX").get<size_t>("samples");
        my_parameter.B_TXRX.samples = parameters.get_child("B_TXRX").get<size_t>("samples");
        my_parameter.A_RX2.samples  = parameters.get_child("A_RX2").get<size_t>("samples");
        my_parameter.B_RX2.samples  = parameters.get_child("B_RX2").get<size_t>("samples");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"samples\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.buffer_len = parameters.get_child("A_TXRX").get<double>("buffer_len");
        my_parameter.B_TXRX.buffer_len = parameters.get_child("B_TXRX").get<double>("buffer_len");
        my_parameter.A_RX2.buffer_len  = parameters.get_child("A_RX2").get<double>("buffer_len");
        my_parameter.B_RX2.buffer_len  = parameters.get_child("B_RX2").get<double>("buffer_len");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"buffer_len\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.burst_off = parameters.get_child("A_TXRX").get<double>("burst_off");
        my_parameter.B_TXRX.burst_off = parameters.get_child("B_TXRX").get<double>("burst_off");
        my_parameter.A_RX2.burst_off  = parameters.get_child("A_RX2").get<double>("burst_off");
        my_parameter.B_RX2.burst_off  = parameters.get_child("B_RX2").get<double>("burst_off");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"burst_off\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.burst_on = parameters.get_child("A_TXRX").get<double>("burst_on");
        my_parameter.B_TXRX.burst_on = parameters.get_child("B_TXRX").get<double>("burst_on");
        my_parameter.A_RX2.burst_on  = parameters.get_child("A_RX2").get<double>("burst_on");
        my_parameter.B_RX2.burst_on  = parameters.get_child("B_RX2").get<double>("burst_on");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"burst_on\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.bw = parameters.get_child("A_TXRX").get<double>("bw");
        my_parameter.B_TXRX.bw = parameters.get_child("B_TXRX").get<double>("bw");
        my_parameter.A_RX2.bw  = parameters.get_child("A_RX2").get<double>("bw");
        my_parameter.B_RX2.bw  = parameters.get_child("B_RX2").get<double>("bw");
     }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"bw\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.delay = parameters.get_child("A_TXRX").get<double>("delay");
        my_parameter.B_TXRX.delay = parameters.get_child("B_TXRX").get<double>("delay");
        my_parameter.A_RX2.delay  = parameters.get_child("A_RX2").get<double>("delay");
        my_parameter.B_RX2.delay  = parameters.get_child("B_RX2").get<double>("delay");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"delay\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.gain = parameters.get_child("A_TXRX").get<double>("gain");
        my_parameter.B_TXRX.gain = parameters.get_child("B_TXRX").get<double>("gain");
        my_parameter.A_RX2.gain  = parameters.get_child("A_RX2").get<double>("gain");
        my_parameter.B_RX2.gain  = parameters.get_child("B_RX2").get<double>("gain");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"gain\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.freq = as_vector<int>(parameters, "A_TXRX","freq");
        my_parameter.B_TXRX.freq = as_vector<int>(parameters, "B_TXRX","freq");
        my_parameter.A_RX2.freq  = as_vector<int>(parameters, "A_RX2","freq");
        my_parameter.B_RX2.freq  = as_vector<int>(parameters, "B_RX2","freq");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"frequency\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.ampl = as_vector<float>(parameters, "A_TXRX","ampl");
        my_parameter.B_TXRX.ampl = as_vector<float>(parameters, "B_TXRX","ampl");
        my_parameter.A_RX2.ampl  = as_vector<float>(parameters, "A_RX2","ampl");
        my_parameter.B_RX2.ampl  = as_vector<float>(parameters, "B_RX2","ampl");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"amplitude\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.wave_type = string_to_w_type_vector( as_vector<std::string>(parameters, "A_TXRX","wave_type") );
        my_parameter.B_TXRX.wave_type = string_to_w_type_vector(as_vector<std::string>(parameters, "B_TXRX","wave_type") );
        my_parameter.A_RX2.wave_type  = string_to_w_type_vector(as_vector<std::string>(parameters, "A_RX2","wave_type") );
        my_parameter.B_RX2.wave_type  = string_to_w_type_vector(as_vector<std::string>(parameters, "B_RX2","wave_type") );
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"wave_type\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.chirp_t = as_vector<float>(parameters, "A_TXRX","chirp_t");
        my_parameter.B_TXRX.chirp_t = as_vector<float>(parameters, "B_TXRX","chirp_t");
        my_parameter.A_RX2.chirp_t  = as_vector<float>(parameters, "A_RX2","chirp_t");
        my_parameter.B_RX2.chirp_t  = as_vector<float>(parameters, "B_RX2","chirp_t");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"chirp_t\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.chirp_f = as_vector<int>(parameters, "A_TXRX","chirp_f");
        my_parameter.B_TXRX.chirp_f = as_vector<int>(parameters, "B_TXRX","chirp_f");
        my_parameter.A_RX2.chirp_f  = as_vector<int>(parameters, "A_RX2","chirp_f");
        my_parameter.B_RX2.chirp_f  = as_vector<int>(parameters, "B_RX2","chirp_f");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"chirp_f\" match the specifications!");
        return false;
    }
    try{
        my_parameter.A_TXRX.swipe_s = as_vector<int>(parameters, "A_TXRX","swipe_s");
        my_parameter.B_TXRX.swipe_s = as_vector<int>(parameters, "B_TXRX","swipe_s");
        my_parameter.A_RX2.swipe_s  = as_vector<int>(parameters, "A_RX2","swipe_s");
        my_parameter.B_RX2.swipe_s  = as_vector<int>(parameters, "B_RX2","swipe_s");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used for descriptor \"swipe_s\" match the specifications!");
        return false;
    }

    //block just added follows:
    try{
        my_parameter.A_TXRX.data_mem_mult = parameters.get_child("A_TXRX").get<double>("data_mem_mult");
        my_parameter.B_TXRX.data_mem_mult = parameters.get_child("B_TXRX").get<double>("data_mem_mult");
        my_parameter.A_RX2.data_mem_mult  = parameters.get_child("A_RX2").get<double>("data_mem_mult");
        my_parameter.B_RX2.data_mem_mult  = parameters.get_child("B_RX2").get<double>("data_mem_mult");
    }catch(std::exception& error){
        std::cout << error.what()<<std::endl;
        print_error("could not parse the JSON file correctly: be sure that the data type used\
        for descriptor \"data_mem_mult\" match the specifications!");
        return false;
    }
    //end of the block

    std::cout<<"printing parameters..."<<std::endl;
    return true;

}

bool is_pfb_active(param ant_parameter){
    bool res = false;
    for(size_t i = 0; i< ant_parameter.wave_type.size(); i++){
        if(ant_parameter.wave_type[i] == TONES or ant_parameter.wave_type[i] == NOISE)res = true;
    }
    return res;
}

//check if the parameters are physically viable
bool chk_param(usrp_param *parameter){


    if(parameter->A_TXRX.mode != OFF){

        if(is_pfb_active(parameter->A_TXRX)){
            if(parameter->A_TXRX.pf_average <= 0)parameter->A_TXRX.pf_average = 1;
            if(parameter->A_TXRX.fft_tones <= 0){
                parameter->A_TXRX.fft_tones = 2;
                std::stringstream ss;
                ss<<"number of fft bins in A_TXRX is too low. Setting it to 2.";
                print_warning(ss.str());
            }
        }
        if(parameter->A_TXRX.buffer_len == 0)parameter->A_TXRX.buffer_len = DEFAULT_BUFFER_LEN;
        if(parameter->A_TXRX.buffer_len > MAX_USEFULL_BUFFER or parameter->A_TXRX.buffer_len < MIN_USEFULL_BUFFER){
            std::stringstream ss;
            ss<<"A_TXRX buffer length was set to "<<parameter->A_TXRX.buffer_len<<" smples. This value is out of limits ["<< MIN_USEFULL_BUFFER<<","<< MAX_USEFULL_BUFFER<<"]. Reset the buffer length to default: "<<DEFAULT_BUFFER_LEN ;
            print_warning(ss.str());
            parameter->A_TXRX.buffer_len = DEFAULT_BUFFER_LEN;
        }
        for(size_t i=0; i< parameter->A_TXRX.wave_type.size(); i++){
            try{
                if(parameter->A_TXRX.wave_type[i] == CHIRP or parameter->A_TXRX.wave_type[i] == TONES){
                    if(std::abs(parameter->A_TXRX.freq.at(i)) > parameter->A_TXRX.rate){
                        std::stringstream ss;
                        ss<< "frequency descriptor "<< i <<" in \'A_TXRX\' parameter is out of Nyquist range: "<< parameter->A_TXRX.freq.at(i)<<">"<<parameter->A_TXRX.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
                if(parameter->A_TXRX.wave_type[i] == CHIRP){
                    if(std::abs(parameter->A_TXRX.chirp_f.at(i)) > parameter->A_TXRX.rate){
                        std::stringstream ss;
                        ss<< "second frequency descriptor "<< i <<" in \'A_TXRX\' parameter is out of Nyquist range: "<< parameter->A_TXRX.chirp_f.at(i)<<">"<<parameter->A_TXRX.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
            }catch(const std::out_of_range& e){
                print_error("Number of frequency descriptor does not match the number of signal mode descriptor in parameter \'A_TXRX\'");
                return false;
            }
        }
    }

    if(parameter->B_TXRX.mode != OFF){

        if(is_pfb_active(parameter->B_TXRX)){
            if(parameter->B_TXRX.pf_average <= 0)parameter->B_TXRX.pf_average = 1;
            if(parameter->B_TXRX.fft_tones <= 0){
                parameter->B_TXRX.fft_tones = 2;
                std::stringstream ss;
                ss<<"number of fft bins in B_TXRX is too low. Setting it to 2.";
                print_warning(ss.str());
            }
        }
        if(parameter->B_TXRX.buffer_len == 0)parameter->B_TXRX.buffer_len = DEFAULT_BUFFER_LEN;
        if(parameter->B_TXRX.buffer_len > MAX_USEFULL_BUFFER or parameter->B_TXRX.buffer_len < MIN_USEFULL_BUFFER){
            std::stringstream ss;
            ss<<"B_TXRX buffer length was set to "<<parameter->B_TXRX.buffer_len<<" smples. This value is out of limits ["<< MIN_USEFULL_BUFFER<<","<< MAX_USEFULL_BUFFER<<"]. Reset the buffer length to default: "<<DEFAULT_BUFFER_LEN ;
            print_warning(ss.str());
            parameter->B_TXRX.buffer_len = DEFAULT_BUFFER_LEN;
        }
        for(size_t i=0; i< parameter->B_TXRX.wave_type.size(); i++){
            try{
                if(parameter->B_TXRX.wave_type[i] == CHIRP or parameter->B_TXRX.wave_type[i] == TONES){
                    if(std::abs(parameter->B_TXRX.freq.at(i)) > parameter->B_TXRX.rate){
                        std::stringstream ss;
                        ss<< "frequency descriptor "<< i <<" in \'B_TXRX\' parameter is out of Nyquist range: "<< parameter->B_TXRX.freq.at(i)<<">"<<parameter->B_TXRX.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
                if(parameter->B_TXRX.wave_type[i] == CHIRP){
                    if(std::abs(parameter->B_TXRX.chirp_f.at(i)) > parameter->B_TXRX.rate){
                        std::stringstream ss;
                        ss<< "second frequency descriptor "<< i <<" in \'B_TXRX\' parameter is out of Nyquist range: "<< parameter->B_TXRX.chirp_f.at(i)<<">"<<parameter->B_TXRX.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
            }catch(const std::out_of_range& e){
                print_error("Number of frequency descriptor does not match the number of signal mode descriptor in parameter \'B_TXRX\'");
                return false;
            }
        }
    }
    if(parameter->A_RX2.mode != OFF){
        if(is_pfb_active(parameter->A_RX2)){
            if(parameter->A_RX2.pf_average <= 0)parameter->A_RX2.pf_average = 1;
            if(parameter->A_RX2.fft_tones <= 0){
                parameter->A_RX2.fft_tones = 2;
                std::stringstream ss;
                ss<<"number of fft bins in A_TXRX is too low. Setting it to 2.";
                print_warning(ss.str());
            }
        }
        if(parameter->A_RX2.buffer_len == 0)parameter->A_RX2.buffer_len = DEFAULT_BUFFER_LEN;
        if(parameter->A_RX2.buffer_len > MAX_USEFULL_BUFFER or parameter->A_RX2.buffer_len < MIN_USEFULL_BUFFER){
            std::stringstream ss;
            ss<<"A_RX2 buffer length was set to "<<parameter->A_RX2.buffer_len<<" smples. This value is out of limits ["<< MIN_USEFULL_BUFFER<<","<< MAX_USEFULL_BUFFER<<"]. Reset the buffer length to default: "<<DEFAULT_BUFFER_LEN ;
            print_warning(ss.str());
            parameter->A_RX2.buffer_len = DEFAULT_BUFFER_LEN;
        }
        for(size_t i=0; i< parameter->A_RX2.wave_type.size(); i++){
            try{
                if(parameter->A_RX2.wave_type[i] == CHIRP or parameter->A_RX2.wave_type[i] == TONES){
                    if(std::abs(parameter->A_RX2.freq.at(i)) > parameter->A_RX2.rate){
                        std::stringstream ss;
                        ss<< "frequency descriptor "<< i <<" in \'A_RX2\' parameter is out of Nyquist range: "<< parameter->A_RX2.freq.at(i)<<">"<<parameter->A_RX2.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
                if(parameter->A_RX2.wave_type[i] == CHIRP){
                    if(std::abs(parameter->A_RX2.chirp_f.at(i)) > parameter->A_RX2.rate){
                        std::stringstream ss;
                        ss<< "second frequency descriptor "<< i <<" in \'A_RX2\' parameter is out of Nyquist range: "<< parameter->A_RX2.chirp_f.at(i)<<">"<<parameter->A_RX2.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
            }catch(const std::out_of_range& e){
                print_error("Number of frequency descriptor does not match the number of signal mode descriptor in parameter \'A_RX2\'");
                return false;
            }
        }
    }
    if(parameter->B_RX2.mode != OFF){
        if(is_pfb_active(parameter->B_RX2)){
            if(parameter->B_RX2.pf_average <= 0)parameter->B_RX2.pf_average = 1;
            if(parameter->B_RX2.fft_tones <= 0){
                parameter->B_RX2.fft_tones = 2;
                std::stringstream ss;
                ss<<"number of fft bins in A_TXRX is too low. Setting it to 2.";
                print_warning(ss.str());
            }
        }
        if(parameter->B_RX2.buffer_len == 0)parameter->B_RX2.buffer_len = DEFAULT_BUFFER_LEN;
        if(parameter->B_RX2.buffer_len > MAX_USEFULL_BUFFER or parameter->B_RX2.buffer_len < MIN_USEFULL_BUFFER){
            std::stringstream ss;
            ss<<"B_RX2 buffer length was set to "<<parameter->B_RX2.buffer_len<<" smples. This value is out of limits ["<< MIN_USEFULL_BUFFER<<","<< MAX_USEFULL_BUFFER<<"]. Reset the buffer length to default: "<<DEFAULT_BUFFER_LEN ;
            print_warning(ss.str());
            parameter->B_RX2.buffer_len = DEFAULT_BUFFER_LEN;
        }
        for(size_t i=0; i< parameter->B_RX2.wave_type.size(); i++){
            try{
                if(parameter->B_RX2.wave_type[i] == CHIRP or parameter->B_RX2.wave_type[i] == TONES){
                    if(std::abs(parameter->B_RX2.freq.at(i)) > parameter->B_RX2.rate){
                        std::stringstream ss;
                        ss<< "frequency descriptor "<< i <<" in \'B_RX2\' parameter is out of Nyquist range: "<< parameter->B_RX2.freq.at(i)<<">"<<parameter->B_RX2.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
                if(parameter->B_RX2.wave_type[i] == CHIRP){
                    if(std::abs(parameter->B_RX2.chirp_f.at(i)) > parameter->B_RX2.rate){
                        std::stringstream ss;
                        ss<< "second frequency descriptor "<< i <<" in \'B_RX2\' parameter is out of Nyquist range: "<< parameter->B_RX2.chirp_f.at(i)<<">"<<parameter->B_RX2.rate;
                        print_error(ss.str());
                        return false;
                    }
                }
            }catch(const std::out_of_range& e){
                print_error("Number of frequency descriptor does not match the number of signal mode descriptor in parameter \'B_RX2\'");
                return false;
            }
        }
    }
    return true;
}

std::string server_ack(std::string payload){
    std::stringstream res;
    boost::property_tree::ptree response;
    response.put("type","ack");
    response.put("payload",payload);
    boost::property_tree::write_json(res,response);
    return res.str();
}

std::string server_nack(std::string payload){
    std::stringstream res;
    boost::property_tree::ptree response;
    response.put("type","nack");
    response.put("payload",payload);
    boost::property_tree::write_json(res,response);
    return res.str();
}
