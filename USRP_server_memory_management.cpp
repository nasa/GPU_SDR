#include "USRP_server_memory_management.hpp"


threading_condition::threading_condition(){ready=(bool)false;}

void threading_condition::wait(){
    boost::unique_lock<boost::mutex> lock(ready_mutex);
    while (!ready)
    {
        ready_cond.wait(lock);
    }    

}
void threading_condition::release(){
    {
        boost::unique_lock<boost::mutex> lock(ready_mutex);
        ready = (bool)true;
    }
    ready_cond.notify_all();
}
void threading_condition::rearm(){
    {
        boost::unique_lock<boost::mutex> lock(ready_mutex);
        ready = (bool)false;
    }
    ready_cond.notify_all();
}


VNA_decimator_helper::VNA_decimator_helper(int init_ppt, int init_buffer_len){
    ppt = init_ppt;
    buffer_len = init_buffer_len;

    
    total_len = buffer_len;
    
    valid_size = (total_len/ppt);
    
    new0 = total_len - ppt*valid_size;
    
    spare_begin = total_len - new0;
    
}

void VNA_decimator_helper::update(){
    
    total_len = buffer_len + new0;
    
    valid_size = (total_len/ppt);
    
    new0 = total_len - ppt*valid_size;
    
    spare_begin = total_len - new0;
    
    //std::cout<<"valid_size: "<< valid_size<<" total_len: "<<total_len << " new0: "<< new0<<std::endl;
}
        

gp_decimator_helper::gp_decimator_helper(int buffer_len_init, int decim_init){

    //if there is a nfft means we are decimating in steps of nfft
    decim = decim_init;
    buffer_len = buffer_len_init;
    
    //initially copy the buffer at the beginning of the pointer
    new_0 = 0;
    
    //there is no reminder at the beginning
    tot_buffer_len = buffer_len;
    
    //calculate the valid output size
    out_size = calculate_outsize();
}


//update the values. in case of different buffer length on next interation use the arg
void gp_decimator_helper::update(int new_buffer_len){
    if(new_buffer_len != 0)buffer_len = new_buffer_len;
    tot_buffer_len = new_0 + buffer_len;
    out_size = calculate_outsize();
    new_0 = calculate_spare();
}

int gp_decimator_helper::calculate_spare(){
    return tot_buffer_len - out_size*decim;
}

int gp_decimator_helper::calculate_outsize(){
    return std::floor(tot_buffer_len/decim);
}

pfb_decimator_helper::pfb_decimator_helper(int init_decim, int init_nfft){
    decim = init_decim;
    nfft = init_nfft;
}

void pfb_decimator_helper::update(int current_batch){
    buffer_len = current_batch * nfft;
    out_size = std::floor( nfft* std::floor(buffer_len/(float)nfft)/(float)decim);
    new_0 = buffer_len - out_size;
}


buffer_helper::buffer_helper (int _n_tones, int _buffer_len, int _average, int _n_eff_tones){

    //static settings
    n_tones = _n_tones; 
    buffer_len = _buffer_len;
    average = _average;
    n_eff_tones = _n_eff_tones;
    
    //initially the effective length is the buffer
    eff_length = _buffer_len;
    
    current_batch = simulate_batching();
    spare_samples = eff_length - current_batch*n_tones;
    spare_begin = eff_length - spare_samples;
    
    //initially the new buffer from USRP has to be copied in the 0 position
    new_0 = 0;
    
    copy_size = n_eff_tones * current_batch;

}

void buffer_helper::update(){

    //the buffer has to be copied after the reminder from las buffer
    new_0 = spare_samples;
    
    //the new effective length will be the reminder + the new buffer
    eff_length = spare_samples + buffer_len;
    
    //the effective batch size has to be adapted to the new effective length
    current_batch = simulate_batching();
    
    copy_size = n_eff_tones * current_batch;
    spare_samples = eff_length - current_batch*n_tones;
    spare_begin = eff_length - spare_samples;
    
    
}

//the formula I wrote had a defect for certain number combinations so..
int buffer_helper::simulate_batching(){

    int offset = 0;
    int batching = 0;
    while(offset + average*n_tones < eff_length){
        offset += n_tones;
        batching++;
    }   
    
    return batching;
    
}
