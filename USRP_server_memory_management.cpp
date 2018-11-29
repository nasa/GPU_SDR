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



template <class vector_type>
preallocator<vector_type>::preallocator(int init_vector_size, int init_pipe_size, bool prefill_init){
    counter = 0;
    prefil = prefill_init; //controls the prefill mechanism
    vector_size = init_vector_size;
    wait_on_full = 5;    
    pipe_size = init_pipe_size;    
    allocated = new  boost::lockfree::queue< intptr_t ,boost::lockfree::fixed_sized<(bool)true>> (init_pipe_size);       
    deallocated = new  boost::lockfree::queue< intptr_t ,boost::lockfree::fixed_sized<(bool)false>>(0);        
    filler = new boost::thread(boost::bind(&preallocator::queue_filler,this));         
    deallocator = new boost::thread(boost::bind(&preallocator::queue_deallocator,this));      
    while(counter<pipe_size-1)boost::this_thread::sleep_for(boost::chrono::milliseconds{200});
            
}

template <class vector_type>
vector_type* preallocator<vector_type>::get(){

    
    intptr_t thatvalue_other;
    
    while(not allocated->pop(thatvalue_other))boost::this_thread::sleep_for(boost::chrono::microseconds{10});

    counter--;
    return reinterpret_cast<vector_type*>(thatvalue_other);
    
}
template <class vector_type>
void preallocator<vector_type>::trash(vector_type* trash_vector){

    while(not deallocated->push(reinterpret_cast<intptr_t>(trash_vector)))boost::this_thread::sleep_for(boost::chrono::microseconds{10});
    counter++;

}
template <class vector_type>
void preallocator<vector_type>::close(){

    deallocator->interrupt();
    deallocator->join();

    filler->interrupt();
    filler->join();

    delete allocated;
    delete deallocated;
    delete filler;
    delete deallocator;
    
}

template <class vector_type>
void preallocator<vector_type>::queue_deallocator(){
    bool active = (bool)true;
    while(active){
        try{
            boost::this_thread::interruption_point();
            intptr_t trash_vector;
            if(deallocated->pop(trash_vector)){
                int err_counter = 0;
                //try to recicle or cancel
                intptr_t thatvalue = reinterpret_cast<intptr_t>(trash_vector);
                while (not allocated->push(thatvalue) or err_counter){
                    err_counter++;
                    boost::this_thread::sleep_for(boost::chrono::microseconds{wait_on_full});
                }
                boost::this_thread::sleep_for(boost::chrono::milliseconds{3});
                
                
                counter++;
            }else boost::this_thread::sleep_for(boost::chrono::microseconds{wait_on_full});
        }catch (boost::thread_interrupted&){active=(bool)false;}
    }
    
    while(not deallocated->empty()){

        intptr_t trash_vector;
        if(deallocated->pop(trash_vector)){
            cudaFreeHost(static_cast<vector_type*>(reinterpret_cast<void*>(trash_vector)));
        }else{
            boost::this_thread::sleep_for(boost::chrono::milliseconds{3});
        }

    }
}

template <class vector_type>
void preallocator<vector_type>::queue_filler(){
    bool active = (bool)true;
    vector_type* h_aPinned;
    intptr_t thatvalue = 1;
    int debug_counter = 0;
    //bool warning = true;
    //initially fill the preallocated queue
    while(counter<pipe_size-1){
    
        cudaError_t status = cudaMallocHost((void**)&h_aPinned, vector_size*sizeof(vector_type));
        if (status != cudaSuccess){
            print_error("Memory manager cannot allocate pinned host memory!");
            exit(-1);
        }
        thatvalue = reinterpret_cast<intptr_t>(h_aPinned);
        while(not allocated->push(thatvalue))boost::this_thread::sleep_for(boost::chrono::microseconds{10*wait_on_full});
        counter++;
    
    }
    if(prefil){
        //print_debug("Queue activated with prefill",0);
        while(active){
            try{
                boost::this_thread::interruption_point(); 
                if(counter < pipe_size/10.){
                    cudaError_t status = cudaMallocHost((void**)&h_aPinned, vector_size*sizeof(vector_type));
                    if (status != cudaSuccess){
                        print_error("Memory manager cannot allocate pinned host memory!");
                        exit(-1);
                    }
                    thatvalue = reinterpret_cast<intptr_t>(h_aPinned);
                    while(not allocated->push(thatvalue))boost::this_thread::sleep_for(boost::chrono::microseconds{10*wait_on_full});
                    counter++;
                    debug_counter++;
                }else if(debug_counter>0 ){//and warning
                    pipe_size+=debug_counter;
                    print_debug("Internal memory manager had to adjust pipe size to ",pipe_size);
                    
                    debug_counter = 0;
                    //warning = false;
                }
                boost::this_thread::sleep_for(boost::chrono::microseconds{10*wait_on_full});
            }catch (boost::thread_interrupted&){active=(bool)false;}
        }
    }else{
        while(active){
            try{
                boost::this_thread::interruption_point();
                boost::this_thread::sleep_for(boost::chrono::microseconds{1000*wait_on_full});
            }catch (boost::thread_interrupted&){active=(bool)false;}
        }
    }
    

    while(not allocated->empty()){

        intptr_t trash_vector;
        if(allocated->pop(trash_vector)){
            cudaFreeHost(static_cast<vector_type*>(reinterpret_cast<void*>(trash_vector)));
        }else{
            boost::this_thread::sleep_for(boost::chrono::milliseconds{1});
        }
    }
}
