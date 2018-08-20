/*
THIS FILE IS PART OF THE USRP SERVER.
THIS FILE CONTAINS USEFULL TOOLS TO MANAGE THE MEMORY INSIDE THE SERVER.
*/

#ifndef USRP_MEMORY_INCLUDED
#define USRP_MEMORY_INCLUDED 1

#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.cpp"

//class for controlling events. The mutex makes it thread safe.
class threading_condition{
    public:
        threading_condition(){ready=(bool)false;}
        
        void wait(){
            boost::unique_lock<boost::mutex> lock(ready_mutex);
            while (!ready)
            {
                ready_cond.wait(lock);
            }    
        
        }
        void release(){
            {
                boost::unique_lock<boost::mutex> lock(ready_mutex);
                ready = (bool)true;
            }
            ready_cond.notify_all();
        }
        void rearm(){
            {
                boost::unique_lock<boost::mutex> lock(ready_mutex);
                ready = (bool)false;
            }
            ready_cond.notify_all();
        }
    private:
        boost::condition_variable   ready_cond;
        boost::mutex                ready_mutex;
        bool                        ready;
};

//helper class for the vna decimator: the vna scan is NOT syncronous with the USRP buffer
class VNA_decimator_helper{
    public:
        int valid_size; //size to download in the current buffer
        int new0; //hwere to copy the new buffer & size of the reminder
        int total_len;
        int spare_begin;
        VNA_decimator_helper(int init_ppt, int init_buffer_len){
            ppt = init_ppt;
            buffer_len = init_buffer_len;

            
            total_len = buffer_len;
            
            valid_size = (total_len/ppt);
            
            new0 = total_len - ppt*valid_size;
            
            spare_begin = total_len - new0;
            
        }
        
        void update(){
            
            total_len = buffer_len + new0;
            
            valid_size = (total_len/ppt);
            
            new0 = total_len - ppt*valid_size;
            
            spare_begin = total_len - new0;
            
            //std::cout<<"valid_size: "<< valid_size<<" total_len: "<<total_len << " new0: "<< new0<<std::endl;
        }
        
    private:
        
        int ppt,buffer_len;

};
//this class helps with the memory copy-paste in post-pfb decimation processes with the fft buffer decimator (see kernels.cu)
/*MAYBE I DON'T NEED THIS AND I CAN SOLVE ADDING A (optional arg to the) CONSTRUCTOR (of) TO THE GP CASE
class pfb_decimator_helper{
    public:
        int new_0; //where to copy the new buffer and spare size
        int out_size;  //size of the output buffer (to be passed to kernel's wrapper)
        int current_batching; // how many valid fft's are present in the buffer
        int nfft; //size of the fft
        int decim; //decimation;
        
        pfb_decimator_helper(int init_nfft, int )
    private:
        
};
*/
//this class helps with the memory copy-paste in post-demodulation decimation processes with the gp decimator (see kernels.cu)
class gp_decimator_helper{
    public:
        int new_0; //where to copy the new buffer and spare size
        int out_size;  //size of the output buffer
        
        gp_decimator_helper(int buffer_len_init, int decim_init){
        
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
        void update(int new_buffer_len = 0){
            if(new_buffer_len != 0)buffer_len = new_buffer_len;
            tot_buffer_len = new_0 + buffer_len;
            out_size = calculate_outsize();
            new_0 = calculate_spare();
        }
        
    private:
    
        int decim; //decimation factor
        int buffer_len; //length of the RX buffer
        int tot_buffer_len; //length of the buffer + reminder
        
        int calculate_spare(){
            return tot_buffer_len - out_size*decim;
        }
        
        int calculate_outsize(){
            return std::floor(tot_buffer_len/decim);
        }
};

//this class helps managing the decimation in the case off pfb
//NOTE: in future could be merged with the other class
class pfb_decimator_helper{
    public:
    
        int out_size; //in samples
        int new_0; // in samples

        pfb_decimator_helper(int init_decim, int init_nfft){
            decim = init_decim;
            nfft = init_nfft;
        }
        
        void update(int current_batch){
            buffer_len = current_batch * nfft;
            out_size = std::floor( nfft* std::floor(buffer_len/(float)nfft)/(float)decim);
            new_0 = buffer_len - out_size;
        }
    private:
        int nfft;
        int decim; //in terms of fft buffers
        int buffer_len; //in samples
};
//helper class for moving around the buffer and the reminder buffer of the polyphase filter bank
//NOTE: this is exclusively for the PFB
class buffer_helper{
    public:
    
        int n_tones;        //number of fft points.
        int eff_length;     //total lenght of the buffer + spare samples from preceding buffer.
        int buffer_len;     //length of the buffer incoming from USRP.
        int average;        //how many buffer are averaged by the polyphase filter.
        int n_eff_tones;    //how many fft points will be effectively copied to host memory.
        int new_0;          //where to copy the next buffer.
        int copy_size;      //size of the device to host memory transfert.
        int current_batch;  //how many batches of the fft are valid.
        int spare_samples;  //how many samples must be copied to the begin of the fft input buffer.
        int spare_begin;    //from which sample in the buffer the spare has to be copied.

        buffer_helper (int _n_tones, int _buffer_len, int _average, int _n_eff_tones){
        
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
        
        void update(){
        
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
    
    private:
    
        //the formula I wrote had a defect for certain number combinations so..
        int simulate_batching(){
        
            int offset = 0;
            int batching = 0;
            while(offset + average*n_tones < eff_length){
                offset += n_tones;
                batching++;
            }   
            
            return batching;
            
        }
};


template <typename vector_type>
class preallocator{


    public:
    
        int vector_size, pipe_size, wait_on_full;
        
        boost::lockfree::queue< intptr_t, boost::lockfree::fixed_sized<(bool)true>>* allocated;
        boost::lockfree::queue< intptr_t, boost::lockfree::fixed_sized<(bool)false>>* deallocated;
    
        bool prefil;//if fals the queue will not adjust automatically
        
        preallocator(int init_vector_size, int init_pipe_size, bool prefill_init = true){
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
                 /*
                if (counter<pipe_size/4 and count > 2){
                    vector_type* h_aPinned;
                    cudaError_t status = cudaMallocHost((void**)&h_aPinned, vector_size*sizeof(vector_type));
                    if (status != cudaSuccess){
                        print_error("Error allocating pinned host memory!");
                    }else{
                        if(warning)print_warning("Buffer reciclyng mechanism failed. Consider increasing the number of preallocated buffers.");
                        return h_aPinned;
                    }
                }
                */       
        vector_type* get(){

            
            intptr_t thatvalue_other;
            
            //int count = 0;
            while(not allocated->pop(thatvalue_other))boost::this_thread::sleep_for(boost::chrono::microseconds{10});
            /*
            while(not allocated->pop(thatvalue_other)){
                count++;
                boost::this_thread::sleep_for(boost::chrono::microseconds{1});

            }
            if(count>10 and warning){
                print_debug("Cannot allocate memory fast enough. Timing mismatch: [usec] ",count);
                //warning = (bool)false;
            }*/
            counter--;
            return reinterpret_cast<vector_type*>(thatvalue_other);
            
        }
        
        void trash(vector_type* trash_vector){
        
            while(not deallocated->push(reinterpret_cast<intptr_t>(trash_vector)))boost::this_thread::sleep_for(boost::chrono::microseconds{10});
            counter++;

        }
        
        void close(){

            deallocator->interrupt();
            deallocator->join();

            filler->interrupt();
            filler->join();

            delete allocated;
            delete deallocated;
            delete filler;
            delete deallocator;
            
        }
        
    private:
        bool warning = (bool)true;
        boost::thread* filler;
        boost::thread* deallocator;
        std::atomic<int> counter;
        
        void queue_deallocator(){
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
        
        void queue_filler(){
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
                //print_debug("Queue activated without prefill",1);
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
};

#endif
