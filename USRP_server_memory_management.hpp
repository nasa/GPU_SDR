#ifndef USRP_MEMORY_INCLUDED
#define USRP_MEMORY_INCLUDED

#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.hpp"

//class for controlling events. The mutex makes it thread safe.
class threading_condition{
    public:
        threading_condition();
        void wait();
        void release();
        void rearm();
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
        VNA_decimator_helper(int init_ppt, int init_buffer_len);
        
        void update();
        
    private:
        
        int ppt,buffer_len;

};

//this class helps with the memory copy-paste in post-demodulation decimation processes with the gp decimator (see kernels.cu)
class gp_decimator_helper{
    public:
        int new_0; //where to copy the new buffer and spare size
        int out_size;  //size of the output buffer
        
        gp_decimator_helper(int buffer_len_init, int decim_init);
        
  
        //update the values. in case of different buffer length on next interation use the arg
        void update(int new_buffer_len = 0);
        
    private:
    
        int decim; //decimation factor
        int buffer_len; //length of the RX buffer
        int tot_buffer_len; //length of the buffer + reminder
        
        int calculate_spare();
        
        int calculate_outsize();
};

//this class helps managing the decimation in the case off pfb
//NOTE: in future could be merged with the other class
class pfb_decimator_helper{
    public:
    
        int out_size; //in samples
        int new_0; // in samples

        pfb_decimator_helper(int init_decim, int init_nfft);
        
        void update(int current_batch);
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

        buffer_helper (int _n_tones, int _buffer_len, int _average, int _n_eff_tones);
        
        void update();
    
    private:
    
        //the formula I wrote had a defect for certain number combinations so..
        int simulate_batching();
};


template <typename vector_type>
class preallocator{


    public:
    
        int vector_size, pipe_size, wait_on_full;
        
        boost::lockfree::queue< intptr_t, boost::lockfree::fixed_sized<(bool)true>>* allocated;
        boost::lockfree::queue< intptr_t, boost::lockfree::fixed_sized<(bool)false>>* deallocated;
    
        bool prefil;//if fals the queue will not adjust automatically
        
        preallocator(int init_vector_size, int init_pipe_size, bool prefill_init = true);
  
        vector_type* get();
        
        void trash(vector_type* trash_vector);
        
        void close();
        
    private:
        bool warning = (bool)true;
        boost::thread* filler;
        boost::thread* deallocator;
        std::atomic<int> counter;
        
        void queue_deallocator();
        
        void queue_filler();
};

#endif
