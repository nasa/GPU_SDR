#pragma once
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
        boost::lockfree::queue< intptr_t, boost::lockfree::fixed_sized<(bool)true>>* deallocated;

        bool prefil;//if fals the queue will not adjust automatically

        preallocator(int init_vector_size, int init_pipe_size, bool prefill_init = true, int core = -1){
            counter = 0;
            prefil = prefill_init; //controls the prefill mechanism
            vector_size = init_vector_size;
            wait_on_full = 5;
            pipe_size = init_pipe_size;
            allocated = new  boost::lockfree::queue< intptr_t ,boost::lockfree::fixed_sized<(bool)true>> (init_pipe_size);
            deallocated = new  boost::lockfree::queue< intptr_t ,boost::lockfree::fixed_sized<(bool)true>>(init_pipe_size);
            filler = new boost::thread(boost::bind(&preallocator::queue_filler,this));
            if(core>-1)Thread_Prioriry(*filler, 40, core);
            deallocator = new boost::thread(boost::bind(&preallocator::queue_deallocator,this));
            //if(core>-1)Thread_Prioriry(*deallocator, 40, core);
            while(counter<pipe_size-1)boost::this_thread::sleep_for(boost::chrono::milliseconds{200});

        }

        vector_type* get(){


            intptr_t thatvalue_other;

            while(not allocated->pop(thatvalue_other))boost::this_thread::sleep_for(boost::chrono::microseconds{10});

            counter--;
            return reinterpret_cast<vector_type*>(thatvalue_other);

        }

        void trash(vector_type* trash_vector){

            while(not deallocated->push(reinterpret_cast<intptr_t>(trash_vector)))boost::this_thread::sleep_for(boost::chrono::microseconds{1});
            counter++;

        }

        void close(){
            deallocator->interrupt();
            deallocator->join();
            delete deallocator;
            deallocator = nullptr;

            filler->interrupt();
            filler->join();
            delete filler;
            filler = nullptr;

            delete allocated;
            delete deallocated;



        }

    private:
        bool warning = (bool)true;
        boost::thread* filler;
        boost::thread* deallocator;
        std::atomic<int> counter;

        void queue_deallocator(){
            set_this_thread_name("queue_deallocator");
            bool active = (bool)true;
            while(active){
                try{
                    boost::this_thread::interruption_point();
                    intptr_t trash_vector = 0;
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
            set_this_thread_name("queue_filler");
            bool active = (bool)true;
            vector_type* h_aPinned=NULL;
            intptr_t thatvalue = 1;
            int debug_counter = 0;

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
};
#endif
