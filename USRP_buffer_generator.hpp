#pragma once
#ifndef USRP_BUFFER_GEN_INCLUDED
#define USRP_BUFFER_GEN_INCLUDED

#include "kernels.cuh"
#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_server_memory_management.hpp"

class TX_buffer_generator{

    public:
    
        int buffer_len; //length of the buffer to transmit to the UHD send function
        param* parameters; //pointer to the struct containing the parameters
        //initialization of the class
        TX_buffer_generator(param* init_parameters);
        
        //wrapper to the correct get function
        void get(float2** __restrict__ in);
        
        //wrapper to the correct cleaning function
        void close();
        
        //pre-fill the queue with some packets. WARNING: for some reason it doesn't update the index parameter inside the called function
        //causing the waveform to restart when get() method is called outside the class (TODO: why?)
        int prefill_queue(tx_queue* queue, preallocator<float2>* memory, param* parameter_tx);
        
    private:
        
        //check if the requested buffer is a mixed type or if every buffer is the same
        bool mixed_buffer_type;
        
        //CPU bookeeping
        size_t last_index;
        
        //this variables are initialized and used only in case of TONES buffer generation.
        size_t TONES_last_sample,TONES_buffer_len;
        
        //scale the chirp buffer
        float scale;
        
        //can be a device or host buffer depending on the kind of buffer that is requested
        float2* base_buffer;
        
        //high prioriry stream used for Async operation. Note: this stream is used only for recursive TX operations and not for initializations.
        cudaStream_t internal_stream;
        
        //pointer to chirp parameters space on device memory
        chirp_parameter* d_parameter;
        
        //chirp parameter
        chirp_parameter h_parameter;
        
        //pointer to function to be assigned in function of the requested kind of buffer
        //the final buffer will be always written in a float2* host buffer
        void (TX_buffer_generator::*get_ptr)(float2** __restrict__);
        
        //pointer to the cleaning function
        void (TX_buffer_generator::*clr_ptr)();
        
        //effective function to get the chirp buffer
        void get_from_chirp(float2** __restrict__ target);
        
        void get_from_noise(float2** __restrict__ target);
        
        //effective function to get the tone buffer
        void get_from_tones(float2** __restrict__ target);
        
        void get_from_noise();
        
        //versions of the cleaning function
        void close_host();
        
        void close_device_chirp();
        
        void close_noise();

};

#endif
