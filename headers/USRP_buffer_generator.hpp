/* @file USRP_buffer_generator.hpp
 * @brief class prototypes for the buffer generator.
 *
 * Contains the prototype of the TX_buffer_generator class. Such class is used to generate the transmission buffer for the SDR.
 *
 * @bug The queue_prefiller method is not working as expected. This is non-critical for most applications
 * @todo Test the behaviour of multiple class instances driving multiple USRPs/channels
 *
*/
#pragma once
#ifndef USRP_BUFFER_GEN_INCLUDED
#define USRP_BUFFER_GEN_INCLUDED

#include "kernels.cuh"
#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_server_memory_management.hpp"

//!@brief This class generates the transimission buffer given a the parameter object in the initialization.
//!The way the buffer is generated is internally managed, different kind of buffers are generated using different strategies. This class is non threadsafe.

/*!@code
//What follows is a minimal example on how to use this class.

int buffer_len = 1024;
param signal_parameters;

//initialize the parameters in a senseful way...

signal_parameters.buffer_len = buffer_len;
float2* samples[buffer_len];

//initialize the signal source
TX_buffer_generator signal_source(&signal_parameters);


for(int t=0; t<1000; t++){
    //fill the buffer with generated signal
    samples = signal_source.get();
    //do stuff with samples...
}

//clean memory.
free(samples);
signal_source.close();
@endcode
*/
//!There are three different behaviour of the class:
class TX_buffer_generator{

    public:
        //!Length of the buffer segment retrived with the TX_buffer_generator::get() method. Usually equalt to the UHD tx_buffer value.
        int buffer_len;

        //!Pointer to the struct containing the parameters used to generate the signal.
        param* parameters;
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
