/* @file USRP_buffer_generator.cpp
 * @brief Signal generation methods for the TX_buffer_generator class.
 *
 * Contains the definitions of the methods used to generates the buffer.
 * Most of the methods used to generate the transmission buffer will call some cuda kernel wrapper.
 *
*/
#include "USRP_buffer_generator.hpp"

TX_buffer_generator::TX_buffer_generator(param* init_parameters){


    //!@brief Initialization method for the server signal generator class.
    /*!The initialization of the class assign the get() and close() methods to the private functions capable of handling and closing the resources needed for buffer generation.
    * Calling this method initialize a maximum priority cudaStream object: while the cuda driver can postpone the analysis process of the
    * received data without causing transmission error, the transmission buffer has to be generated on time to avoid transmission timing errors.
    */
    parameters = init_parameters;
    buffer_len = parameters->buffer_len;

    //check what kind of function to use with the requested buffer
    w_type last_w_type = parameters->wave_type[0];
    mixed_buffer_type = false;
    int chirp_counter = 0;
    for(size_t i = 0; i < parameters->wave_type.size(); i++){
        if(parameters->wave_type[i]!=last_w_type)mixed_buffer_type = true;
        if(parameters->wave_type[i]==CHIRP)chirp_counter++;
    }
    if(chirp_counter>1){
        print_error("Multiple chirp TX buffer generation has been requested. This feature is not implemented yet.");
        exit(-1);
    }
    if(mixed_buffer_type){
        print_error("Mixed TX buffer generation has been requested. This feature is not implemented yet.");
        exit(-1);
    }

    //assign the function pointer to the correct generator and initialize the buffer
    switch(last_w_type){
        case NODSP:
            print_error("NODSP CASE NOT IMPLEMENTED.");
            exit(-1);

        case SWONLY:
            print_error("NODSP CASE NOT IMPLEMENTED.");
            exit(-1);

        case RAMP:
            print_error("RAMP CASE NOT IMPLEMENTED.");
            exit(-1);

        case DIRECT:
            print_error("RAMP CASE NOT IMPLEMENTED.");
            exit(-1);

        case NOISE:
            get_ptr = &TX_buffer_generator::get_from_noise;
            clr_ptr = &TX_buffer_generator::close_noise;

        case TONES:

            //assign the get() and close() function pointers
            get_ptr = &TX_buffer_generator::get_from_tones;
            clr_ptr = &TX_buffer_generator::close_host;

            //initialize variables used in the get() method
            TONES_buffer_len = parameters->rate;
            TONES_last_sample = 0;

            //wrap tone information in the apposite struct
            tone_parameters info;
            info.tones_number = parameters->wave_type.size();
            info.tone_frquencies = parameters->freq.data();
            info.tones_amplitudes = parameters->ampl.data();

            //generate the buffer
            base_buffer = tone_gen(&info,parameters->rate);

            //correct the situation if the buffer length is bigger than the sampling rate
            if(buffer_len > parameters->rate){

                //take the biggest number of base buffer length that contains the transmission buffer
                int buffer_ratio = std::ceil((float)buffer_len/(float)parameters->rate);
                TONES_buffer_len = buffer_ratio*parameters->rate;

                //expand the buffer and replicate the first chunk
                base_buffer = (float2*)realloc(base_buffer, (TONES_buffer_len)*sizeof(float2));
                for(int j = 1; j < buffer_ratio; j++){
                    memcpy(base_buffer + j*parameters->rate, base_buffer, parameters->rate * sizeof(float2));
                }
            }

            //extend the base buffer so to compensate a possible buffer/buffer irrationality
            base_buffer = (float2*)realloc(base_buffer, (TONES_buffer_len + buffer_len)*sizeof(float2));

            //copy the first pice of the buffer on the last pice.
            memcpy(base_buffer + TONES_buffer_len, base_buffer, buffer_len * sizeof(float2));

            break;

        case CHIRP:

            //set the correct function pointers for get() and close() methods
            get_ptr = &TX_buffer_generator::get_from_chirp;
            clr_ptr = &TX_buffer_generator::close_device_chirp;

            //create an high priority stream (allow to overlap TX and RX operation on same GPU)
            int low_p,high_p;
            cudaDeviceGetStreamPriorityRange ( &low_p, &high_p );
            cudaStreamCreateWithPriority(&internal_stream,cudaStreamNonBlocking, high_p);

            //wrap TX signal information in the apposite struct

            h_parameter.num_steps = parameters->swipe_s[0];
            if(h_parameter.num_steps<1){
                print_warning("Number of frequency steps of the chirp signal is not set. Setting it to maximum (chirp time * sampling rate).");
                h_parameter.num_steps = parameters->chirp_t[0] * parameters->rate;
            }
            if(h_parameter.num_steps<2){
                print_warning("Number of frequency steps of the chirp signal is less than 2. This may result in single tone generation.");
            }

            //how long each tone is in samples
            h_parameter.length = parameters->chirp_t[0] * parameters->rate / h_parameter.num_steps;
            if(h_parameter.length<1){
                print_warning("Duration of each frequency in chirp signal cannot be less than one sample. Setting duration of each tone to 1.");
                h_parameter.length = 1;
                h_parameter.num_steps = parameters->chirp_t[0] * parameters->rate;
            }

            //the chirpness is expressed as double this expression somewhere
            h_parameter.chirpness = ((std::pow(2,32)-1)*(parameters->chirp_f[0]-parameters->freq[0])/((double)h_parameter.num_steps-1.))/(double)parameters->rate;

            //the algorithm used for chirp generation use this value as frequency offset
            //h_parameter.f0 = (std::pow(2,64)-1) * ((double)parameters->freq[0]/(double)parameters->rate);
            h_parameter.f0 =  (std::pow(2,32)-1) * ((double)parameters->freq[0]/(double)parameters->rate);

            //bookeeping of the last sample generated. Countrary to the single tone version this value is updated in the kernel call
            last_index = 0;

            //this variable is now deprecated.
            h_parameter.freq_norm = 0;


            //set the chirp signal amplitude
            scale = parameters->ampl[0];
            if(scale == 0)print_warning("Chirp signal amplitude is 0");

            //upload the parameter struct to the gpu
            cudaMalloc((void **)&d_parameter,sizeof(chirp_parameter));
            cudaMemcpy(d_parameter, &h_parameter, sizeof(chirp_parameter),cudaMemcpyHostToDevice);

            //allocte memory for the kernel operations
            cudaMalloc((void **)&base_buffer,sizeof(float2)*buffer_len);
            //print_chirp_params("TX",h_parameter);

            break;

    }
}

//wrapper to the correct get function
void TX_buffer_generator::get(float2** __restrict__ in){
    //!@brief Write a portion of the buffer in a memory location pointed by the argument.
    /*!The argument has to be a pointer to the pointer because the effective method used when this function is called depends on the kind of buffer descibed in the param argument of class initialization. The length of the buffer as well is set by that argument. This function does not check that the memory is available (for performance reasons).
    * @param in pointer to pointer to the buffer that will be filled with samples. The float2 is interpreted as a complex number. The user has to take care that the memory is allocated.
    */
    (this->*get_ptr)(in);
}

//wrapper to the correct cleaning function
void TX_buffer_generator::close(){
    //!@brief Clean the dynamic memory allocated in the initialization call.
    /*!This method is also the destructor of the class.
    */
    (this->*clr_ptr)();
}

//pre-fill the queue with some packets. WARNING: for some reason it doesn't update the index parameter inside the called function
//causing the waveform to restart when get() method is called outside the class (why?)
int TX_buffer_generator::prefill_queue(tx_queue* queue, preallocator<float2>* memory, param* parameter_tx){
    //! @brief This method fill the buffer queue associated with the tx streaming process.
    /*!A call to this method will retun 0 do nothing if the buffer generation method is not dynamic. in that case there is no latency associated with the buffer generation except the one associated with pointer passing that is negligible. The preallocation stops when the queue is short of nodes.
    * @param queue the queue that will be filled with buffers pointer.
    * @param memory the memory allocator using to allocate buffer's memory.
    * @param parameter_tx: the parameter describing the buffer.
    * @return The number of buffers allocated.
    */
    bool filling = true;
    // how many packets have been produced
    int filled = 0;
    float2* tmp;
    bool dynamic = parameter_tx->dynamic_buffer();
    //print_debug("preffiler is using dynamic allocation? ",dynamic);
    while(filling){
        if(dynamic)tmp = memory->get();
        get(&tmp);
        filling = queue->push(tmp);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        if(filling)filled++;
    }

    return filled;
}


//effective function to get the chirp buffer
void TX_buffer_generator::get_from_chirp(float2** __restrict__ target){

    //generate the chirp signal on gpu
    chirp_gen_wrapper(base_buffer,buffer_len,d_parameter,last_index,internal_stream,scale);

    //update index
    last_index = (last_index + parameters->buffer_len) % (h_parameter.num_steps * h_parameter.length);

    //download it to the host
    cudaMemcpyAsync(*target,base_buffer,sizeof(float2)*buffer_len,cudaMemcpyDeviceToHost,internal_stream);

    //wait for operation to be completed before returning
    cudaStreamSynchronize(internal_stream);
}

void TX_buffer_generator::get_from_noise(float2** __restrict__ target){}

//effective function to get the tone buffer
void TX_buffer_generator::get_from_tones(float2** __restrict__ target){
    *target = base_buffer + TONES_last_sample;
    TONES_last_sample = (TONES_last_sample + buffer_len) % TONES_buffer_len;
}

void TX_buffer_generator::get_from_noise(){}

//versions of the cleaning function
void TX_buffer_generator::close_host(){
    free(base_buffer);
}

void TX_buffer_generator::close_device_chirp(){
    cudaFree(base_buffer);
    cudaFree(d_parameter);
    cudaStreamDestroy(internal_stream);
}

void TX_buffer_generator::close_noise(){}
