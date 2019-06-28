#include "USRP_demodulator.hpp"
#include "fir.hpp"
#define checkcublas(X) assert( ( X ) == CUBLAS_STATUS_SUCCESS )

//initialization: the parameters are coming directly from the client (from the async communication thread)
//diagnostic allows to print the window on a binary file and stores some diagnostic information //TODO on a hdf5 file??
RX_buffer_demodulator::RX_buffer_demodulator(param* init_parameters, bool init_diagnostic){

    //enable or disable information diagnostic
    diagnostic = init_diagnostic;

    //store initialization parameters
    parameters = init_parameters;

    //check what kind of demodulator has been requested
    bool mixed_buffer_type = false;
    int chirp_counter = 0;

    w_type last_w_type = NODSP;

    //wrapper around the parameters (it could be empty: see default behaviour)
    try{last_w_type = parameters->wave_type.at(0);}catch(const std::out_of_range& e){
        if(diagnostic)print_warning("No signal processing options found. Transmitting full buffer.");
        last_w_type = NODSP;
    }

    for(size_t i = 0; i < parameters->wave_type.size(); i++){
        if(parameters->wave_type[i]!=last_w_type)mixed_buffer_type = true;
        if(parameters->wave_type[i]==CHIRP)chirp_counter++;
    }
    if(chirp_counter>1){
        print_error("Multiple chirp RX buffer demodulation has been requested. This feature is not implemented yet.");
        exit(-1);
    }
    // In the future this can be removed and different tones generator can be used to compose a single signal.
    if(mixed_buffer_type){
        print_error("Mixed RX buffer demodulation has been requested. This feature is not implemented yet.");
        exit(-1);
    }

    //create an high priority stream (allow to overlap TX and RX operation on same GPU)
    int low_p,high_p;
    cudaDeviceGetStreamPriorityRange ( &low_p, &high_p );
    cudaStreamCreateWithPriority(&internal_stream,cudaStreamNonBlocking, low_p);

    //needed only in case of buffer mismamtch
    int in_out_len = 0;

    //if there is a valid decimation enable the decimator
    decimator_active = (parameters->decim > 0)?true:false;
    //if(decimator_active)std::cout<<"Decimator is active: "<<parameters->decim <<std::endl;
    //print_debug("decimator is active?",decimator_active);
    //initialize the memory pointer for eventual spare buffers
    spare_size = 0;

    switch(last_w_type){
        // Insert new DSP cases below!

        case DIRECT:
          if(diagnostic)print_warning("Demodulator diagnostic enabled.");
          // Initialization code!

          // Create the device vector of frequencies
          cudaMalloc((void **)&DIRECT_tone_frquencies, parameters->wave_type.size()*sizeof(int));

          //No care for phase yet
          cudaMalloc((void **)&DIRECT_tone_phases, parameters->wave_type.size()*sizeof(int));
          cudaMemset(DIRECT_tone_phases, 0, parameters->wave_type.size()*sizeof(int));

          DIRECT_tones = (int*)malloc(parameters->wave_type.size()*sizeof(int));
          for(uint k=0; k<parameters->wave_type.size(); k++ ){
            DIRECT_tones[k] = parameters->freq[k];
          }

          // Upload the frequency vector
          cudaMemcpy(DIRECT_tone_frquencies, DIRECT_tones, parameters->freq.size() * sizeof(int),cudaMemcpyHostToDevice);

          // Allocate input memory
          cudaMalloc((void **)&direct_input,parameters->buffer_len*sizeof(float2));

          //Optimization: calculate this once.
          DIRECT_output_size = parameters->buffer_len * parameters->wave_type.size();

          // Allocate output memory
          cudaMalloc((void **)&direct_output, DIRECT_output_size*sizeof(float2));

          //Initialize bookeeping
          DIRECT_current_index = 0;

          //Attach the demodulator handle to the stream.
          cublasCreate(&handle);
          cublasSetStream(handle,internal_stream);

          DIRECT_FIR_output_size = DIRECT_output_size;

          if(parameters->decim > 0){

            //Initialize FIR taps
            fir_taps = make_sinc_window(parameters->decim * parameters->pf_average, 0.75/(parameters->decim*2), false, true);
            //for (int y = 0; )
            //make_hamming_window<float2>(parameters->buffer_len, 0, false, true)

            DIRECT_FIR_output_size /= parameters->decim;
            cudaMalloc((void **)&FIR_output, DIRECT_FIR_output_size*sizeof(float2));
            //Initialize FIR
            //NOTE: the FIR will work on one channel at time
            DIRECT_FIR = (FIR**)malloc(sizeof(FIR*)*parameters->wave_type.size());

            //could make use of multiple streams
            for(size_t k=0; k<parameters->wave_type.size(); k++)DIRECT_FIR[k] = new FIR(handle, internal_stream, fir_taps, parameters->decim, parameters->pf_average, parameters->buffer_len);
          }

          cudaMalloc((void **)&transposed, DIRECT_FIR_output_size*sizeof(float2));

          //declare pointers to functions
          process_ptr = &RX_buffer_demodulator::process_direct;
          clr_ptr = &RX_buffer_demodulator::close_direct;

          break;

        case TONES:

            //warn the user about the diagnostic
            if(diagnostic)print_warning("Demodulator diagnostic enabled.");

            //assign the correct cleaning and process function pointer
            process_ptr = &RX_buffer_demodulator::process_pfb;
            clr_ptr = &RX_buffer_demodulator::close_pfb;

            //the cut-off frequency is calculated in function of the number of tones
            fcut = 1./(2*parameters->fft_tones);

            //calculate the window and stores it in gpu memory. NOTE: this is done on the default stream.
            window = make_sinc_window(parameters->fft_tones*parameters->pf_average, fcut, diagnostic,false);

            //convert the parameter struct to a gpu version for the kernels and upload it.
            upload_multitone_parameters();

            //length of the gpu buffer for applying the polyphase filter bank
            in_out_len = parameters->fft_tones * batching;

            //allocates needed space. for description see the pointers.
            cudaMalloc((void **)&raw_input,in_out_len*sizeof(float2));
            cudaMalloc((void **)&input,in_out_len*sizeof(float2));

            cudaMalloc((void **)&output,(decimator_active?3:1)*in_out_len*sizeof(float2));
            cudaMalloc((void **)&reduced_output,parameters->wave_type.size()*batching*sizeof(float2));

            //crate the fft plan
            cufftPlanMany(&plan,1,&parameters->fft_tones,
                NULL,1,parameters->fft_tones,
                NULL,1,parameters->fft_tones,
                CUFFT_C2C, batching);

            //set the fft on the priority stream
            cufftSetStream(plan, internal_stream);

            //initialize the buffer manager
            buf_setting = new buffer_helper(parameters->fft_tones, parameters->buffer_len, parameters->pf_average, parameters->wave_type.size());

            //if the user asks for decimation extra space is needed.
            if(decimator_active){
                //TODO Currently I'm decimating the full fft buffer when only the "selected tones" should be decimated.
                //This require an apposite buffer copy kernel due to the structure of the selected tones memory.

                //the size used for this buffer it's bigger than expectes because the input size could variate.
                cudaMalloc((void **)&decim_output,sizeof(float2)*(2.*parameters->buffer_len)/(parameters->decim));

                //initialize the post demodulator decimation buffer manager
                pfb_decim_helper = new pfb_decimator_helper(parameters->decim, parameters->fft_tones);

                print_warning("When using TONES demodulation type, the decimation should be achieved increasing the number of pfb channels");
            }

            break;

        case CHIRP: //valid only for single chirp

            //warn the user about the diagnostic
            if(diagnostic)print_warning("Demodulator diagnostic enabled.");

            //set the correct function pointers for get() and close() methods
            process_ptr = &RX_buffer_demodulator::process_chirp;
            clr_ptr = &RX_buffer_demodulator::close_chirp;

            //create an high priority stream (allow to overlap TX and RX operation on same GPU)
            int low_p,high_p;
            cudaDeviceGetStreamPriorityRange ( &low_p, &high_p );
            cudaStreamCreateWithPriority(&internal_stream,cudaStreamNonBlocking, high_p);


            h_parameter.num_steps = parameters->swipe_s[0];
            if(h_parameter.num_steps<1){
                print_warning("Number of frequency steps of the chirp demodulator is not set. Setting it to maximum (chirp time * sampling rate).");
                h_parameter.num_steps = parameters->chirp_t[0] * parameters->rate;
            }
            if(h_parameter.num_steps<2){
                print_warning("Number of frequency steps of the chirp demodulator is less than 2. This may result in single tone demodulation.");
            }

            //how long each tone is in samples
            h_parameter.length = parameters->chirp_t[0] * parameters->rate / h_parameter.num_steps;
            if(h_parameter.length<1){
                print_warning("Duration of each frequency in chirp signal cannot be less than one sample. Setting duration of each tone to 1.");
                h_parameter.length = 1;
            }

            //the chirpness is expressed as double this expression somewhere
            //h_parameter.chirpness = ((float)(parameters->chirp_f[0]-parameters->freq[0])/((float)h_parameter.num_steps-1.))/(float)parameters->rate;
            h_parameter.chirpness = ((std::pow(2,32)-1)*(parameters->chirp_f[0]-parameters->freq[0])/((double)h_parameter.num_steps-1.))/(double)parameters->rate;

            //the algorithm used for chirp generation use this value as frequency offset
            //h_parameter.f0 = (float)parameters->freq[0]/(float)parameters->rate;
            h_parameter.f0 =  (std::pow(2,32)-1) * ((double)parameters->freq[0]/(double)parameters->rate);

            //bookeeping updated on CPU
            last_index = 0;

            //upload the parameter struct to the gpu
            cudaMalloc((void **)&d_parameter,sizeof(chirp_parameter));
            cudaMemcpy(d_parameter, &h_parameter, sizeof(chirp_parameter),cudaMemcpyHostToDevice);

            //allocte memory for the kernel operations
            cudaMalloc((void **)&input,sizeof(float2)*parameters->buffer_len);
            cudaMalloc((void **)&output,(decimator_active?3:1)*sizeof(float2)*parameters->buffer_len);

            //if the user asks for decimation extra space is needed.
            if(decimator_active){

                //set the decimator parameter
                ppt = h_parameter.length * parameters->decim;

                if(parameters->decim>1)print_warning("A decimation factor >1 requested in chirp demodulation. There is interpreted as ppt*decim");

                vna_helper = new VNA_decimator_helper(ppt, parameters->buffer_len);

                //initialize handle for cublas ops
                cublasCreate(&handle);
                cublasSetStream(handle,internal_stream);


                zero = make_cuComplex (0.0f, 0.0f);
                one = make_cuComplex (1.0f, 0.0f);

                //creates a profile for filtering and decimation
                profile = make_flat_window(ppt,ppt/10,false);

                //profile = make_sinc_window(ppt,0.02);
                //scale_buffer<<<1024,32>>>(profile,ppt,1./parameters->decim);
                //do not discard any point
                side = 0;
                //the size used for this buffer it's bigger than expectes because the input size could variate.
                cudaMalloc((void **)&decim_output,sizeof(float2)*std::ceil((float)parameters->buffer_len/(float)ppt)*8);

                //initialize the post demodulator decimation buffer manager
                //decim_helper = new gp_decimator_helper(parameters->buffer_len,parameters->decim);

            }

            //print_chirp_params("RX",h_parameter);

            break;

        case NOISE: //returns the full spectrum result without selecting tones.

            //warn the user about the diagnostic
            if(diagnostic)print_warning("Demodulator diagnostic enabled.");

            //assign the correct cleaning and process function pointer
            process_ptr = &RX_buffer_demodulator::process_pfb_spec;
            clr_ptr = &RX_buffer_demodulator::close_pfb_spec;

            //the cut-off frequency is calculated in function of the number of tones
            fcut = 1./(2*parameters->fft_tones);

            //calculate the window and stores it in gpu memory. NOTE: this is done on the default stream.
            window = make_sinc_window(parameters->fft_tones*parameters->pf_average, fcut, diagnostic, false);

            //convert the parameter struct to a gpu version for the kernels and upload it.
            upload_multitone_parameters();

            //length of the gpu buffer for applying the polyphase filter bank
            in_out_len = parameters->fft_tones * batching;

            //allocates needed space. for description see the pointers.
            cudaMalloc((void **)&raw_input,in_out_len*sizeof(float2));
            cudaMalloc((void **)&input,(decimator_active?3:1)*in_out_len*sizeof(float2));
            cudaMalloc((void **)&output,in_out_len*sizeof(float2));


            //crate the fft plan
            cufftPlanMany(&plan,1,&parameters->fft_tones,
                NULL,1,parameters->fft_tones,
                NULL,1,parameters->fft_tones,
                CUFFT_C2C, batching);

            //set the fft on the priority stream
            cufftSetStream(plan, internal_stream);

            //initialize the buffer manager
            buf_setting = new buffer_helper(parameters->fft_tones, parameters->buffer_len, parameters->pf_average, parameters->fft_tones);

            if(decimator_active){

                //the size used for this buffer it's bigger than expectes because the input size could variate.
                //cudaMalloc((void **)&decim_output,sizeof(float2)*(2*parameters->buffer_len)/(parameters->decim));
                cudaMalloc((void **)&decim_output,sizeof(float2)*(2*parameters->buffer_len));

                //initialize the post demodulator decimation buffer manager
                pfb_decim_helper = new pfb_decimator_helper(parameters->decim, parameters->fft_tones);
            }

            break;

        case NODSP:

            //this case is a pass through.
            process_ptr = &RX_buffer_demodulator::process_nodsp;
            clr_ptr = &RX_buffer_demodulator::close_nodsp;

            break;
        default: //the default case means that no wave type has been selected. will transmit the full buffer. //TODO
            print_error("Void demodulation operation has not been implemented yet!");
            exit(-1);
            break;
    }
}

//wrapper to the correct get function
int RX_buffer_demodulator::process(float2** __restrict__ in, float2** __restrict__ out){ return (this->*process_ptr)(in,out); }

//wrapper to the correct cleaning function
void RX_buffer_demodulator::close(){ (this->*clr_ptr)(); }

int RX_buffer_demodulator::process_nodsp(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer){
    //TODO: a bypass on demodulator call would be more efficient
    std::memcpy(*output_buffer, *input_buffer, parameters->buffer_len*sizeof(float2));
    return parameters->buffer_len;
}

//process a packet demodulating with chirp
int RX_buffer_demodulator::process_chirp(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer){

    int valid_size;

    //upload the input buffer
    cudaMemcpyAsync(input, *input_buffer, parameters->buffer_len*sizeof(float2),cudaMemcpyHostToDevice, internal_stream);

    //zero_mem<<<1024,64,0,internal_stream>>>(input,parameters->buffer_len,1);

    //demodulate the chirp signal. NOTE: the index bookeeping is done on the device
    chirp_demodulator_wrapper(input,output+spare_size,parameters->buffer_len,last_index,d_parameter,internal_stream);

    //update bookeeping index
    last_index = (last_index + parameters->buffer_len) % (h_parameter.num_steps * h_parameter.length);


    //gpuErrchk( cudaStreamSynchronize(internal_stream) );
    if(decimator_active){

        valid_size = vna_helper->valid_size;

        cublas_decim(output,decim_output,profile,&zero,&one,valid_size,ppt, &handle);
        //gpuErrchk( cudaStreamSynchronize(internal_stream) );

        //download the result on the host
        cudaMemcpyAsync(*output_buffer, decim_output, sizeof(float2)*valid_size, cudaMemcpyDeviceToHost, internal_stream);

        spare_size = vna_helper->new0;

        //gpuErrchk( cudaStreamSynchronize(internal_stream) );
        //move the part of the buffer that has already to be analyzed at the begin of the buffer
        if(spare_size > 0){
            move_buffer_wrapper(
                output,  //from
                output,  //to
                vna_helper->new0, //size
                vna_helper->spare_begin,0,
                internal_stream); // destination offset
        }

        vna_helper->update();

    }else{

        //download back to the host
        cudaMemcpyAsync(*output_buffer, output, sizeof(float2)*parameters->buffer_len, cudaMemcpyDeviceToHost, internal_stream);
        //cudaMemcpyAsync(*output_buffer, input, sizeof(float2)*parameters->buffer_len, cudaMemcpyDeviceToHost, internal_stream);
        //set the valid putput size
        valid_size = parameters->buffer_len;
    }

    cudaStreamSynchronize(internal_stream);
    //wait for operation to be completed before returning

    return valid_size;
}

// @todo this function has to be splitted in decimated and undecimated case. branching is not ok here.
int RX_buffer_demodulator::process_direct(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer){
  int ret;
  size_t output_channel_len = parameters->buffer_len/std::max((int)(parameters->decim),1);
  //Load the memory
  cudaMemcpyAsync(direct_input, *input_buffer, parameters->buffer_len*sizeof(float2),cudaMemcpyHostToDevice, internal_stream);
  //std::cout<< "total length: "<<DIRECT_output_size<<" single tone length: "<<parameters->buffer_len <<std::endl;
  //Call the kernel
  direct_demodulator_wrapper(
      DIRECT_tone_frquencies,
      DIRECT_tone_phases,
      parameters->rate,
      DIRECT_current_index,
      parameters->buffer_len,
      DIRECT_output_size,
      direct_input,
      direct_output,
      internal_stream
  );

  //Apply FIR filtering to each channel
  if (parameters->decim>0){
      for(size_t i = 0; i< parameters->wave_type.size(); i++) DIRECT_FIR[i]->run_fir((direct_output)+(i*parameters->buffer_len), (FIR_output)+i*output_channel_len);
      checkcublas(cublasCgeam(
        handle,
        CUBLAS_OP_T, //op A
        CUBLAS_OP_N, //op B
        parameters->wave_type.size(), //m
        output_channel_len, //n
        &onef, //aplha
        FIR_output,output_channel_len, //A
        &zerof, //beta
        direct_input,output_channel_len, //B not important as beta == 0
        transposed,(int)parameters->wave_type.size() //C
      ));
  }

  //Update bookeeping
  DIRECT_current_index+=parameters->buffer_len;

  //Numerically control this value: if too big adds noise in _sinf()
  DIRECT_current_index = DIRECT_current_index % parameters->rate;

  if (parameters->decim<=0){
    //cudaMemcpyAsync(*output_buffer, direct_output, sizeof(float2)*DIRECT_output_size, cudaMemcpyDeviceToHost, internal_stream);
    checkcublas(cublasCgeam(
      handle,
      CUBLAS_OP_T, //op A
      CUBLAS_OP_N, //op B
      parameters->wave_type.size(), //m
      parameters->buffer_len, //n
      &onef, //aplha
      direct_output,parameters->buffer_len, //A
      &zerof, //beta
      direct_input,parameters->buffer_len, //B not important as beta == 0
      transposed,(int)parameters->wave_type.size() //C
    ));
    cudaMemcpyAsync(*output_buffer, transposed, sizeof(float2)*DIRECT_output_size, cudaMemcpyDeviceToHost, internal_stream);
    ret =  DIRECT_output_size;
  }else{
    ret = DIRECT_output_size/parameters->decim;
    cudaMemcpyAsync(*output_buffer, transposed, sizeof(float2)*ret, cudaMemcpyDeviceToHost, internal_stream);
  }
  cudaStreamSynchronize(internal_stream);
  return ret;
}

void RX_buffer_demodulator::close_direct(){
  cudaStreamDestroy(internal_stream);
  cudaFree(DIRECT_tone_frquencies);
  cudaFree(DIRECT_tone_phases);
  cudaFree(direct_input);
  cudaFree(direct_output);
  cudaFree(transposed);
  free(DIRECT_tones);
  if (parameters->decim>0){
      for(size_t k=0; k<parameters->wave_type.size(); k++)delete(DIRECT_FIR[k]);
      free(DIRECT_FIR);
      free(fir_taps);
      cudaFree(FIR_output);
  }
  return;
}

//process a packet with the pfb and set the variables for the next
// returns the valid length of the output packet
int RX_buffer_demodulator::process_pfb(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer){

    int output_buffer_valid_len;

    //upload the input buffer in a position determined by the buffer helper
    cudaMemcpyAsync(
        raw_input+buf_setting->new_0,   //device address
        *input_buffer,                  //host address
        parameters->buffer_len*sizeof(float2),
        cudaMemcpyHostToDevice,internal_stream);

    //apply the polyphase filter to the buffer
    polyphase_filter_wrapper(raw_input,input,d_params,internal_stream);

    //execute the fft
    cufftExecC2C(plan, input, output+spare_size, CUFFT_FORWARD);

    //move the part of the buffer that has already to be analyzed at the begin of the buffer
    move_buffer_wrapper(
        raw_input,  //from
        raw_input,  //to
        buf_setting->spare_samples, //size
        buf_setting->spare_begin,0,
        internal_stream); // destination offset

    if(decimator_active){

        pfb_decim_helper->update(buf_setting->current_batch);
        spare_size = pfb_decim_helper->new_0;

        decimate_pfb(output,decim_output,parameters->fft_tones,parameters->decim,pfb_decim_helper->out_size,internal_stream);

        //move the spare buffer
        move_buffer_wrapper(
            output,  //from
            output,  //to
            pfb_decim_helper->new_0, //size
            pfb_decim_helper->out_size,0,
            internal_stream); // destination offset

        //move the selected tones in an other buffer excluding exceeding samples
        tone_select_wrapper(output, reduced_output, d_params, std::floor(buf_setting->current_batch/(float)parameters->decim),internal_stream);

        //download the result in the host memory
        cudaMemcpyAsync(*output_buffer,reduced_output,
            parameters->wave_type.size()*batching*sizeof(float2),
            cudaMemcpyDeviceToHost,internal_stream);

        output_buffer_valid_len = parameters->wave_type.size()*std::floor(buf_setting->current_batch/(float)parameters->decim);

    }else{

        //move the selected tones in an other buffer excluding exceeding samples
        tone_select_wrapper(output, reduced_output, d_params, buf_setting->current_batch,internal_stream);

        //download the result in the host memory
        cudaMemcpyAsync(*output_buffer,reduced_output,
            parameters->wave_type.size()*batching*sizeof(float2),
            cudaMemcpyDeviceToHost,internal_stream);

        output_buffer_valid_len = parameters->wave_type.size()*buf_setting->current_batch;
        //print_debug("source batching is: ",buf_setting->current_batch);
    }


    //update the buffer helper
    buf_setting->update();

    //wait for operation to be completed before returning
    cudaStreamSynchronize(internal_stream);

    //fwrite(*output_buffer, output_buffer_valid_len * sizeof(float2), 1, raw_out_file);


    //NOTE: this variable has been calculated before the buffer helper update
    //is needed to know how many samples to stream
    return output_buffer_valid_len;


}

//same process as the pfb but there is no tone selection and the buffer is bully downloaded
int RX_buffer_demodulator::process_pfb_spec(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer){

    int output_buffer_valid_len;

    //upload the input buffer in a position determined by the buffer helper
    cudaMemcpyAsync(
        raw_input+buf_setting->new_0,   //device address
        *input_buffer,                  //host address
        parameters->buffer_len*sizeof(float2),
        cudaMemcpyHostToDevice,internal_stream);

    //apply the polyphase filter to the buffer
    polyphase_filter_wrapper(raw_input,input,d_params,internal_stream);

    //execute the fft
    cufftExecC2C(plan, input, output+pfb_out, CUFFT_FORWARD);

    //move the part of the buffer that has already to be analyzed at the begin of the buffer
    move_buffer_wrapper(
        raw_input,  //from
        raw_input,  //to
        buf_setting->spare_samples, //size
        buf_setting->spare_begin,0,
        internal_stream); // destination offset

    if(decimator_active){

        /*
        pfb_decim_helper->update(buf_setting->current_batch);
        spare_size = pfb_decim_helper->new_0;
        //std::cout<<"new_0: "<<spare_size<<std::endl<< "current_batch"<<buf_setting->current_batch<<std::endl<<"out_size"<<pfb_decim_helper->out_size<<std::endl;
        decimate_pfb(output,decim_output,parameters->decim,parameters->fft_tones,pfb_decim_helper->out_size,internal_stream);
        */

        //std::cout<<"Input len is: "<<buf_setting->spare_begin<<std::endl;
        output_len = parameters->fft_tones*int((buf_setting->spare_begin/parameters->fft_tones)/(float)parameters->decim);
        int input_len = output_len * parameters->decim;
        //std::cout<<"Output len will be: "<<output_len<<std::endl;
        decimate_spectra( output, decim_output, parameters->decim, parameters->fft_tones, input_len, output_len, internal_stream);
        //move the spare buffer
        //std::cout<<"Resifual buffer is: "<<buf_setting->spare_begin - input_len    <<std::endl;
        move_buffer_wrapper(
            output,  //from
            output,  //to
            buf_setting->spare_begin - input_len, //size
            input_len,0,
            internal_stream); // destination offset


        //download the result in the host memory
        cudaMemcpyAsync(*output_buffer,decim_output,
            buf_setting->copy_size*sizeof(float2),
            cudaMemcpyDeviceToHost,internal_stream);

        //get the valid length for this buffer before updating the buffer helper but with decimation applied
        //output_buffer_valid_len = pfb_decim_helper->out_size;
        output_buffer_valid_len = output_len;

    }else{


        //download the result in the host memory
        cudaMemcpyAsync(*output_buffer,output,
            buf_setting->copy_size*sizeof(float2),
            cudaMemcpyDeviceToHost,internal_stream);

        //std::cout<<"Quick check on rx buffer"<<std::endl;
        //for(int i = 0; i < 10; i++)std::cout<< i<<"\t"<<(*output_buffer)[i].x*1e7<<" +j*"<<(*output_buffer)[i].y*1e7<<std::endl;

        //get the valid length for this buffer before updating the buffer helper
        output_buffer_valid_len = buf_setting->copy_size;
    }

    //update the buffer helper
    buf_setting->update();

    //wait for operation to be completed before returning
    cudaStreamSynchronize(internal_stream);

    //NOTE: this variable has been calculated befor the buffer helper update
    return output_buffer_valid_len;
}
//clean up the pfb allocations
void RX_buffer_demodulator::close_pfb(){
    cufftDestroy(plan);
    cudaStreamDestroy(internal_stream);
    cudaFree(d_params);
    cudaFree(input);
    cudaFree(output);
    cudaFree(reduced_output);
    cudaFree(raw_input);
    delete(buf_setting);
    if(decimator_active){
        delete(pfb_decim_helper);
        cudaFree(decim_output);
    }
}

//clean up the pfb full spectrum
void RX_buffer_demodulator::close_pfb_spec(){
    cufftDestroy(plan);
    cudaStreamDestroy(internal_stream);
    cudaFree(d_params);
    cudaFree(input);
    cudaFree(output);
    cudaFree(raw_input);
    delete(buf_setting);
    if(decimator_active){
        delete(pfb_decim_helper);
        cudaFree(decim_output);
    }
}

//clean up the chirp demod allocation
void RX_buffer_demodulator::close_chirp(){
    cudaStreamDestroy(internal_stream);
    cudaFree(d_parameter);
    cudaFree(input);
    cudaFree(output);
    if(decimator_active){
        cudaFree(decim_output);
        delete(vna_helper);
    }
}

void RX_buffer_demodulator::close_nodsp(){cudaStreamDestroy(internal_stream);}

//converts general purpose parameters into kernel wrapper parameters on gpu.
//THIS ONLY TAKES CARE OF MULTI TONES MEASUREMENT
void RX_buffer_demodulator::upload_multitone_parameters(){

    //calculate the maximum batching. The number 3 at the end of the formula is empirical
    // batching = std::floor((float)(parameters->buffer_len - parameters->fft_tones*parameters->pf_average)/(float)parameters->fft_tones) +3*parameters->fft_tones*parameters->pf_average;
    batching = std::ceil((float)parameters->buffer_len/(float)parameters->fft_tones) + parameters->pf_average + 5;


    //copy-paste parameters
    h_param.window = window;
    h_param.length = parameters->buffer_len;
    h_param.n_tones = parameters->fft_tones;
    h_param.average_buffer = parameters->pf_average;
    h_param.batching = batching;

    //tones to be downloaded from the gpu at the end of a single packet analysis process
    h_param.eff_n_tones = parameters->wave_type.size();

    int *tone_bins;
    tone_bins = (int*)malloc(h_param.eff_n_tones*sizeof(int));

    std::vector<double> bin_axis(parameters->fft_tones);

    double bin_size = (double)parameters->rate/(double)parameters->fft_tones;

    for(size_t i = 0; i<bin_axis.size(); i++){
      bin_axis[i] = i*bin_size - bin_size * (parameters->fft_tones/2);
      for(int u = 0; u<h_param.eff_n_tones; u++){
        if((parameters->freq[u] < bin_axis[i] + bin_size) && (parameters->freq[u] > bin_axis[i] - bin_size) ){
            tone_bins[u] = (i + (parameters->fft_tones/2))%parameters->fft_tones;
            //std::cout<<"parameter f: "<<parameters->freq[u]<<" goes in bin: "<<tone_bins[u]<<std::endl;
          }
        }
    }
    //convert the frequency parameter to fft bin
    /*
    for(int u = 0; u<h_param.eff_n_tones;u++){

        tone_bins[u] = parameters->freq[u]>0?
            round((double)parameters->fft_tones * (double)parameters->freq[u]/(double)parameters->rate):
            round((double)parameters->fft_tones*((double)1.-(double)parameters->freq[u]/(double)parameters->rate));
        std::cout<<"parameter f: "<<parameters->freq[u]<<" goes in bin: "<<tone_bins[u]<<std::endl;
    }
    */
     //allocate memory for the tone device pointer array
    cudaMalloc((void **)&h_param.tones, h_param.eff_n_tones*sizeof(int));

    // Copy host memory to device
    cudaMemcpy(h_param.tones, tone_bins, h_param.eff_n_tones*sizeof(int),cudaMemcpyHostToDevice);

    //in case of diagnostic, print frequency - bin table
    if(diagnostic){
        std::stringstream ss;
        ss<<"Polyphase filter bank diagnostic:"<<std::endl<<"frequ\tbin"<<std::endl;
        for(int u = 0; u<h_param.eff_n_tones;u++)ss<< parameters->freq[u]<<"\t"<<tone_bins[u]<<std::endl;
        std::cout<<ss.str()<<std::endl;
    }

    //cleanup the host version
    free(tone_bins);

    //allocates spaces for the gpu copy of the parameter struct
    cudaMalloc((void **)&d_params, sizeof(filter_param));

    // Copy the parameters to device
    cudaMemcpy(d_params, &h_param, sizeof(filter_param),cudaMemcpyHostToDevice);

}
