#include "kernels.cuh"

__global__ void DIRECT_decimator(
  uint single_tone_length,
  size_t total_length,
  float2* __restrict intput,
  float2* __restrict output
);

//Direct demodulation kernel. This kernel takes the raw input from the SDR and separate channels. Note: does not do any filtering.
__global__ void direct_demodulator_fp64(
  double* __restrict tone_frquencies,
  size_t index_counter,
  uint single_tone_length,
  size_t total_length,
  float2* __restrict intput,
  float2* __restrict output
){
    double _i,_q;
    double tone_calculated_phase;
    uint input_index;

    for(uint i = blockIdx.x * blockDim.x + threadIdx.x;
        i < total_length;
        i += gridDim.x*blockDim.x
    ){

        input_index = i % single_tone_length;

        //here's the core: reading from the pointer should be automatically cached.
        tone_calculated_phase = 2. * tone_frquencies[i/single_tone_length] * (index_counter + input_index);

        //generate sine and cosine
        sincospi(tone_calculated_phase,&_q,&_i);

        //demodulate
        output[i].x = intput[input_index].x * _i + intput[input_index].y * _q;
        output[i].y = intput[input_index].y * _i - intput[input_index].x * _q;

  }

}


__global__ void direct_demodulator_integer(
  int* __restrict tone_frequencies,
  int* __restrict tone_phases,
  int wavetablelen,
  size_t index_counter,
  size_t single_tone_length,
  size_t total_length,
  float2* __restrict input,
  float2* __restrict output
){
    double _i,_q;
    double tone_calculated_phase;
    size_t input_index;

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < total_length;
        i += gridDim.x*blockDim.x
    ){
        input_index = i % single_tone_length;
        size_t ch = i / single_tone_length;
        long long int tf = tone_frequencies[ch];
        long long int ii = (input_index + index_counter)%wavetablelen;
        long long int tp = tone_phases[ch];
        long long int my_phase = (tp + (tf * ii)%wavetablelen);
        tone_calculated_phase = 2. * (my_phase / (double)wavetablelen);

        //generate sine and cosine
        sincospi(tone_calculated_phase,&_q,&_i);

        //demodulate (strided acces is due to the way python inteprets packets)
        //output[input_index*2+ch].y = input_index;//input[input_index].y * _i - input[input_index].x * _q;
        //output[input_index*2+ch].x = tf;//input[input_index].x * _i + input[input_index].y * _q;

        //multichannel diagnostic
        //output[i].y = input_index;
        //output[i].x = tf;

        output[i].y = input[input_index].y * _i - input[input_index].x * _q;
        output[i].x = input[input_index].x * _i + input[input_index].y * _q;

      }
}

//Wrapper for the direct demodulation
void direct_demodulator_wrapper(
  int* __restrict tone_frequencies,
  int* __restrict tone_phases,
  int wavetablelen,
  size_t index_counter,
  size_t single_tone_length,
  size_t total_length,
  float2* __restrict input,
  float2* __restrict output,
  cudaStream_t internal_stream){

    direct_demodulator_integer<<<1024,32,0,internal_stream>>>(tone_frequencies,tone_phases,wavetablelen,index_counter,single_tone_length,total_length,input,output);
}

//allocates memory on gpu and fills with a real hamming window. returns a pointer to the window on the device.
//note that this is a host function that wraps some device calls
//TODO: the window function should be made in a class not in a function. Once we have the class we can put the class template directly in the header file to avoid undefined reference to specialized templates during linking.
template <typename T>
T* make_hamming_window(int length, int side, bool diagnostic, bool host_ret){

    T *d_win,*h_win = (T*)malloc(length*sizeof(T));

    //allocate some memory on the GPU
    cudaMalloc((void **)&d_win, length*sizeof(T));


    //initialize the accumulator used for normalization
    float scale = 0;

    for(int i = 0; i < side; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = length - side; i < length; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = 0; i < length - side; i++){
        h_win[i+side].y = 0;

        //make hamming
        h_win[i+side].x = (0.54-0.46*cos(2.f*pi_f*i/((length-side)-1)));
        scale += h_win[i+side].x;

    }
    //normalize the window
    for(int i = 0; i < length; i++) h_win[i].x /= scale;

    //upload the window on the GPU
    cudaMemcpy(d_win, h_win, length*sizeof(T),cudaMemcpyHostToDevice);

    if(diagnostic){
        //write a diagnostic binary file containing the window.
        //TODO there hsould be a single hdf5 file containing all the diagnostic informations
        FILE* window_diagnostic = fopen("USRP_hamming_filter_window.dat", "wb");
        fwrite(static_cast<void*>(h_win), length, sizeof(T), window_diagnostic);
        fclose(window_diagnostic);
    }

    //cleanup
    free(h_win);
    cudaDeviceSynchronize();
    return d_win;
}
//specializing the template to avoid error during the linking process
template <>
float2* make_hamming_window<float2>(int length, int side, bool diagnostic, bool host_ret){

    float2 *ret,*d_win,*h_win = (float2*)malloc(length*sizeof(float2));

    //initialize the accumulator used for normalization
    float scale = 0;

    for(int i = 0; i < side; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = length - side; i < length; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = 0; i < length - side; i++){
        h_win[i+side].y = 0;

        //make hamming
        h_win[i+side].x = (0.54-0.46*cos(2.f*pi_f*i/((length-side)-1)));
        scale += h_win[i+side].x;

    }

    //normalize the window
    for(int i = 0; i < length; i++) h_win[i].x /= scale;

    if(diagnostic){
        //write a diagnostic binary file containing the window.
        //TODO there hsould be a single hdf5 file containing all the diagnostic informations
        FILE* window_diagnostic = fopen("USRP_hamming_filter_window.dat", "wb");
        fwrite(static_cast<void*>(h_win), length, sizeof(float2), window_diagnostic);
        fclose(window_diagnostic);
    }
    if (not host_ret){

        //allocate some memory on the GPU
        cudaMalloc((void **)&d_win, length*sizeof(float2));

        //upload the window on the GPU
        cudaMemcpy(d_win, h_win, length*sizeof(float2),cudaMemcpyHostToDevice);

        //cleanup
        free(h_win);
        cudaDeviceSynchronize();
        ret =  d_win;
    }else{
        ret = h_win;
    }

    return ret;
}

float2* make_flat_window(int length, int side, bool diagnostic){

    float2 *d_win,*h_win = (float2*)malloc(length*sizeof(float2));

    //allocate some memory on the GPU
    cudaMalloc((void **)&d_win, length*sizeof(float2));


    //initialize the accumulator used for normalization
    float scale = 0;

    for(int i = 0; i < side; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = length - side; i < length; i++){
        h_win[i].y = 0;
        h_win[i].x = 0;
    }
    for(int i = 0; i < length - side; i++){
        //make flat
        h_win[i+side].x = 1.;
        h_win[i+side].y = 0.;
        scale += h_win[i+side].x;

    }

    //normalize the window
    for(int i = 0; i < length; i++) h_win[i].x /= scale;

    //upload the window on the GPU
    cudaMemcpy(d_win, h_win, length*sizeof(float2),cudaMemcpyHostToDevice);

    if(diagnostic){
        //write a diagnostic binary file containing the window.
        //TODO there hsould be a single hdf5 file containing all the diagnostic informations
        FILE* window_diagnostic = fopen("USRP_hamming_filter_window.dat", "wb");
        fwrite(static_cast<void*>(h_win), length, sizeof(float2), window_diagnostic);
        fclose(window_diagnostic);
    }

    //cleanup
    free(h_win);
    cudaDeviceSynchronize();
    return d_win;
}


//allocates memory on gpu and fills with a real sinc window. returns a pointer to the window on the device.
//note that this is a host function that wraps some device calls
float2* make_sinc_window(int length, float fc, bool diagnostic = false, bool host_ret = false){

    float2 *ret,*d_win,*h_win = (float2*)malloc(length*sizeof(float2));

    int sinc_index;

    //initialize the accumulator used for normalization
    float scale = 0;
    for(int i = 0; i < length; i++){

        sinc_index = i - (length-1)/2;

        //no immaginary part, the window is considered purely real
        h_win[i].y = 0;

        //generate sinc
        sinc_index != 0 ?
            h_win[i].x = (2.f*fc)*sin(2.f*pi_f*fc*sinc_index)/(2.f*pi_f*fc*sinc_index):
            h_win[i].x = (2.f*fc);

        //apply hamming
        h_win[i].x *= (0.54-0.46*cos(2.f*pi_f*i/(length-1)));

        scale += h_win[i].x;

    }

    //normalize the window
    for(int i = 0; i < length; i++) h_win[i].x /= scale;



    if(diagnostic){
        //write a diagnostic binary file containing the window.
        //TODO there hsould be a single hdf5 file containing all the diagnostic informations
        FILE* window_diagnostic = fopen("USRP_polyphase_filter_window.dat", "wb");
        fwrite(static_cast<void*>(h_win), length, sizeof(float2), window_diagnostic);
        fclose(window_diagnostic);
    }
    if(not host_ret){
      //allocate some memory on the GPU
      cudaMalloc((void **)&d_win, length*sizeof(float2));
      //upload the window on the GPU
      cudaMemcpy(d_win, h_win, length*sizeof(float2),cudaMemcpyHostToDevice);
      //cleanup
      free(h_win);
      cudaDeviceSynchronize();
      ret = d_win;
    }else{
      ret = h_win;
    }
    return ret;
}

__global__ void init_states(curandState *state, int twice_vector_len){

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < twice_vector_len;
        i += gridDim.x*blockDim.x
        ){

        curand_init(1337,i,0,&state[i]);
    }
}

__global__ void make_rand(curandState *state, float2 *vector, int len, float scale = 1){

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < len;
            i += gridDim.x*blockDim.x
            ){
        vector[i].x = scale * 2*curand_uniform(&state[2*i])-1;
        vector[i].y = scale * 2*curand_uniform(&state[2*i]+1)-1;
    }
}


void print_chirp_params(std::string comment, chirp_parameter cp){

}

__device__ float modulus(float number, float modulus){
    return number - __float2int_rd(number/modulus)*modulus;
}

__device__ unsigned int round_index(unsigned int last_index, unsigned int offset, unsigned int num_f, unsigned int f_len){

    unsigned long int pos = last_index + offset;

    unsigned long int chirp_len = f_len * num_f;

    return  (pos - ((pos/chirp_len) * chirp_len));//pos%chirp_len;

}


//generate a chirp waveform in a gpu buffer
__global__ void chirp_gen(

    float2* __restrict__ output, //pointer to the gpu buffer
    unsigned int output_size, //size of the buffer
    chirp_parameter* __restrict__ info, //chirp information
    unsigned long int last_index,
    float scale = 1 //scale the amplitude of the chirp

    ){
    unsigned long int effective_index; //index relative to the current position in signal generation (a single chirp may be generated in more than one kernel call).
    unsigned long int frequency_index; //actual frequency step in the chirp generation.
    unsigned long int phase_correction; //phase correction to allow parallel coherent phase generation.
    int index; //index to use in sinus and cosinus.
    for(unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
                offset < output_size;
                offset += gridDim.x*blockDim.x
        ){

        //take in account previous calls and bookeep phase
        effective_index = (last_index + offset) % (info->num_steps * info->length);
        //effective_index = round_index(info->last_index,offset,info->num_steps,info->length);
        //effective_index = round_index

        //calculate current frequency to generate
        frequency_index = effective_index/info->length;

        unsigned long int q_phase = (frequency_index/2)*(frequency_index +1) + (frequency_index % 2)*((frequency_index +1)/2);

        //correct the pahse. needed for parallel chirp generation.
        phase_correction = ( info->chirpness * (info->length * q_phase) );

        //evaluate sine index
        index =  (effective_index * (info->f0 + frequency_index * info->chirpness ) - phase_correction);

        output[offset].x = sinpi(((double)(index)/2147483647.5))*scale;
        output[offset].y = -cospi(((double)(index)/2147483647.5))*scale;

    }

}
void chirp_gen_wrapper(

    float2* __restrict__ output, //pointer to the gpu buffer
    unsigned int output_size, //size of the buffer
    chirp_parameter* __restrict__ info, //chirp information
    unsigned long int last_index,
    cudaStream_t internal_stream,
    float scale = 1 //scale the amplitude of the chirp

    ){

    chirp_gen<<<1024,32,0,internal_stream>>>(output,output_size,info,last_index,scale);

}


__global__ void chirp_demodulator(
    float2* __restrict__ input,  //pointer to the input buffer
    float2* __restrict__ output, //pointer to the gpu buffer
    unsigned int output_size, //size of the buffers
    unsigned long int last_index,
    chirp_parameter* __restrict__ info //chirp information
    ){
    unsigned long int effective_index; //index relative to the current position in signal generation (a single chirp may be generated in more than one kernel call).
    unsigned long int frequency_index; //actual frequency step in the chirp generation.
    unsigned long int phase_correction; //phase correction to allow parallel coherent phase generation.
    int index; //index to use in sinus and cosinus.
    float2 chirp; //calculate the chirp
    for(unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
                offset < output_size;
                offset += gridDim.x*blockDim.x
        ){

        //take in account previous calls and bookeep phase
        effective_index = (last_index + offset) % (info->num_steps * info->length);
        //effective_index = round_index(info->last_index,offset,info->num_steps,info->length);

        //calculate current frequency to generate
        frequency_index = effective_index/info->length;

        unsigned long int q_phase = (frequency_index/2)*(frequency_index +1) + (frequency_index % 2)*((frequency_index +1)/2);

        //correct the pahse. needed for parallel chirp generation.
        phase_correction = ( info->chirpness * (info->length * q_phase) );

        //evaluate sine index
        index =  (effective_index * (info->f0 + frequency_index * info->chirpness ) - phase_correction) ;

        chirp.x =  sinpi(((double)(index)/2147483647.5));
        chirp.y =  -cospi(((double)(index)/2147483647.5));

        output[offset].x = chirp.x*input[offset].x + chirp.y*input[offset].y;
        output[offset].y = chirp.x*input[offset].y - chirp.y*input[offset].x;
    }
}
__device__ float absolute(float2 number){return sqrt(number.x*number.x+number.y+number.y);}

void chirp_demodulator_wrapper(
    float2* __restrict__ input,  //pointer to the input buffer
    float2* __restrict__ output, //pointer to the gpu buffer
    unsigned int output_size, //size of the buffers
    unsigned long int last_index,
    chirp_parameter* __restrict__ info, //chirp information
    cudaStream_t internal_stream
    ){

    chirp_demodulator<<<1024,32,0,internal_stream>>>(input,output,output_size,last_index,info);

}


__global__ void move_buffer(
    float2* __restrict__ from,
    float2* __restrict__ to,
    int size,
    int from_offset,
    int to_offset
    ){

    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
            offset < size;
            offset += gridDim.x*blockDim.x
            ){
        to[offset+to_offset] = from[offset+from_offset];
        //if(absolute(from[offset])==0)printf("0 found in %d\n",offset);
    }
}
void move_buffer_wrapper(
    float2* __restrict__ from,
    float2* __restrict__ to,
    int size,
    int from_offset,
    int to_offset,
    cudaStream_t internal_stream
    ){
    move_buffer<<<1024,64,0,internal_stream>>>(from,to,size,from_offset,to_offset);

}


//kernel used to apply the polyphase filter to a buffer using a window
__global__ void polyphase_filter(
    float2* __restrict__ input,
    float2* __restrict__ output,
    filter_param* __restrict__ filter_info
    ){

    //loop over the entire device buffer (last samples may be meaningless but this is accounted in the host loop)
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
            offset < filter_info->batching * filter_info->n_tones;
            offset += gridDim.x*blockDim.x
        ){

        //check if there are enough samples for averaging
        if(offset + filter_info->n_tones*(filter_info->average_buffer) < filter_info->batching * filter_info->n_tones){

            //accumulate average in an initialized register
            float2 acc;
            acc.x=0;
            acc.y=0;

            //loop over the sample to average and add to accumulator. NOTE: the register acc is private to the thread.
            for(int i = 0; i<(filter_info->average_buffer); i++ ){

                //calculate the index of the sample to average
                int sample_index = offset + i* (filter_info->n_tones);

                //calculate the corresponding window sample
                int win_index = offset%filter_info->n_tones+i*filter_info->n_tones;

                //apply the window and accumulate. NOTE the window is considered purely REAL
                acc.x += input[sample_index].x * (filter_info->window)[win_index].x;
                acc.y += input[sample_index].y * (filter_info->window)[win_index].x;
            }

            //last averaging step NO because it's a normalized window
            //acc.x = acc.x/filter_info->average_buffer;
            //acc.y = acc.y/filter_info->average_buffer;

            //finally write the filtered sample to the output buffer
            output[offset] = acc;
        }
    }
}

void polyphase_filter_wrapper(
    float2* __restrict__ input,
    float2* __restrict__ output,
    filter_param* __restrict__ filter_info,
    cudaStream_t internal_stream
    ){

    polyphase_filter<<<896*2,64,0,internal_stream>>>(input,output,filter_info);
}



//select the tones from the fft result and reorder them in a new buffer
__global__ void tone_select(
    float2* __restrict__ input, //must be the fft output
    float2* __restrict__ output,//the buffer that will then be downloaded to host
    filter_param* __restrict__ filter_info, //information about the filtering process
    int effective_batching //how many samples per tone have been effectively calculated
    ){

    //filter_info->eff_n_tones is the number of 'selected' tones
    //filter_info->n_tones is te number of fft bins
    //effective_batching counts how many fft's are present in the buffer
    //filter_info->tones has the information about which are the selected tones
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
            offset < effective_batching*filter_info->eff_n_tones;
            offset += gridDim.x*blockDim.x
            ){

            //calculate from where to take the sample
            int index = (offset/filter_info->eff_n_tones)*filter_info->n_tones + filter_info->tones[offset % filter_info->eff_n_tones];

            //write the sample in the output buffer
            output[offset] = input[index];

    }
}

void tone_select_wrapper(
    float2* __restrict__ input, //must be the fft output
    float2* __restrict__ output,//the buffer that will then be downloaded to host
    filter_param* __restrict__ filter_info, //information about the filtering process
    int effective_batching, //how many samples per tone have been effectively calculated
    cudaStream_t internal_stream
    ){

    tone_select<<<1024,64,0,internal_stream>>>(input,output,filter_info,effective_batching);

}


//scale a float2 buffer for a float scalar
__global__ void scale_buffer(
    float2* __restrict__ input,
    int input_size,
    float scale
    ){

    //just loop over the array and multiply both component
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
            offset < input_size;
            offset += gridDim.x*blockDim.x
            ){

            input[offset].x = input[offset].x * scale;
            input[offset].y = input[offset].y * scale;
    }
}

//generate a set of tones and return host pointer to the buffer unless the device option is true.
//NOTE the length of the buffer is the sampling_rate
float2* tone_gen(
    tone_parameters* info, //tone information (all host side)
    int sampling_rate,
    float scale, //scale the whole buffer (all tones) for a scalar
    bool device //the function return device buffer pointer instead
    ){

    //base for the fft. will be used as buffer recipient later.
    float2* base_vector;
    base_vector = (float2*)malloc(sampling_rate*sizeof(float2));

    //set the cuda fft plan
    cufftHandle plan;
    if (cufftPlan1d(&plan, sampling_rate, CUFFT_C2C, 1) != CUFFT_SUCCESS){
        //print_error("Cannot allocate memory on the gpu for tone generation.");
        std::cout<<"CUDA ERROR IN cufftHandle"<<std::endl;
        return NULL;
    }

    //device base for the fft. NOTE in place transformation applies.
    float2* device_base;
    cudaMalloc((void **)&device_base, sampling_rate*sizeof(float2));

    //accumulator for normalization of the buffer
    float normalization = 0;

    //zero the host memory
    for(int i = 0; i < sampling_rate; i++){
        base_vector[i].x = 0;
        base_vector[i].y = 0;
    }

    //set the tones in the host base vector
    for(int i = 0; i < info->tones_number; i++){

        //rotate frequencies in case of negative frequency offset
        int freq;
        info->tone_frquencies[i] > 0 ?
            freq = info->tone_frquencies[i]:
            freq = sampling_rate + info->tone_frquencies[i];

        //set the corresponding amplitude (NOTE this only work if fft_length == sampling_rate)
        base_vector[freq].x = info->tones_amplitudes[i];// * std::cos(i * pi_f/info->tones_number);

        //all same phase for now (or distributed to avoid power spikes? NO?)
        base_vector[freq].y = 0;//info->tones_amplitudes[i] * std::sin(i * pi_f/info->tones_number);

        //add tone amplitude to normalixation accumulator
        normalization += info->tones_amplitudes[i];
    }

    //finalize normaization coefficient calculation
    //normalization = 1./normalization;

    //upload in the device the host base vector
    cudaMemcpy(device_base, base_vector, sampling_rate*sizeof(float2),cudaMemcpyHostToDevice);

    //execute the inverse FFT transform
    if (cufftExecC2C(plan, device_base, device_base, CUFFT_INVERSE) != CUFFT_SUCCESS){
	    //print_error("Cannot execute fft transform for tone generation.");
	    std::cout<<"CUDA ERROR: Cannot execute fft transform for tone generation."<<std::endl;
        return NULL;
    }

    //apply normalization to the device buffer
    //scale_buffer<<<1024, 32>>>(device_base, sampling_rate, normalization);

    //if the user set a scale, apply scalar multiplication
    if(scale>1.) std::cout<<"CUDA WARNING: Maximum amplitude of the TX buffer is > 1."<<std::endl;//print_warning("Maximum amplitude of the TX buffer is > 1.");
    if(scale!=1.) scale_buffer<<<1024, 32>>>(device_base, sampling_rate, scale);

    //download the buffer from gpu to host
    cudaMemcpy(base_vector,device_base,sampling_rate*sizeof(float2),cudaMemcpyDeviceToHost);

    //clean the GPU fft plan
    cufftDestroy(plan);

    //if this option is true, the function returns the device pointer instead of the host pointer
    if(device){

        //clean the host buffer
        free(base_vector);

        //return device pointer
        return device_base;

    }else{

        //clean the GPU buffer
        cudaFree(device_base);

        //return the pointer
        return base_vector;

    }
}

//overlap two device buffer in one device buffer.
__global__ void mix_buffers(
        float2* __restrict__ buffer1,
        float2* __restrict__ buffer2,
        float2* __restrict__ output,
        int length
    ){
    //loop over the buffers.
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
            offset < length;
            offset += gridDim.x*blockDim.x
            ){
        output[offset].x = buffer1[offset].x + buffer2[offset].x;
        output[offset].y = buffer1[offset].y + buffer2[offset].y;
    }
}


__global__ void average_spectra(
        float2* __restrict__ input,
        float2* __restrict__ output,
        int decim,
        int nfft,
        int input_len

    ){
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
        offset < input_len;
        offset += gridDim.x*blockDim.x
        ){

        int output_offset = offset%nfft + nfft*int(offset/(nfft*decim));
        atomicAdd(&output[output_offset].x, input[offset].x);
        atomicAdd(&output[output_offset].y, input[offset].y);


    }

}

void decimate_spectra(
        float2* __restrict__ input, //output of the pfb
        float2* __restrict__ output,//decimated output
        int decim,                  //decimation factor (multiplicative to the pfb one)
        int nfft,                   //length of the fft
        int input_len,              //could be calculated inside but I wrote an apposite class for it
        int output_len,
        cudaStream_t stram_f        //stream on which to launch the decimator
        ){

    //input len must be chopped to the exact amount of data


    average_spectra<<<1024, 32, 0, stram_f>>>(
        input,
        output,
        decim,
        nfft,
        input_len
    );

    scale_buffer<<<1024, 32>>>(output, output_len, 1./decim);

}

//decimate the output of the fft without tone selection
//NOTE: this thread has to be launched from its wrapper or witha Nblocks*Nthreads == out_len and
//it is not protected from accessing data outside input_len (see wrapper)
__global__ void accumulate_ffts(
        float2* __restrict__ input,
        float2* __restrict__ output,
        int decim,
        int nfft,
        int output_length
        ){

    //declare and initialize some shared memory
    extern __shared__ float2 shared_buffer[];

    //accumulate samples in shared memory iterating on the output
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
        offset < output_length;
        offset += gridDim.x*blockDim.x
        ){

        //there is a float2 per thread needed as accumulator
        shared_buffer[threadIdx.x].x = 0;
        shared_buffer[threadIdx.x].y = 0;

        int input_index;
        //this iterate over the samples to accumulate
        for(int j = 0; j < decim; j++){

            input_index = j * (offset % nfft);

            //accumulate samples
            shared_buffer[threadIdx.x].x += input[input_index].x;
            shared_buffer[threadIdx.x].y += input[input_index].y;
        }

        //copy back the memory in the output
        output[offset].x = shared_buffer[threadIdx.x].x/decim;
        output[offset].y = shared_buffer[threadIdx.x].y/decim;
    }
}


__global__ void zero_mem(float2* __restrict__ input, int input_len, float value){
    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
                offset < input_len;
                offset += gridDim.x*blockDim.x
                ){
            input[offset].x = value;
            input[offset].y = 0;
            //if(offset == 0) printf("Using zeromem!!\n");
    }

}

__device__ float magnitude(float2 sample){
    return sqrt(sample.x*sample.x+sample.y*sample.y);
}



//build_vna_profile()
#ifdef CUBLAS_API_H_
// cuBLAS API errors
void _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {

        case CUBLAS_STATUS_NOT_INITIALIZED:
            //print_error("CUBLAS_STATUS_NOT_INITIALIZED");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_NOT_INITIALIZED"<<std::endl;
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            //print_error( "CUBLAS_STATUS_ALLOC_FAILED");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_ALLOC_FAILED"<<std::endl;
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            //print_error( "CUBLAS_STATUS_INVALID_VALUE");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_INVALID_VALUE"<<std::endl;
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            //print_error( "CUBLAS_STATUS_ARCH_MISMATCH");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_ARCH_MISMATCH"<<std::endl;
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            //print_error( "CUBLAS_STATUS_MAPPING_ERROR");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_MAPPING_ERROR"<<std::endl;
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            //print_error( "CUBLAS_STATUS_EXECUTION_FAILED");
            std::cout<<"CUBLAS ERROR: CUBLAS_STATUS_EXECUTION_FAILED"<<std::endl;
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            //print_error( "CUBLAS_STATUS_INTERNAL_ERROR");
            std::cout<<"CUBLAS ERROR: UBLAS_STATUS_INTERNAL_ERROR"<<std::endl;
            break;
    }
}
#endif

// Used in the VNA as lock-in decimator.
void cublas_decim(
        float2* __restrict__ input,
        float2* __restrict__ output,
        float2* __restrict__ profile,
        cuComplex* __restrict__ zero,
        cuComplex* __restrict__ one,
        int ppt,
        int n_freqs,
        cublasHandle_t* __restrict__ handle
        ){


    cublasStatus_t err = cublasCgemv(*handle, CUBLAS_OP_T,
                           n_freqs, ppt,
                           one,
                           input, n_freqs,
                           profile, 1,
                           zero,
                           output, 1);
    _cudaGetErrorEnum(err);
}


//wrapper for the previous fft decimation function. decimates the pfb output.
//NOTE: this function does not take care of the reminder and suppose that calculation
//to determine the output_length has already been externally done.
void decimate_pfb(
        float2* __restrict__ input, //output of the pfb
        float2* __restrict__ output,//decimated output
        int decim,                  //decimation factor (multiplicative to the pfb one)
        int nfft,                   //length of the fft
        int output_length,          //could be calculated inside but I wrote an apposite class for it
        cudaStream_t stram_f        //stream on which to launch the decimator
        ){

        //the number of blocks can variate as the nuber of valid batches changes
        int blocks = std::ceil(output_length/PFB_DECIM_TPB);
        accumulate_ffts<<<blocks, PFB_DECIM_TPB, PFB_DECIM_TPB*sizeof(float2), stram_f>>>(
        input,output,decim,nfft,output_length);
}



void D_cublas_decim(
        double2* __restrict__ input,
        double2* __restrict__ output,
        double2* __restrict__ profile,
        cuDoubleComplex* __restrict__ zero,
        cuDoubleComplex* __restrict__ one,
        int ppt,
        int n_freqs,
        cublasHandle_t* __restrict__ handle
        ){


    cublasStatus_t err = cublasZgemv(*handle, CUBLAS_OP_T,
                           n_freqs, ppt,
                           one,
                           input, n_freqs,
                           profile, 1,
                           zero,
                           output, 1);
    _cudaGetErrorEnum(err);
}
__global__ void double2float(
        double2* __restrict__ input, //from float
        float2* __restrict__ output,//to double
        int length
    ){

    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
        offset < length;
        offset += gridDim.x*blockDim.x
        ){
        output[offset].x = input[offset].x;
        output[offset].y = input[offset].y;
    }
}

__global__ void float2double(
        float2* __restrict__ input, //from float
        double2* __restrict__ output,//to double
        int length
    ){

    for(int offset = blockIdx.x * blockDim.x + threadIdx.x;
        offset < length;
        offset += gridDim.x*blockDim.x
        ){
        output[offset].x = input[offset].x;
        output[offset].y = input[offset].y;
    }
}
