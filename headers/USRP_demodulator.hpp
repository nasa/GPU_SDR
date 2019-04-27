#pragma once
#ifndef USRP_DEMODULATOR_INCLUDED
#define USRP_DEMODULATOR_INCLUDED

#include "kernels.cuh"
#include "USRP_server_settings.hpp"
#include "USRP_server_diagnostic.hpp"
#include "USRP_server_memory_management.hpp"


//! @brief This class handles the DSP of the buffer coming from the the SDR.
//! This is the class to implement to add a different DSP algorithm to the sevrer.
class buffer_DSP{

    public:

        //! @brief Initialization method for the class called when a new command is received.
        //! \param init_parameters pointer to the antenna configuration holding all the parameters.
        //! \param init_diagnostic Define if diagnostic for the class is active or not. Depending on the implementation the diagnostic can output filter profiles on file, print some timing line and so on.
        buffer_DSP(param* init_parameters, bool init_diagnostic = false);

        //! @brief Destructor of the DSP handler class. Deallocates all the memory used in the process on both the RAM and the GPU RAM.
        virtual ~buffer_DSP();

        //! @brief PAcket handler for DSP class. This method process information pointed by the in parameter and write the output in the memory pointed by out parameter.
        //! Both in and out are CPU RAM pointers already allocated. The memory input memory is NOT modified. The output memory has to be already allocated.
        //! This function assumes that there is enough memory allocated to write the full output.
        //! \return valid size of the output memory. Many processes cannot output a buffer of constant length, the return of this function tells the meaningful size of the memory.
        //! \param in Pointer to CPU input memory.
        //! \param out Pointer to CPU output memory.
        virtual size_t process(float2* __restrict__ in, float2* __restrict__ out);

}

class RX_buffer_demodulator{
    public:

        //stores the signal processing parameters
        param* parameters;

        //cut-off frequency fo the window. 1.f is Nyquist.
        float fcut;

        //initialization: the parameters are coming directly from the client (from the async communication thread)
        //diagnostic allows to print the window on a binary file and stores some diagnostic information
        RX_buffer_demodulator(param* init_parameters, bool init_diagnostic = false);

        //wrapper to the correct get function
        int process(float2** __restrict__ in, float2** __restrict__ out);

        //wrapper to the correct cleaning function
        void close();

    private:

        //enable or disable diagnostic information
        bool diagnostic;

        //enable or disable the post-demodulation decimator
        bool decimator_active;

        //pointer to the demodulation function
        int (RX_buffer_demodulator::*process_ptr)(float2** __restrict__, float2** __restrict__);

        //pointer to the cleaning function
        void (RX_buffer_demodulator::*clr_ptr)();

        //class to manage buffer pointers. Used to determine where to copy a buffer
        buffer_helper *buf_setting;

        //helper class for post-pfb decimators
        pfb_decimator_helper *pfb_decim_helper;

        //polyphase filter parameter wrapper + utility variables for buffer reminder
        filter_param settings;

        //device pointer. will store the window.
        float2* window;

        //gpu pointer to the wrapper of the filter parameters for the gpu kernels
        filter_param *d_params;

        //gpu pointer to the parameter wrapper for the chirp demodulation kernel
        //TODO this parameters notation for different functions is confusing. should be changed
        chirp_parameter* d_parameter;

        //the total batching used in the fft.
        //NOTE: last samples do not make sense; they are used to account for eventual buffers ratio irrationalities.
        int batching;

        //device pointer accounting for host 2 device memory tranfer of the untouched input
        float2 *raw_input;

        //device pointer for the output of the polyphase filter and the input of the fft
        float2 *input;

        //device pointer to the output of the fft
        float2 *output;

        //device pointer to a reordered portion of the fft output
        //this buffer will be copied to the host
        float2 *reduced_output;

        //host pointer to the result
        float2 *result;

        //device pointer to the eventual decimated output
        float2* decim_output;

        //plan of for the FFT
        cufftHandle plan;

        //internal stream of the class. NOTE: only recursive operation are performed on this stream.
        cudaStream_t internal_stream;

        //host version of the filter parameter struct
        filter_param h_param;

        //where to copy the new buffer in case of spare buffer back insertion (see decimation & buffermove)
        int spare_size;

        //VNA specific variable representing the number of points per tone
        int ppt;

        //VNA specific variable representing the number of samples (half) to discard in each tone of the VNA
        int side;

        //CPU bookeeping
        unsigned long int last_index;

        //helper class for vna decimation
        VNA_decimator_helper *vna_helper;

        //eventually used in decimation operations
        cublasHandle_t handle;

        cufftComplex zero;

        cufftComplex one;

        float2* profile;

        //wrap RX signal information in the apposite struct
        chirp_parameter h_parameter;

        //process a packet demodulating with chirp
        int process_chirp(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer);

        //process a packet with the pfb and set the variables for the next
        // returns the valid length of the output packet
        int process_pfb(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer);

        int pfb_out = 0;

        int output_len = 0;

        //same process as the pfb but there is no tone selection and the buffer is bully downloaded
        int process_pfb_spec(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer);

        //! @brief Process the nodsp case by simply copying the input in the output.
        //! @todo This function should't exist: it would be better to directly bypass the RX_buffer_demodulator::process() call in the TXRX::rx_single_link() process.
        int process_nodsp(float2** __restrict__ input_buffer, float2** __restrict__ output_buffer);

        //! Close the NODSP case: just destroy the cuda stream object.
        void close_nodsp();

        //clean up the pfb allocations
        void close_pfb();

        //clean up the pfb full spectrum
        void close_pfb_spec();

        //clean up the chirp demod allocation
        void close_chirp();

        //converts general purpose parameters into kernel wrapper parameters on gpu.
        //THIS ONLY TAKES CARE OF MULTI TONES MEASUREMENT
        void upload_multitone_parameters();

};
#endif
