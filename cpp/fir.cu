
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fir.hpp"

#define checkcublas(X) assert( ( X ) == CUBLAS_STATUS_SUCCESS )

 FIR::FIR(cublasHandle_t handle, cudaStream_t stream, float2 *hcoeff, int M, int f, int nt) :
	_handle(handle),_stream(stream),_M(M),_f(f)
{
	_ntap = M * f;
	_nb = nt / M;
	assert(nt % M == 0);
	_nout = nt + f - 1;
	_nt = nt;

	cudaMalloc(&_dout,_nout*sizeof(float2));
	assert(_dout != NULL);
	cudaMemset(&_dout,0,_nout*sizeof(float2));

	cudaMalloc(&_dcoeff,_ntap*sizeof(float2));
	assert(_dcoeff != NULL);
	cudaMemcpy(_dcoeff,hcoeff,_ntap*sizeof(float2),cudaMemcpyHostToDevice);

	cudaMalloc(&_dtrapz,_nb*_f*sizeof(float2));
	assert(_dtrapz != NULL);
}

FIR::~FIR()
{
	cudaFree(_dout);
	cudaFree(_dtrapz);
	cudaFree(_dcoeff);
	memset(this,0,sizeof(*this));
}

void FIR::fir_apply(const float2 *din)
{
	float2 alpha = {1.0f,0.0f};
	float2 beta = {0.0f,0.0f};
	checkcublas(cublasCgemm(_handle,CUBLAS_OP_T,CUBLAS_OP_N,
				_nb,_f,_M,
				&alpha,
				din,_M,
				_dcoeff,_M,
				&beta,
				_dtrapz,_nb));

	for(int i=0;i<_f;i++) {
		checkcublas(cublasCaxpy(_handle,_nb,
					&alpha,
					&_dtrapz[i*_nb],1,
					&_dout[_f-i-1],1));
	}
}

void FIR::fir_shift()
{
	int rem = _f - 1;
	cudaMemcpyAsync(_dout,&_dout[_nb],rem*sizeof(float2),cudaMemcpyDeviceToDevice,_stream);
	cudaMemsetAsync(&_dout[rem],0,_nb*sizeof(float2),_stream);
}

void FIR::fir_to_host(float2 *hout)
{
	cudaMemcpyAsync(hout,_dout,_nb*sizeof(float2),cudaMemcpyDeviceToHost,_stream);	// M is the decimation factor

}

// To be refined
//! @todo refine names
void FIR::fir_to_dev(float2 *dout)
{
  cudaMemcpyAsync(dout,_dout,_nb*sizeof(float2),cudaMemcpyDeviceToDevice,_stream);	// M is the decimation factor
}
void FIR::run_fir(const float2 *din, float2 *hout)
{
	fir_apply(din);
	fir_to_dev(hout);
	fir_shift();
}
