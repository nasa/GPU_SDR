#pragma once

#include <stdint.h>
#include <cublas_v2.h>


class FIR
{
	public:
	// M is decimation factor
	// f should be around 8
	// M * f is the length of the coefficient vector
	// nt is the number of input samples the fir will work on
	FIR(cublasHandle_t handle, cudaStream_t stream, float2 *hcoeff, int M, int f, int nt);
	~FIR();

	cublasHandle_t _handle;
	cudaStream_t _stream;
	int _M;		// decimation factor
	int _f;		// taps per decimation factor
	int _ntap;  // M * f - number of taps in FIR filter
	int _nb;  // nt / M - output samples per input block
	int _nt;  // nb * M - input samples
	int _nout;  // f + nb - 1	// Length of output storage vector
	float2 *_dcoeff;  // f x M
	float2 *_dout; // nout
	float2 *_hout;  // nb
	float2 *_dtrapz; // nb x f

	void fir_apply(const float2 *din);
	void fir_shift();
	void fir_to_host(float2 *hout);
	void fir_to_dev(float2 *dout);

	void run_fir(const float2 *din, float2 *hout);

};
