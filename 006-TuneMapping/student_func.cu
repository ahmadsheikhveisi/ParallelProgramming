/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <limits>

#define BLOCK_SIZE 1024
void reduce_unitTest(void);
void histogram_unitTest(void);
void scan_unitTest(void);

typedef float (*pointFunction_t)(float, float);

__device__ float d_min_fun(float a, float b)
{
	//printf("d_min_fun");
	return (a > b)?b:a;
}
__device__ pointFunction_t p_d_min_fun = d_min_fun;

__device__ float d_max_fun(float a, float b)
{
	//printf("d_max_fun");
	return (a > b)?a:b;
}
__device__ pointFunction_t p_d_max_fun = d_max_fun;

__global__ void reduce_kernel(const float * const d_in,
		float *d_out, int size , float identity, pointFunction_t op)
{
	extern __shared__ float sdata[];

	int t_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int t_num = threadIdx.x;

	// if the thread is out of range of the input,
	// fill the data with identity
	if (t_idx >= size)
	{
		sdata[t_num] = identity;
	}
	else
	{
		// load to shared memory from glabal memory
		sdata[t_num] = d_in[t_idx];
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (t_num < s)
		{
			sdata[t_num] = op(sdata[t_num],sdata[t_num + s]);
		}
		__syncthreads();//
	}
	if (t_num == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

__global__ void histogram_kernel(const float *d_in, unsigned int * const d_bins,
		const size_t numBins, size_t size, size_t pix_thread,
		float lumMin, float lumRange)
{
	extern __shared__ unsigned int s_bins[];

	int global_id = blockIdx.x * blockDim.x + threadIdx.x;

	/*setting the bins to zero*/

	if (threadIdx.x < numBins)
	{
		s_bins[threadIdx.x] = 0;
	}

	__syncthreads();

	for (int cnt = 0; cnt <  pix_thread;++cnt)
	{
		if (global_id*pix_thread + cnt < size)
		{
			int bin = (int)((d_in[global_id*pix_thread + cnt] - lumMin)/lumRange * numBins);

			/*only for the maximum the bin is equal to numBIns*/
			bin = bin >= numBins? numBins - 1: bin;

			atomicAdd(&s_bins[bin],1);
		}
	}

	__syncthreads();

	if (threadIdx.x < numBins)
	{
		atomicAdd(&d_bins[threadIdx.x],s_bins[threadIdx.x]);
	}


}


__global__ void double_buffer_prefix_sum_kernel(const unsigned int *d_bins, unsigned int *d_cdf, int numBins)
{
	extern __shared__ unsigned int temp[];
	// allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// load input into shared memory.
	// Exclusive scan: shift right by one and set first element to 0
	temp[thid] = (thid > 0) ? d_bins[thid-1] : 0;
	__syncthreads();
	for( int offset = 1; offset < numBins; offset <<= 1 )
	{
		// swap double buffer indices
		pout = 1 - pout;
		pin  = 1 - pout;
		if(thid >= offset)
			temp[pout*numBins+thid] = temp[pin*numBins+thid] + temp[pin*numBins+thid - offset];
		else
			temp[pout*numBins+thid] = temp[pin*numBins+thid];
		__syncthreads();
	}
	d_cdf[thid] = temp[pout*numBins+thid]; // write output

}

void reduce (float & h_out, const float * const d_in,
		 int size, pointFunction_t h_pointFunction, float identity)
{
	if (size == 1)
	{
		// if size is equal to one
		// there is no need to reduce
		checkCudaErrors(cudaMemcpy(&h_out,d_in,sizeof(float),cudaMemcpyDeviceToHost));

	}
	int numThreads = BLOCK_SIZE;

	int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	float *d_intermediate;

	//intermediate size is set to the size; it contains the input
	checkCudaErrors(cudaMalloc(&d_intermediate, size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_intermediate, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice));

	// this is the size of first step output
	int out_size_bytes = sizeof(float) * numBlocks;
	float *d_out;
	checkCudaErrors(cudaMalloc(&d_out, out_size_bytes));

	float * d_input = d_intermediate;
	float * d_output = d_out;

	// continue until one element is left
	while (size != 1)
	{

		reduce_kernel<<<numBlocks,numThreads,numThreads*sizeof(float)>>>(d_input,d_output,size,identity,h_pointFunction);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// new size is equal to number of blocks since for each block
		// we get an element
		size = numBlocks;
		numBlocks = (numBlocks + BLOCK_SIZE - 1)/BLOCK_SIZE;

		if (size != 1)
		{
			float * t_d_input = d_input;
			d_input = d_output;
			d_output = t_d_input;
		}
	}

	checkCudaErrors(cudaMemcpy(&h_out,d_output,sizeof(float),cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_intermediate));
	checkCudaErrors(cudaFree(d_out));


}

void histogram(const float *d_in, unsigned int * const d_bins,
		const size_t numBins, size_t size,
		float lumMin, float lumRange)
{
	size_t pix_per_thread = 16;

	size_t pix_per_block = pix_per_thread * BLOCK_SIZE;

	size_t numBlocks = (size + pix_per_block - 1) / (pix_per_block);

	histogram_kernel<<<numBlocks,BLOCK_SIZE,numBins*sizeof(unsigned int)>>>(d_in, d_bins,
		numBins, size, pix_per_thread,
		lumMin, lumRange);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


}

void scan(unsigned int *d_bins, unsigned int *d_cdf, int numBins)
{
	// since we have more processors than work (numBins is equal to 1024), I chose Hillis Steele scan
	// it is more step efficient
	// double buffered prefix sum assumes that numBins is equal to block size
	// which is the case here

	double_buffer_prefix_sum_kernel<<<1,BLOCK_SIZE,2*numBins*sizeof(unsigned int)>>>(d_bins, d_cdf, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
       */
	//This problem is a reduce problem. we can approach the problem as 1D.
	// I used function pointer to pass the reduce operator

	//the host-side function pointer to  __device__ function
	pointFunction_t h_pointFunction;

	int size_elements = numRows * numCols;

	//copy the function pointers to the host equivalent
	checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction, p_d_max_fun, sizeof(pointFunction_t)));

	reduce(max_logLum,d_logLuminance,size_elements,h_pointFunction, std::numeric_limits<float>::min());

	checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction, p_d_min_fun, sizeof(pointFunction_t)));

	reduce(min_logLum,d_logLuminance,size_elements,h_pointFunction, std::numeric_limits<float>::max());

	/*
    2) subtract them to find the range
    */
	float lumRange = max_logLum - min_logLum;

	//std::cout << min_logLum << " " << max_logLum << std::endl;

	/*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
       */
	unsigned int *d_bins;

	checkCudaErrors(cudaMalloc(&d_bins,numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_bins,0,numBins*sizeof(unsigned int)));

	histogram(d_logLuminance, d_bins,
		numBins, size_elements,
		min_logLum, lumRange);
	/*
	unsigned int *h_bins = new unsigned int[numBins];
	checkCudaErrors(cudaMemcpy(h_bins,d_bins, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	for (int cnt = 0; cnt < numBins; ++cnt)
	{
		std::cout << h_bins[cnt] << " ";
		if ((cnt + 1) % 16 == 0)
		{
			std::cout << std::endl;
		}
	}

	delete[] h_bins;
	*/

	/*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	scan(d_bins, d_cdf, numBins);

	checkCudaErrors(cudaFree(d_bins));

}

void reduce_unitTest(void)
{
	//the host-side function pointer to  __device__ function
	pointFunction_t h_pointFunction;
	float max_logLum;
	float min_logLum;

	float *d_test_arr;
	float test_arr[] = {1.0f, 2.0f,2.0f,-3.3f,5.0f,3.2f,8.0f,8.0f,100.0f,-1.0f,-15.0f,3.0f,1.0f};
	checkCudaErrors(cudaMalloc(&d_test_arr, sizeof(float)*13));
	checkCudaErrors(cudaMemcpy(d_test_arr, test_arr, sizeof(float)*13, cudaMemcpyHostToDevice));

	int size_elements = 13;

	//copy the function pointers to the host equivalent
	checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction, p_d_max_fun, sizeof(pointFunction_t)));

	reduce(max_logLum,d_test_arr,size_elements,h_pointFunction, std::numeric_limits<float>::min());

	std::cout << "maximum out " << max_logLum << std::endl;

	checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction, p_d_min_fun, sizeof(pointFunction_t)));

	reduce(min_logLum,d_test_arr,size_elements,h_pointFunction, std::numeric_limits<float>::max());

	std::cout << "minimum out " << min_logLum << std::endl;

	checkCudaErrors(cudaFree(d_test_arr));
}

#define N_HISTO_TEST_SIZE 10
void histogram_unitTest(void)
{
	int numBins = 5;
	unsigned int *h_bins = new unsigned int(numBins);
	unsigned int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins,numBins*sizeof(unsigned int)));

	float h_test_arr[N_HISTO_TEST_SIZE] = {0.0f, 0.1f, 0.5f, 1.2f, 3.5f,
							4.1f, 5.0f, 3.0f, 1.0f, 4.5f};

	float *d_test_arr;
	checkCudaErrors(cudaMalloc(&d_test_arr, N_HISTO_TEST_SIZE * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_test_arr, h_test_arr, N_HISTO_TEST_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice));

	histogram(d_test_arr, d_bins,
		numBins, N_HISTO_TEST_SIZE,
		0.0f, 5.0f);

	checkCudaErrors(cudaMemcpy(h_bins, d_bins, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	for (int cnt = 0; cnt < numBins; ++cnt)
		std::cout << h_bins[cnt] << " ";
	std::cout << std::endl;

	checkCudaErrors(cudaFree(d_test_arr));
	checkCudaErrors(cudaFree(d_bins));

	delete[] h_bins;
}

void scan_unitTest(void)
{
	unsigned int h_bins[BLOCK_SIZE];
	for (int cnt = 0; cnt < BLOCK_SIZE; ++cnt)
	{
		h_bins[cnt] = 1;
	}

	unsigned int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int)*BLOCK_SIZE));
	checkCudaErrors(cudaMemcpy(d_bins, h_bins, sizeof(unsigned int)*BLOCK_SIZE, cudaMemcpyHostToDevice));

	unsigned int *d_cdf;
	checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int)*BLOCK_SIZE));
	checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * BLOCK_SIZE));


	scan(d_bins, d_cdf, BLOCK_SIZE);

	unsigned int h_cdf[BLOCK_SIZE];
	memset(h_cdf, 0, sizeof(unsigned int)* BLOCK_SIZE);
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int)* BLOCK_SIZE, cudaMemcpyDeviceToHost));
	for (int cnt = 0; cnt < BLOCK_SIZE; ++cnt)
	{
		std::cout << h_cdf[cnt] << " ";
		if ((cnt+1) % 16 == 0)
		{
			std::cout << std::endl;
		}
	}

	checkCudaErrors(cudaFree(d_bins));
	checkCudaErrors(cudaFree(d_cdf));
}
