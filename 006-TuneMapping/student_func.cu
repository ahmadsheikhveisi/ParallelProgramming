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

__global__ void histogram_kernel(const float *d_in, float *d_bins, const int BIN_COUNTS)
{

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

void histogram(void)
{

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

	std::cout << "size " << size_elements << std::endl;

	/*
    2) subtract them to find the range
    */
	float lumRange = max_logLum - min_logLum;



	/*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
       */


	/*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


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
