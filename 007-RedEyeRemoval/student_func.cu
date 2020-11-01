//Udacity HW 4
//Radix Sorting

#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define USE_THRUST 1

#if USE_THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#endif

// must be a divisor of 32
// this number also determines the number of bins in histogram
// for 4 bits per pass we need 16 bins
#define NUM_BITS_PASS 4

#define HISTO_ELEMS_THREAD 16

// must be always a power of 2 for prescan
#define BLOCK_SIZE 1024

void prescan_UnitTest(void);

__global__ void predicate_kernel(unsigned int * d_out,
								unsigned int * d_in,
								size_t numElems,
								int bit_num)
{
	size_t thid = blockIdx.x * blockDim.x + threadIdx.x;

	if (thid >= numElems)
		return;

	int filter = ((1 << NUM_BITS_PASS) - 1) << bit_num;

	int num = (d_in[thid] & filter) >> bit_num;

	d_out[num * numElems + thid] = 1;
}

__global__ void histogram_kernel(const unsigned int *d_in, unsigned int * const d_bins,
		const size_t numBins, size_t size, size_t pix_thread, int bit_num)
{
	extern __shared__ unsigned int s_bins[];
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;

	int filter = ((1 << NUM_BITS_PASS) - 1) << bit_num;

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
			int bin = (d_in[global_id*pix_thread + cnt] & filter) >> bit_num;

			atomicAdd(&s_bins[bin],1);
		}
	}

	__syncthreads();

	if (threadIdx.x < numBins)
	{
		atomicAdd(&d_bins[threadIdx.x],s_bins[threadIdx.x]);
	}


}

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> (LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(_n) ((_n) >> LOG_NUM_BANKS)
#endif

__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n)
{
	extern __shared__ unsigned int temp[];// must be twice size of the block size
											// block size * sizeof(unsigned int) * 2 = 8KBytes
	int thid = threadIdx.x;

	int offset = 1;
	//A
	int ai = thid;
	int bi = thid + (n/2);//(BLOCK_SIZE/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];

	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			//B
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	//C
	if (thid == 0)
	{
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; // clear the last element
	}
	for (int d = 1; d < n; d <<= 1) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			//D
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	//E
	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //PUT YOUR SORT HERE

	/* Explore the data
	std::cout << "numElems " << numElems << std::endl; //220480

	unsigned int* h_inputVals = new unsigned int[numElems];

	checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));

	for (int cnt = 0; cnt < 100; ++cnt)
	{
		std::cout << h_inputVals[cnt] << " " ;
		if ((cnt + 1)%10 == 0)
		{
			std::cout << std::endl;
		}
	}

	delete[] h_inputVals;
	*/

#if USE_THRUST
	thrust::device_vector<unsigned int> d_keys(d_inputVals, d_inputVals + numElems);
	thrust::device_vector<unsigned int> d_values(d_inputPos, d_inputPos + numElems);

	thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

	thrust::copy(d_keys.begin(), d_keys.end(), thrust::device_pointer_cast(d_outputVals));
	thrust::copy(d_values.begin(), d_values.end(), thrust::device_pointer_cast(d_outputPos));


#else
	int num_bits = sizeof(unsigned int) * 8;
	int num_splits = 1 << NUM_BITS_PASS;
	int num_blocks = (numElems + BLOCK_SIZE + 1) / BLOCK_SIZE;

	unsigned int *d_pred;
	// for each split we need one predicate
	// for example for four way split we need four split
	checkCudaErrors(cudaMalloc(&d_pred, num_splits * numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_pred, 0, num_splits * numElems * sizeof(unsigned int)));

	unsigned int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, num_splits * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_bins, 0, num_splits * sizeof(unsigned int)));

	prescan_UnitTest();

	for(int cnt = 0; cnt < num_bits; cnt += NUM_BITS_PASS)
	{
		predicate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_pred, d_inputVals, numElems, cnt);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		histogram_kernel<<<num_blocks, BLOCK_SIZE,num_splits * sizeof(unsigned int) >>>(d_inputVals,
				d_bins, num_splits, numElems, HISTO_ELEMS_THREAD, cnt);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// scan
		//prescan_UnitTest();

		// move

		checkCudaErrors(cudaMemset(d_pred, 0, (1 << NUM_BITS_PASS) * numElems * sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_bins, 0, num_splits * sizeof(unsigned int)));
	}

	checkCudaErrors(cudaFree(d_pred));
	checkCudaErrors(cudaFree(d_bins));

#endif
}


#define PRESCAN_TEST_SIZE 2048
void prescan_UnitTest(void)
{
	unsigned int h_i[PRESCAN_TEST_SIZE];
	for(int cnt = 0; cnt < PRESCAN_TEST_SIZE; ++cnt)
	{
		h_i[cnt] = 1;//cnt + 1;
	}
	unsigned int *d_i;

	checkCudaErrors(cudaMalloc(&d_i, PRESCAN_TEST_SIZE * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_i, h_i, PRESCAN_TEST_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));
	unsigned int *d_o;
	checkCudaErrors(cudaMalloc(&d_o, PRESCAN_TEST_SIZE * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_o,0,PRESCAN_TEST_SIZE*sizeof(unsigned int)));

	prescan<<<1,PRESCAN_TEST_SIZE >> 1, PRESCAN_TEST_SIZE * sizeof(unsigned int)>>>(d_o,d_i, PRESCAN_TEST_SIZE);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(h_i, d_o, PRESCAN_TEST_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for(int cnt = 0; cnt < PRESCAN_TEST_SIZE; ++cnt)
	{
		std::cout << h_i[cnt] << " ";
		if ((cnt + 1) % 10 == 0)
		{
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;


	checkCudaErrors(cudaFree(d_i));
	checkCudaErrors(cudaFree(d_o));
}
