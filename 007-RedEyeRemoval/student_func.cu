//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

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

#define USE_THRUST 0


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

#endif
}
