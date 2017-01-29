//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

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

__global__ void getBin(unsigned int* d_input, unsigned int* const d_bin, int pos, int numElems) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numElems) {
    return ;
  }
  d_bin[id] = (d_input[id] & (1 << pos)) >> pos;
}


__global__ void getBinCount(unsigned int* const d_in,
                            unsigned int* const d_bin_count,
                            int numElems) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numElems) {
    return ;
  }
  atomicAdd(&d_bin_count[d_in[id]], 1);
}

__global__ void maxReduce(unsigned int* const d_index_partial,
                          unsigned int* const d_sum_prefix_block,
                          int numElems) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= numElems) {
    return ;
  }
  extern __shared__ unsigned int local[];
  local[threadIdx.x] = d_index_partial[id];
  __syncthreads();

  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (id + i < numElems && threadIdx.x + i < blockDim.x && local[threadIdx.x + i] > local[threadIdx.x]) {
      local[threadIdx.x] = local[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_sum_prefix_block[blockIdx.x] = local[0];
  }
}

// exclusive sum
// hills and steele scan
__global__ void getIndexPerBlock(unsigned int* const d_bin,
                                 unsigned int* const d_index_partial,
                                 unsigned int* const d_index,
                                 int numElems,
                                 int flag) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numElems) {
    return ;
  }

  int tid = threadIdx.x;
  extern __shared__ unsigned int local[];

  if (d_bin[id] == flag) {
    local[tid] = 1;
  } else {
    local[tid] = 0;
  }
  __syncthreads();

  for (int i = 1; i < blockDim.x; i <<= 1) {
    if (tid - i >= 0) {
      unsigned int temp = local[tid - i];
      __syncthreads();
      local[tid] += temp;
      __syncthreads();
    }
  }

  /* __syncthreads(); */
  if (d_bin[id] != flag) {
    d_index_partial[id] = 0;
  } else {
    d_index_partial[id] = local[tid];
  }
}

__global__ void prefixSum(unsigned int* const d_sum,
                          int num) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= num) {
    return ;
  }
  int tid = threadIdx.x;
  extern __shared__ unsigned int local[];
  local[tid] = d_sum[id];
  __syncthreads();

  for (int i = 1; i <= num; i <<= 1) {
    if (tid - i < 0) {
      continue;
    }
    unsigned int temp = local[tid - i];
    __syncthreads();
    local[tid] += temp;
    __syncthreads();
  }

  __syncthreads();
  d_sum[id] = local[tid] - d_sum[id];
}

__global__ void rearrange(unsigned int* const d_input,
                          unsigned int* const d_input_pos,
                          unsigned int* const d_index,
                          unsigned int* const d_output,
                          unsigned int* const d_output_pos,
                          int numElems) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numElems) {
    return ;
  }
  d_output[d_index[id]] = d_input[id];
  d_output_pos[d_index[id]] = d_input_pos[id];
}


__global__ void getIndex(unsigned int* const d_sum_prefix_block,
                         unsigned int* const d_index_partial,
                         unsigned int* const d_bin,
                         unsigned int* const d_index,
                         int numElems,
                         int flag,
                         unsigned int offset) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numElems) {
    return ;
  }
  if (d_bin[id] == flag) {
    d_index[id] = d_sum_prefix_block[blockIdx.x] + d_index_partial[id] + offset - 1;
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) {
  unsigned int* d_bin;
  unsigned int* d_bin_count;
  unsigned int* d_index;
  unsigned int* d_index_partial;

  unsigned int* d_input;
  unsigned int* d_output;
  unsigned int* d_input_pos;
  unsigned int* d_output_pos;
  unsigned int* d_sum_prefix_block;
  unsigned int* h_offset;
  unsigned int* h_debug;

  dim3 blockSize = 1024;
  dim3 gridSize = (numElems + 1024 - 1) / 1024;

  cudaMalloc(&d_bin, sizeof(unsigned int) * numElems);
  cudaMalloc(&d_bin_count, sizeof(unsigned int) * 2);
  cudaMalloc(&d_index, sizeof(unsigned int) * numElems);
  cudaMalloc(&d_index_partial, sizeof(unsigned int) * numElems);
  cudaMallocHost(&h_offset, sizeof(unsigned int) * 2);
  cudaMalloc(&d_sum_prefix_block, sizeof(unsigned) * gridSize.x);
  cudaMallocHost(&h_debug, sizeof(unsigned int) * numElems);

  for (int i = 0; i < 32; ++i) {
    cudaMemset(d_bin_count, 0, sizeof(unsigned int) * 2);
    if (i & 1) {
      d_input = d_outputVals;
      d_input_pos = d_outputPos;
      d_output = d_inputVals;
      d_output_pos = d_inputPos;
    } else {
      d_input = d_inputVals;
      d_input_pos = d_inputPos;
      d_output = d_outputVals;
      d_output_pos = d_outputPos;
    }

    getBin<<<gridSize, blockSize>>>(d_input, d_bin, i, numElems);
    cudaMemcpy(h_debug, d_bin, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost);

    getBinCount<<<gridSize, blockSize>>>(d_bin, d_bin_count, numElems);
    cudaMemcpy(h_offset, d_bin_count, sizeof(unsigned int) * 2, cudaMemcpyDeviceToHost);

    for (int j = 0; j < 2; ++j) {
      unsigned int offset = 0;
      if (j > 0) {
        offset = h_offset[j - 1];
      }

      getIndexPerBlock<<<gridSize, blockSize, 1024 * sizeof(unsigned int)>>>(d_bin, d_index_partial, d_index, numElems, j);

      maxReduce<<<gridSize, blockSize, 1024 * sizeof(unsigned int)>>>(d_index_partial, d_sum_prefix_block, numElems);
      prefixSum<<<1, gridSize.x, gridSize.x * sizeof(unsigned int)>>>(d_sum_prefix_block, gridSize.x);

      getIndex<<<gridSize, blockSize>>>(d_sum_prefix_block, d_index_partial, d_bin,
                                        d_index, numElems, j, offset);
    }

    rearrange<<<gridSize, blockSize>>>(d_input, d_input_pos, d_index, d_output, d_output_pos, numElems);
  }


  unsigned int* d_temp;
  cudaMalloc(&d_temp, sizeof(unsigned int) * numElems);
  cudaMemcpy(d_temp, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputVals, d_temp, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);

  cudaMemcpy(d_temp, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_inputPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputPos, d_temp, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);

  /* cudaMemcpy(h_debug, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost); */
    /* cudaMemcpy(h_debug, d_bin, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost); */
    /* for (int k = 0; k < numElems; ++k) { */
      /* fprintf(stdout, "0 [%d] == %u\n", k, h_debug[k]); */
    /* } */

  cudaFree(d_temp);
  cudaFree(d_bin);
  cudaFree(d_bin_count);
  cudaFree(d_index_partial);
  cudaFree(d_index);
  cudaFreeHost(h_offset);
}
