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
#include <cstdio>


__global__ void minReduce(float* const d_array, const size_t size) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= size) {
    return ;
  }

  for (int i = size >> 1; i > 0; i >>= 1) {
    if (id + i < size && d_array[id + i] < d_array[id]) {
      d_array[id] = d_array[id + i];
    }
    __syncthreads();
  }
}

__global__ void maxReduce(float* const d_array, const size_t size) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= size) {
    return ;
  }

  for (int i = size >> 1; i > 0; i >>= 1) {
    if (id + i < size && d_array[id + i] > d_array[id]) {
      d_array[id] = d_array[id + i];
    }
    __syncthreads();
  }
}

__global__ void getBins(const float* const d_logLuminance, int* const d_bins,
                        float lumRange, float logLumMin,
                        int size, const size_t numBins) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id > size) {
    return ;
  }

  int bin = (d_logLuminance[id] - logLumMin) / lumRange * numBins;
  if (bin > numBins) {
    bin = numBins - 1;
  }
  atomicAdd(&d_bins[bin], 1);
}

__global__ void scan(const int* const d_bins, unsigned int* const d_cdf,
                     const size_t numBins) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= numBins) {
    return ;
  }

  extern __shared__ int local[];
  local[id] = d_bins[id];
  __syncthreads();

  for (int i = 1; i < numBins; i <<= 1) {
    if (id - i < 0) {
      continue;
    }
    int temp = local[id - i];
    __syncthreads();
    local[id] += temp;
    __syncthreads();
  }

  __syncthreads();
  d_cdf[id] = id == 0 ? 0 : local[id - 1];
}

// d_cfd/min_logLum/max_logLum need to be set, other parameters are read-only.
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  int size = numRows * numCols;

  float* d_array;
  float* h_out;
  checkCudaErrors(cudaMalloc(&d_array, sizeof(float) * size));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(float)));

  dim3 blockSize = 1024;
  dim3 gridSize = (size + 1024 - 1) / 1024;

  checkCudaErrors(cudaMemcpy(d_array, d_logLuminance, sizeof(float) * size, cudaMemcpyDeviceToDevice));
  minReduce<<<gridSize, blockSize>>>(d_array, size);
  minReduce<<<1, blockSize>>>(d_array, numCols);
  checkCudaErrors(cudaMemcpy(h_out, d_array, sizeof(float), cudaMemcpyDeviceToHost));
  min_logLum = *h_out;

  checkCudaErrors(cudaMemcpy(d_array, d_logLuminance, sizeof(float) * size, cudaMemcpyDeviceToDevice));
  maxReduce<<<gridSize, blockSize>>>(d_array, size);
  maxReduce<<<1, blockSize>>>(d_array, numCols);
  checkCudaErrors(cudaMemcpy(h_out, d_array, sizeof(float), cudaMemcpyDeviceToHost));
  max_logLum = *h_out;

  float lumRange = max_logLum - min_logLum;

  int* d_bins;
  checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
  checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));
  getBins<<<gridSize, blockSize>>>(d_logLuminance, d_bins, lumRange, min_logLum, size, numBins);
  gridSize = (numBins + 1024 - 1) / 1024;
  scan<<<gridSize, blockSize, numBins * sizeof(int)>>>(d_bins, d_cdf, numBins);

  int* h_data;
  checkCudaErrors(cudaMallocHost(&h_data, sizeof(int) * numBins));
  checkCudaErrors(cudaMemcpy(h_data, d_cdf, sizeof(int) * numBins, cudaMemcpyDeviceToHost));


  checkCudaErrors(cudaFree(d_array));
  checkCudaErrors(cudaFreeHost(h_out));
  checkCudaErrors(cudaFree(d_bins));
  checkCudaErrors(cudaFreeHost(h_data));
}
