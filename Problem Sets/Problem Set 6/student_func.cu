//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <vector>

/* __device__ int d_interiorPixelSize; */
int h_interiorPixelSize;

__global__
void swap(const float* lhs, const float* rhs) {
  const float* temp;
  temp = lhs;
  lhs = rhs;
  rhs = temp;
}

__global__
void getMask(uchar4* const d_sourceImg, unsigned char* const d_mask, int size) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= size) {
    return ;
  }
  d_mask[id] = (d_sourceImg[id].x + d_sourceImg[id].y + d_sourceImg[id].z < 3 * 255) ? 1 : 0;
}

__global__
void getRegionMask(const unsigned char* const d_mask,
                   unsigned char* const d_boarderPixels,
                   unsigned char* const d_strictInteriorPixels,
                   const size_t numRow,
                   const size_t numCol) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < 1 || x > numCol - 2 || y < 1 || y > numRow - 2) {
    return ;
  }

  int id = x + y * numCol;
  if (d_mask[id]) {
    if (d_mask[(x - 1) + y * numCol] && d_mask[(x + 1) + y * numCol]
        && d_mask[x + (y - 1) * numCol] && d_mask[x + (y + 1) * numCol]) {
      d_boarderPixels[id] = 0;
      d_strictInteriorPixels[id] = 1;
    } else {
      d_boarderPixels[id] = 1;
      d_strictInteriorPixels[id] = 0;
    }
  } else {
    d_boarderPixels[id] = 0;
    d_strictInteriorPixels[id] = 0;
  }
}

__global__
void splitChannel(const uchar4* const d_img,
                  unsigned char* const d_red,
                  unsigned char* const d_blue,
                  unsigned char* const d_green,
                  const size_t size) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= size) {
    return ;
  }
  d_red[id] = d_img[id].x;
  d_blue[id] = d_img[id].y;
  d_green[id] = d_img[id].z;
}

__global__
void computeG(const unsigned char* const d_channel,
              const uint2* const d_interiorPixelList,
              float* const d_g,
              const size_t numCol,
              const size_t interiorSize) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= interiorSize) {
    return ;
  }
  uint2 coord = d_interiorPixelList[id];
  unsigned int offset = coord.x * numCol + coord.y;
  float sum = 4.f * d_channel[offset];
  sum -= (float)d_channel[offset - 1] + (float)d_channel[offset + 1];
  sum -= (float)d_channel[offset + numCol] + (float)d_channel[offset - numCol];
  d_g[offset] = sum;
}

void getInteriorPixelList(unsigned char* const d_strictInteriorPixels,
                          uint2** const d_interiorPixelList, // OUT
                          const size_t srcSize,
                          const size_t numCol) {
  unsigned char* h_strictInteriorPixels;
  checkCudaErrors(cudaMallocHost(&h_strictInteriorPixels, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMemcpy(h_strictInteriorPixels, d_strictInteriorPixels,
                  sizeof(unsigned char) * srcSize, cudaMemcpyDeviceToHost));

  std::vector<uint2> interiors;
  for (int i = 0; i < srcSize; ++i) {
    if (h_strictInteriorPixels[i]) {
      int c = i % numCol;
      int r = i / numCol;
      interiors.push_back(make_uint2(r, c));
    }
  }

  uint2* h_interiors;
  checkCudaErrors(cudaMallocHost(&h_interiors, sizeof(uint2) * interiors.size()));
  for (int i = 0; i < (int)interiors.size(); ++i) {
    h_interiors[i] = interiors[i];
  }
  h_interiorPixelSize = interiors.size();

  checkCudaErrors(cudaMalloc(d_interiorPixelList, sizeof(uint2) * interiors.size()));
  checkCudaErrors(cudaMemcpy(*d_interiorPixelList, h_interiors, sizeof(uint2) * interiors.size(),
                  cudaMemcpyHostToDevice));

  checkCudaErrors(cudaFreeHost(h_strictInteriorPixels));
  checkCudaErrors(cudaFreeHost(h_interiors));
}

__device__
float minGPU(float a, float b) {
  return a > b ? b : a;
}

__device__
float maxGPU(float a, float b) {
  return a > b ? a : b;
}

__global__
void computeIteration(const unsigned char* const d_dst,
                      const unsigned char* const d_strictInteriorPixels,
                      const unsigned char* const d_boarderPixels,
                      const uint2* const d_interiorPixelList,
                      const size_t numCol,
                      const float* const d_blendedVals_1,
                      const float* const d_g,
                      const int interiorSize,
                      float* const d_blendedVals_2) { // OUT
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= interiorSize) {
    return ;
  }

  float blendedSum = 0.f;
  float borderSum  = 0.f;
  uint2 coord = d_interiorPixelList[id];
  unsigned int offset = coord.x * numCol + coord.y;

  if (d_strictInteriorPixels[offset - 1]) {
    blendedSum += d_blendedVals_1[offset - 1];
  } else {
    borderSum += d_dst[offset - 1];
  }

  if (d_strictInteriorPixels[offset + 1]) {
    blendedSum += d_blendedVals_1[offset + 1];
  } else {
    borderSum += d_dst[offset + 1];
  }

  if (d_strictInteriorPixels[offset - numCol]) {
    blendedSum += d_blendedVals_1[offset - numCol];
  } else {
    borderSum += d_dst[offset - numCol];
  }

  if (d_strictInteriorPixels[offset + numCol]) {
    blendedSum += d_blendedVals_1[offset + numCol];
  } else {
    borderSum += d_dst[offset + numCol];
  }

  float next_val = (blendedSum + borderSum + d_g[offset]) / 4.f;
  d_blendedVals_2[offset] = minGPU(255.f, maxGPU(0.f, next_val));
}

__global__
void copyToBlendImage(const uint2* const d_interiorPixelList,
                      const size_t size,
                      const size_t numCol,
                      const float* const d_blendedValsRed,
                      const float* const d_blendedValsBlue,
                      const float* const d_blendedValsGreen,
                      uchar4* const d_blendedImg) { // OUT
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= size) {
    return ;
  }
  uint2 coord = d_interiorPixelList[id];
  unsigned int offset = coord.x * numCol + coord.y;
  d_blendedImg[offset].x = d_blendedValsRed[offset];
  d_blendedImg[offset].y = d_blendedValsBlue[offset];
  d_blendedImg[offset].z = d_blendedValsGreen[offset];
}

__global__
void memcpy(float* const dst,
            const unsigned char* const src,
            const size_t size) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= size) {
    return ;
  }
  dst[id] = (float)src[id];
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  int srcSize = numRowsSource * numColsSource;

  uchar4* d_sourceImg;
  checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));

  uchar4* d_destImg;
  checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));

  unsigned char *d_mask;
  checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned char) * srcSize));

  dim3 blockSize = 1024;
  dim3 gridSize = (srcSize + 1024 - 1) / 1024;
  getMask<<<gridSize, blockSize>>>(d_sourceImg, d_mask, srcSize);

  unsigned char *d_boarderPixels;
  unsigned char *d_strictInteriorPixels;
  checkCudaErrors(cudaMalloc(&d_boarderPixels, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_strictInteriorPixels, sizeof(unsigned char) * srcSize));

  checkCudaErrors(cudaMemset(d_boarderPixels, 0, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMemset(d_strictInteriorPixels, 0, sizeof(unsigned char) * srcSize));

  blockSize = dim3(32, 32);
  gridSize = dim3((numColsSource + 32 - 1) / 32, (numRowsSource + 32 - 1) / 32);
  getRegionMask<<<gridSize, blockSize>>>(d_mask,
                                         d_boarderPixels,
                                         d_strictInteriorPixels,
                                         numRowsSource,
                                         numColsSource);

  uint2* d_interiorPixelList;
  getInteriorPixelList(d_strictInteriorPixels, &d_interiorPixelList, srcSize,
                       numColsSource);

  unsigned char* d_red_src;
  unsigned char* d_blue_src;
  unsigned char* d_green_src;
  checkCudaErrors(cudaMalloc(&d_red_src, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blue_src, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_green_src, sizeof(unsigned char) * srcSize));

  unsigned char* d_red_dst;
  unsigned char* d_blue_dst;
  unsigned char* d_green_dst;
  checkCudaErrors(cudaMalloc(&d_red_dst, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blue_dst, sizeof(unsigned char) * srcSize));
  checkCudaErrors(cudaMalloc(&d_green_dst, sizeof(unsigned char) * srcSize));

  blockSize = 1024;
  gridSize = (srcSize + 1024 - 1) / 1024;
  splitChannel<<<gridSize, blockSize>>>(d_sourceImg, d_red_src, d_blue_src, d_green_src, srcSize);
  splitChannel<<<gridSize, blockSize>>>(d_destImg, d_red_dst, d_blue_dst, d_green_dst, srcSize);

  float* d_g_red;
  float* d_g_blue;
  float* d_g_green;
  checkCudaErrors(cudaMalloc(&d_g_red, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_g_blue, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_g_green, sizeof(float) * srcSize));
  checkCudaErrors(cudaMemset(d_g_red, 0, sizeof(float) * srcSize));
  checkCudaErrors(cudaMemset(d_g_blue, 0, sizeof(float) * srcSize));
  checkCudaErrors(cudaMemset(d_g_green, 0, sizeof(float) * srcSize));

  blockSize = 1024;
  gridSize = (h_interiorPixelSize + 1024 - 1) / 1024;

  computeG<<<gridSize, blockSize>>>(d_red_src, d_interiorPixelList, d_g_red,
                                    numColsSource, h_interiorPixelSize);
  computeG<<<gridSize, blockSize>>>(d_blue_src, d_interiorPixelList, d_g_blue,
                                    numColsSource, h_interiorPixelSize);
  computeG<<<gridSize, blockSize>>>(d_green_src, d_interiorPixelList, d_g_green,
                                    numColsSource, h_interiorPixelSize);

  float* d_blendedValsRed_1;
  float* d_blendedValsRed_2;
  float* d_blendedValsBlue_1;
  float* d_blendedValsBlue_2;
  float* d_blendedValsGreen_1;
  float* d_blendedValsGreen_2;

  checkCudaErrors(cudaMalloc(&d_blendedValsRed_1, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blendedValsRed_2, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blendedValsBlue_1, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blendedValsBlue_2, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blendedValsGreen_1, sizeof(float) * srcSize));
  checkCudaErrors(cudaMalloc(&d_blendedValsGreen_2, sizeof(float) * srcSize));

  blockSize = 1024;
  gridSize = (srcSize + 1024 - 1) / 1024;
  memcpy<<<gridSize, blockSize>>>(d_blendedValsRed_1, d_red_src, srcSize);
  memcpy<<<gridSize, blockSize>>>(d_blendedValsRed_2, d_red_src, srcSize);
  memcpy<<<gridSize, blockSize>>>(d_blendedValsBlue_1, d_blue_src, srcSize);
  memcpy<<<gridSize, blockSize>>>(d_blendedValsBlue_2, d_blue_src, srcSize);
  memcpy<<<gridSize, blockSize>>>(d_blendedValsGreen_2, d_green_src, srcSize);
  memcpy<<<gridSize, blockSize>>>(d_blendedValsGreen_2, d_green_src, srcSize);

  /* uint2* temp; */
  /* checkCudaErrors(cudaMallocHost(&temp, sizeof(uint2) * h_interiorPixelSize)); */
  /* checkCudaErrors(cudaMemcpy(temp, d_interiorPixelList, sizeof(uint2) * h_interiorPixelSize, cudaMemcpyDeviceToHost)); */
  /* for (int i = 0; i < h_interiorPixelSize; ++i) { */
    /* printf("device %d %d\n", temp[i].x, temp[i].y); */
  /* } */

  blockSize = 1024;
  gridSize = (h_interiorPixelSize + 1024 - 1) / 1024;
  const int iternum = 800;
  for (int i = 0; i < iternum; ++i) { // red
    computeIteration<<<gridSize, blockSize>>>(d_red_dst,
                                              d_strictInteriorPixels,
                                              d_boarderPixels,
                                              d_interiorPixelList,
                                              numColsSource,
                                              d_blendedValsRed_1,
                                              d_g_red,
                                              h_interiorPixelSize,
                                              d_blendedValsRed_2);
    swap<<<1, 1>>>(d_blendedValsRed_1, d_blendedValsRed_2);
  }

  for (int i = 0; i < iternum; ++i) { // blue
    computeIteration<<<gridSize, blockSize>>>(d_blue_dst,
                                              d_strictInteriorPixels,
                                              d_boarderPixels,
                                              d_interiorPixelList,
                                              numColsSource,
                                              d_blendedValsRed_1,
                                              d_g_blue,
                                              h_interiorPixelSize,
                                              d_blendedValsRed_2);
    swap<<<1, 1>>>(d_blendedValsBlue_1, d_blendedValsBlue_2);
  }

  for (int i = 0; i < iternum; ++i) { // green
    computeIteration<<<gridSize, blockSize>>>(d_green_dst,
                                              d_strictInteriorPixels,
                                              d_boarderPixels,
                                              d_interiorPixelList,
                                              numColsSource,
                                              d_blendedValsRed_1,
                                              d_g_green,
                                              h_interiorPixelSize,
                                              d_blendedValsRed_2);
    swap<<<1, 1>>>(d_blendedValsGreen_1, d_blendedValsGreen_2);
  }

  swap<<<1, 1>>>(d_blendedValsRed_1, d_blendedValsRed_2);
  swap<<<1, 1>>>(d_blendedValsBlue_1, d_blendedValsBlue_2);
  swap<<<1, 1>>>(d_blendedValsGreen_1, d_blendedValsGreen_2);

  /* float* temp; */
  /* checkCudaErrors(cudaMallocHost(&temp, sizeof(float) * srcSize)); */
  /* checkCudaErrors(cudaMemcpy(temp, d_blendedValsRed_2, sizeof(float) * srcSize, cudaMemcpyDeviceToHost)); */
  /* for (int j = 0; j < srcSize; ++j) { */
      /* if (temp[j] != 255) { */
        /* printf("device %d %f\n", j, temp[j]); */
      /* } */
  /* } */

  uchar4* d_blendedImg;
  checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * srcSize));
  checkCudaErrors(cudaMemcpy(d_blendedImg, h_destImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));
  blockSize = 1024;
  gridSize = (h_interiorPixelSize + 1024 - 1) / 1024;
  copyToBlendImage<<<gridSize, blockSize>>>(d_interiorPixelList,
                                            h_interiorPixelSize,
                                            numColsSource,
                                            d_blendedValsRed_2,
                                            d_blendedValsBlue_2,
                                            d_blendedValsGreen_2,
                                            d_blendedImg);
  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * srcSize, cudaMemcpyDeviceToHost));

  // Free
  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));

  checkCudaErrors(cudaFree(d_mask));

  checkCudaErrors(cudaFree(d_boarderPixels));
  checkCudaErrors(cudaFree(d_strictInteriorPixels));

  checkCudaErrors(cudaFree(d_interiorPixelList));
  checkCudaErrors(cudaFree(d_red_src));
  checkCudaErrors(cudaFree(d_blue_src));
  checkCudaErrors(cudaFree(d_green_src));

  checkCudaErrors(cudaFree(d_red_dst));
  checkCudaErrors(cudaFree(d_blue_dst));
  checkCudaErrors(cudaFree(d_green_dst));

  checkCudaErrors(cudaFree(d_g_red));
  checkCudaErrors(cudaFree(d_g_blue));
  checkCudaErrors(cudaFree(d_g_green));

  checkCudaErrors(cudaFree(d_blendedValsRed_1));
  checkCudaErrors(cudaFree(d_blendedValsRed_2));
  checkCudaErrors(cudaFree(d_blendedValsBlue_1));
  checkCudaErrors(cudaFree(d_blendedValsBlue_2));
  checkCudaErrors(cudaFree(d_blendedValsGreen_1));
  checkCudaErrors(cudaFree(d_blendedValsGreen_2));

  checkCudaErrors(cudaFree(d_blendedImg));

  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
