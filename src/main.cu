#include <iostream>

#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernel.cuh"

#define GRID_SIZE 32

#define POOL_SIZE 3
#define ERO_KERNEL_WIDTH 6
#define ERO_KERNEL_HEIGHT 4
#define DIL_KERNEL_WIDTH 25
#define DIL_KERNEL_HEIGHT 15

unsigned int
divUp(const unsigned int& a, const unsigned int& b) {
  if (a % b != 0) {
    return a / b + 1;
  } else {
    return a / b;
  }
}

uint8_t
getMaxVal(const uint8_t* arr,
          const unsigned int pooledWidth,
          const unsigned int pooledHeight) {
  uint8_t maxVal = 0;
  for (int i = 0; i < pooledHeight * pooledWidth; i++) {
    uint8_t curVal = *(arr + i);
    if (maxVal < curVal) {
      maxVal = curVal;
    }
  }

  return maxVal;
}

int
main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: gpgpu <src-image.jpb> <dst-image.png>" << std::endl;
    exit(1);
  }

  int width, height, bpp;

  uint8_t* rgb = stbi_load(argv[1], &width, &height, &bpp, 3);

  int imageSize = sizeof(uint8_t) * (width * height);

  uint8_t* origDev;
  uint8_t* grayDev;
  cudaMalloc((void**)&origDev, imageSize * 3);
  cudaMalloc((void**)&grayDev, imageSize);
  cudaMemcpy(origDev, rgb, imageSize * 3, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(GRID_SIZE, GRID_SIZE);
  dim3 numBlocks(divUp(width, GRID_SIZE), divUp(height, GRID_SIZE));

  // Compute the grayscale.
  grayscale<<<numBlocks, threadsPerBlock>>>(origDev, width, grayDev);

  stbi_image_free(rgb);

  cudaDeviceSynchronize();

  cudaFree(origDev);

  int pooledWidth = width / POOL_SIZE;
  int pooledHeight = height / POOL_SIZE;
  int pooledImageSize = sizeof(uint32_t) * (pooledWidth * pooledHeight);

  uint32_t* pooledXDev;
  uint32_t* pooledYDev;
  cudaMalloc((void**)&pooledXDev, pooledImageSize);
  cudaMalloc((void**)&pooledYDev, pooledImageSize);
  cudaMemset(pooledXDev, 0, pooledImageSize);
  cudaMemset(pooledYDev, 0, pooledImageSize);

  // Sobel and add to pool.
  sobelImage<<<numBlocks, threadsPerBlock>>>(
      grayDev, width, height, POOL_SIZE, pooledWidth, pooledXDev, pooledYDev);

  cudaDeviceSynchronize();

  cudaFree(grayDev);
  
  dim3 threadsPerBlockPool(GRID_SIZE, GRID_SIZE);
  dim3 numBlocksPool(divUp(pooledWidth, GRID_SIZE),
                     divUp(pooledHeight, GRID_SIZE));

  computeResponse<<<numBlocksPool, threadsPerBlockPool>>>(
      pooledYDev, POOL_SIZE, pooledWidth, pooledXDev);

  cudaDeviceSynchronize();

  cudaFree(pooledYDev);
  
  uint8_t* erodedDev;
  cudaMalloc((void**)&erodedDev, pooledImageSize);

  // Erode the image.
  erode<<<numBlocksPool, threadsPerBlockPool>>>(pooledXDev,
                                                pooledWidth,
                                                pooledHeight,
                                                ERO_KERNEL_WIDTH / 2,
                                                ERO_KERNEL_HEIGHT / 2,
                                                erodedDev);

  cudaDeviceSynchronize();

  cudaFree(pooledXDev);

  uint8_t* dilatatedDev;
  cudaMalloc((void**)&dilatatedDev, pooledImageSize);

  // Dilatate the image.
  dilatate<<<numBlocksPool, threadsPerBlockPool>>>(erodedDev,
                                                   pooledWidth,
                                                   pooledHeight,
                                                   DIL_KERNEL_WIDTH / 2,
                                                   DIL_KERNEL_HEIGHT / 2,
                                                   dilatatedDev);

  cudaDeviceSynchronize();

  cudaFree(erodedDev);
  
  uint8_t* dilatatedHost = (uint8_t*)malloc(pooledImageSize);
  cudaMemcpy(
      dilatatedHost, dilatatedDev, pooledImageSize, cudaMemcpyDeviceToHost);

  uint8_t maxVal = getMaxVal(dilatatedHost, pooledWidth, pooledHeight);

  // Threshold the image by 50%.
  threshold<<<numBlocksPool, threadsPerBlockPool>>>(
      pooledWidth, maxVal, dilatatedDev);

  cudaDeviceSynchronize();
  
  cudaMemcpy(
      dilatatedHost, dilatatedDev, pooledImageSize, cudaMemcpyDeviceToHost);

  cudaFree(dilatatedDev);
  
  stbi_write_png(
      argv[2], pooledWidth, pooledHeight, 1, dilatatedHost, pooledWidth);

  free(dilatatedHost);
}
