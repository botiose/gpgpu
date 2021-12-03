#include <iostream>

#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernel.cuh"

#define GRIDVAL 20.0

unsigned int
divUp(const unsigned int& a, const unsigned int& b) {
  if (a % b != 0) {
    return a / b + 1;
  } else {
    return a / b;
  }
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
  uint8_t* grayHost = (uint8_t*)malloc(imageSize);

  uint8_t* origDev;
  uint8_t* grayDev;
  cudaMalloc((void**)&origDev, imageSize * 3);
  cudaMalloc((void**)&grayDev, imageSize);
  cudaMemcpy(origDev, rgb, imageSize * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(grayDev, grayHost, imageSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(GRIDVAL, GRIDVAL);
  dim3 numBlocks(divUp(width, GRIDVAL), divUp(height, GRIDVAL));

  grayscale<<<numBlocks, threadsPerBlock>>>(origDev, width, grayDev);

  stbi_image_free(rgb);

  cudaDeviceSynchronize();

  // TODO: start remove
  // cudaMemcpy(grayHost, grayDev, imageSize, cudaMemcpyDeviceToHost);
  // stbi_write_png("etc/data/grayscale.png", width, height, 1, grayHost, width);
  // TODO: end remove

  int sobelImageSize = sizeof(uint32_t) * (width * height);

  uint32_t* sobelXDev;
  cudaMalloc((void**)&sobelXDev, sobelImageSize);
  uint32_t* sobelYDev;
  cudaMalloc((void**)&sobelYDev, sobelImageSize);

  sobelImage<<<numBlocks, threadsPerBlock>>>(
      grayDev, width, height, sobelXDev, sobelYDev);

  cudaDeviceSynchronize();

  // TODO: start remove
  // uint32_t* sobelXHost = (uint32_t*)malloc(sobelImageSize);
  // uint32_t* sobelYHost = (uint32_t*)malloc(sobelImageSize);
  // cudaMemcpy(sobelYHost, sobelYDev, sobelImageSize, cudaMemcpyDeviceToHost);
  // cudaMemcpy(sobelXHost, sobelXDev, sobelImageSize, cudaMemcpyDeviceToHost);
  // uint8_t* sobelYHost8 = (uint8_t*)malloc(imageSize);
  // uint8_t* sobelXHost8 = (uint8_t*)malloc(imageSize);
  // for (int i = 0; i < width * height; i++) {
  //   *(sobelXHost8 + i) = *(sobelXHost + i);
  //   *(sobelYHost8 + i) = *(sobelYHost + i);
  // }
  // stbi_write_png("etc/data/sobelx.png", width, height, 1, sobelXHost8, width);
  // stbi_write_png("etc/data/sobely.png", width, height, 1, sobelYHost8, width);
  // TODO: end remove

  int poolSize = 3;
  int pooledWidth = width / poolSize;
  int pooledHeight = height / poolSize;
  int pooledImageSize = sizeof(uint32_t) * (pooledWidth * pooledHeight);

  uint32_t* pooledXDev;
  uint32_t* pooledYDev;
  cudaMalloc((void**)&pooledXDev, pooledImageSize);
  cudaMalloc((void**)&pooledYDev, pooledImageSize);
  cudaMemset(pooledXDev, 0, pooledImageSize);
  cudaMemset(pooledYDev, 0, pooledImageSize);

  addToPool<<<numBlocks, threadsPerBlock>>>(sobelXDev,
                                            sobelYDev,
                                            width,
                                            height,
                                            poolSize,
                                            pooledWidth,
                                            pooledXDev,
                                            pooledYDev);

  // cudaDeviceSynchronize();

  dim3 threadsPerBlockPool(GRIDVAL, GRIDVAL);
  dim3 numBlocksPool(divUp(pooledWidth, GRIDVAL), divUp(pooledHeight, GRIDVAL));

  averagePool<<<pooledWidth * pooledHeight, 1>>>(
      poolSize, pooledWidth, pooledXDev, pooledYDev);

  cudaDeviceSynchronize();

  // TODO: start remove
  // uint32_t* pooledXHost = (uint32_t*)malloc(pooledImageSize);
  // uint32_t* pooledYHost = (uint32_t*)malloc(pooledImageSize);
  // cudaMemcpy(pooledXHost, pooledXDev, pooledImageSize, cudaMemcpyDeviceToHost);
  // cudaMemcpy(pooledYHost, pooledYDev, pooledImageSize, cudaMemcpyDeviceToHost);
  // uint8_t* pooledXHost8 = (uint8_t*)malloc(pooledImageSize);
  // uint8_t* pooledYHost8 = (uint8_t*)malloc(pooledImageSize);
  // for (int i = 0; i < pooledWidth * pooledHeight; i++) {
  //   *(pooledXHost8 + i) = *(pooledXHost + i);
  //   *(pooledYHost8 + i) = *(pooledYHost + i);
  // }
  // stbi_write_png("etc/data/pooledx.png",
  //                pooledWidth,
  //                pooledHeight,
  //                1,
  //                pooledXHost8,
  //                pooledWidth);
  // stbi_write_png("etc/data/pooledy.png",
  //                pooledWidth,
  //                pooledHeight,
  //                1,
  //                pooledYHost8,
  //                pooledWidth);
  // TODO: end remove

  uint32_t* responseDev;
  cudaMalloc((void**)&responseDev, pooledImageSize);

  computeResponse<<<numBlocksPool, threadsPerBlockPool>>>(
      pooledXDev, pooledYDev, pooledWidth, responseDev);

  cudaDeviceSynchronize();

  // TODO: start remove
  // uint32_t* responseHost = (uint32_t*)malloc(pooledImageSize);
  // cudaMemcpy(
  //     responseHost, responseDev, pooledImageSize, cudaMemcpyDeviceToHost);
  // uint8_t* responseHost8 = (uint8_t*)malloc(pooledImageSize);
  // for (int i = 0; i < pooledWidth * pooledHeight; i++) {
  //   *(responseHost8 + i) = *(responseHost + i);
  // }
  // stbi_write_png("etc/data/response.png",
  //                pooledWidth,
  //                pooledHeight,
  //                1,
  //                responseHost8,
  //                pooledWidth);
  // TODO: end remove

  int eroKernelWidth = 6;
  int eroKernelHeight = 3;

  uint32_t* erodedDev;
  cudaMalloc((void**)&erodedDev, pooledImageSize);

  erode<<<numBlocksPool, threadsPerBlockPool>>>(responseDev,
                                                pooledWidth,
                                                pooledHeight,
                                                eroKernelWidth / 2,
                                                eroKernelHeight / 2,
                                                erodedDev);

  cudaDeviceSynchronize();

  // TODO: start remove
  // uint32_t* erodedHost = (uint32_t*)malloc(pooledImageSize);
  // cudaMemcpy(erodedHost, erodedDev, pooledImageSize, cudaMemcpyDeviceToHost);
  // uint8_t* erodedHost8 = (uint8_t*)malloc(pooledImageSize);
  // for (int i = 0; i < pooledWidth * pooledHeight; i++) {
  //   *(erodedHost8 + i) = *(erodedHost + i);
  // }
  // stbi_write_png("etc/data/eroded.png",
  //                pooledWidth,
  //                pooledHeight,
  //                1,
  //                erodedHost8,
  //                pooledWidth);
  // TODO: end remove

  int dilKernelWidth = 25;
  int dilKernelHeight = 15;
  
  uint32_t* dilatatedDev;
  cudaMalloc((void**)&dilatatedDev, pooledImageSize);

  dilatate<<<numBlocksPool, threadsPerBlockPool>>>(erodedDev,
                                                   pooledWidth,
                                                   pooledHeight,
                                                   dilKernelWidth / 2,
                                                   dilKernelHeight / 2,
                                                   dilatatedDev);

  cudaDeviceSynchronize();

  uint32_t* dilatatedHost = (uint32_t*)malloc(pooledImageSize);
  cudaMemcpy(
      dilatatedHost, dilatatedDev, pooledImageSize, cudaMemcpyDeviceToHost);

  // TODO: start remove
  // uint8_t* dilatatedHost8 = (uint8_t*)malloc(pooledImageSize);
  // for (int i = 0; i < pooledWidth * pooledHeight; i++) {
  //   *(dilatatedHost8 + i) = *(dilatatedHost + i);
  // }
  // stbi_write_png("etc/data/dilatated.png",
  //                pooledWidth,
  //                pooledHeight,
  //                1,
  //                dilatatedHost8,
  //                pooledWidth);
  // TODO: end remove

  uint32_t maxVal = 0;
  for (int i = 0; i < pooledHeight * pooledWidth; i++) {
    uint32_t curVal = *(dilatatedHost + i);
    if (maxVal < curVal) {
      maxVal = curVal;
    }
  }

  uint32_t* resultDev;
  cudaMalloc((void**)&resultDev, pooledImageSize);

  threshold<<<numBlocksPool, threadsPerBlockPool>>>(
      dilatatedDev, pooledWidth, maxVal, resultDev);

  // TODO: start remove
  uint32_t* resultHost = (uint32_t*)malloc(pooledImageSize);
  cudaMemcpy(resultHost, resultDev, pooledImageSize, cudaMemcpyDeviceToHost);

  uint8_t* resultHost8 = (uint8_t*)malloc(pooledImageSize);
  for (int i = 0; i < pooledWidth * pooledHeight; i++) {
    *(resultHost8 + i) = *(resultHost + i);
  }
  stbi_write_png(argv[2],
                 pooledWidth,
                 pooledHeight,
                 1,
                 resultHost8,
                 pooledWidth);
  // TODO: end remove
}
