#include <iostream>
#include <valgrind/callgrind.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernel.hh"

#define POOL_SIZE 3
#define ERO_KERNEL_WIDTH 6
#define ERO_KERNEL_HEIGHT 3
#define DIL_KERNEL_WIDTH 30
#define DIL_KERNEL_HEIGHT 15

int
main(int argc, char** argv) {

  if (argc != 3) {
    std::cerr << "Usage: gpgpu <src-image.jpb> <dst-image.png>" << std::endl;
    exit(1);
  }

  int width, height, bpp;

  uint8_t* rgb = stbi_load(argv[1], &width, &height, &bpp, 3);

  uint8_t* gray = (uint8_t*)malloc(sizeof(uint8_t) * width * height);

  CALLGRIND_START_INSTRUMENTATION;
  grayscale(rgb, width, height, gray);

  stbi_image_free(rgb);

  int pooledWidth = width / POOL_SIZE;
  int pooledHeight = height / POOL_SIZE;
  int pooledImageSize = sizeof(uint32_t) * (pooledWidth * pooledHeight);

  uint32_t* pooledX = (uint32_t*)malloc(pooledImageSize);
  uint32_t* pooledY = (uint32_t*)malloc(pooledImageSize);
  memset(pooledX, 0, pooledImageSize);
  memset(pooledY, 0, pooledImageSize);

  sobelImage(gray,
             width - width % POOL_SIZE,
             height - height % POOL_SIZE,
             POOL_SIZE,
             pooledWidth,
             pooledX,
             pooledY);

  free(gray);

  computeResponse(pooledY, POOL_SIZE, pooledWidth, pooledHeight, pooledX);

  free(pooledY);

  pooledImageSize = sizeof(uint8_t) * (pooledWidth * pooledHeight);

  uint8_t* eroded = (uint8_t*)malloc(pooledImageSize);

  erode(pooledX,
        pooledWidth,
        pooledHeight,
        ERO_KERNEL_WIDTH / 2,
        ERO_KERNEL_HEIGHT / 2,
        eroded);

  free(pooledX);

  uint8_t* dilatated = (uint8_t*)malloc(pooledImageSize);

  dilatate(eroded,
           pooledWidth,
           pooledHeight,
           DIL_KERNEL_WIDTH / 2,
           DIL_KERNEL_HEIGHT / 2,
           dilatated);

  free(eroded);

  uint8_t maxVal = getMaxVal(dilatated, pooledWidth, pooledHeight);

  // Threshold the image by 50%.
  threshold(pooledWidth, pooledHeight, maxVal, dilatated);
  CALLGRIND_STOP_INSTRUMENTATION;

  uint8_t* result = (uint8_t*)malloc(sizeof(uint8_t) * (pooledWidth * pooledHeight));

  for (int i = 0; i < pooledWidth * pooledHeight; i++) {
    *(result + i) = (uint8_t)*(dilatated + i);
  }

  stbi_write_jpg(argv[2], pooledWidth, pooledHeight, 1, dilatated, 100);

  free(dilatated);

}
