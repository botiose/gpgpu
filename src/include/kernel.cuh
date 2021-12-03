#pragma once

__global__ void
sobelImage(const uint8_t* a,
           const unsigned int w,
           const unsigned int h,
           uint32_t* sx,
           uint32_t* sy);

__global__ void
addToPool(const uint32_t* sobelX,
          const uint32_t* sobelY,
          const unsigned int width,
          const unsigned int height,
          const unsigned int poolSize,
          const unsigned int pooledWidth,
          uint32_t* pooledX,
          uint32_t* pooledY);

__global__ void
averagePool(const unsigned int poolSize,
            const unsigned int pooledWidth,
            uint32_t* pooledX,
            uint32_t* pooledY);

__global__ void
computeResponse(const uint32_t* pooledX,
                const uint32_t* pooledY,
                const unsigned int pooledWidth,
                uint32_t* response);

__global__ void
grayscale(const uint8_t* orig, const unsigned int width, uint8_t* gray);

__global__ void
erode(const uint32_t* response,
      const unsigned int width,
      const unsigned int height,
      const int kernelHalfWidth,
      const int kernelHalfHeight,
      uint32_t* eroded);

__global__ void
dilatate(const uint32_t* eroded,
         const unsigned int width,
         const unsigned int height,
         const int kernelHalfWidth,
         const int kernelHalfHeight,
         uint32_t* dilatated);

__global__ void
threshold(const uint32_t* dilatated,
          const int width,
          const int maxVal,
          uint32_t* result);