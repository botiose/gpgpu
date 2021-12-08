#pragma once

void
grayscale(const uint8_t* orig,
          const unsigned int width,
          const unsigned int height,
          uint8_t* gray);

void
sobelImage(const uint8_t* a,
           const unsigned int w,
           const unsigned int h,
           const unsigned int poolSize,
           const unsigned int pooledWidth,
           uint32_t* pooledX,
           uint32_t* pooledY
           );

void
computeResponse(const uint32_t* pooledY,
                const unsigned int poolSize,
                const unsigned int pooledWidth,
                const unsigned int pooledHeight,
                uint32_t* pooledX);

void
erode(const uint32_t* response,
      const unsigned int width,
      const unsigned int height,
      const int kernelHalfWidth,
      const int kernelHalfHeight,
      uint8_t* eroded);

void
dilatate(const uint8_t* eroded,
         const unsigned int width,
         const unsigned int height,
         const int kernelHalfWidth,
         const int kernelHalfHeight,
         uint8_t* dilatated);

uint8_t
getMaxVal(const uint8_t* arr,
          const unsigned int pooledWidth,
          const unsigned int pooledHeight);

void
threshold(const int width,
          const int height,
          const int maxVal,
          uint8_t* dilatated);
