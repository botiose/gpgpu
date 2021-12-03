#include <stdint.h>

#include "kernel.cuh"

inline __device__ uint8_t
idx(const uint8_t* a, const unsigned int width, int x, int y) {
  return a[y * width + x];
}

__global__ void
sobelImage(const uint8_t* a,
           const unsigned int w,
           const unsigned int h,
           uint32_t* sx,
           uint32_t* sy) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
    sx[y * w + x] =
        abs(-1 * idx(a, w, x - 1, y - 1) + -2 * idx(a, w, x - 1, y) +
            -1 * idx(a, w, x - 1, y + 1) + idx(a, w, x + 1, y - 1) +
            2 * idx(a, w, x + 1, y) + idx(a, w, x + 1, y + 1));
    sy[y * w + x] =
        abs(idx(a, w, x - 1, y - 1) + 2 * idx(a, w, x, y - 1) +
            idx(a, w, x + 1, y - 1) + -1 * idx(a, w, x - 1, y + 1) +
            -2 * idx(a, w, x, y + 1) + -1 * idx(a, w, x + 1, y + 1));
  }
}


__global__ void
addToPool(const uint32_t* sobelX,
          const uint32_t* sobelY,
          const unsigned int width,
          const unsigned int height,
          const unsigned int poolSize,
          const unsigned int pooledWidth,
          uint32_t* pooledX,
          uint32_t* pooledY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int px = x / poolSize;
  int py = y / poolSize;

  atomicAdd(pooledX + px + py * pooledWidth, *(sobelX + x + y * width));
  atomicAdd(pooledY + px + py * pooledWidth, *(sobelY + x + y * width));
}

__global__ void
averagePool(const unsigned int poolSize,
            const unsigned int pooledWidth,
            uint32_t* pooledX,
            uint32_t* pooledY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  *(pooledX + x + y * pooledWidth) /= poolSize;
  *(pooledY + x + y * pooledWidth) /= poolSize;
}

__global__ void
computeResponse(const uint32_t* pooledX,
                const uint32_t* pooledY,
                const unsigned int pooledWidth,
                uint32_t* response) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int d = *(pooledX + x + y * pooledWidth) - *(pooledY + x + y * pooledWidth);

  *(response + x + y * pooledWidth) = d < 0 ? 0 : d;
}

__global__ void
grayscale(const uint8_t* orig, const unsigned int width, uint8_t* gray) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const uint8_t* p = orig + (x + y * width) * 3;
  *(gray + x + y * width) = (*p + *(p + 1) + *(p + 2)) / 3;
}

__global__ void
erode(const uint32_t* response,
      const unsigned int width,
      const unsigned int height,
      const int kernelHalfWidth,
      const int kernelHalfHeight,
      uint32_t* eroded) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  uint32_t minVal = UINT32_MAX;
  for (int i = -1 * kernelHalfWidth; i < kernelHalfWidth; i++) {
    for (int j = -1 * kernelHalfHeight; j < kernelHalfHeight; j++) {
      int resx = x + i;
      int resy = y + j;
      if (resx >= 0 && resy >= 0 && resx <= width && resy <= height) {
        minVal = min(minVal, *(response + resx + resy * width));
      }
    }
  }

  if (minVal != UINT32_MAX) {
    *(eroded + x + y * width) = minVal;
  }
}

__global__ void
dilatate(const uint32_t* eroded,
         const unsigned int width,
         const unsigned int height,
         const int kernelHalfWidth,
         const int kernelHalfHeight,
         uint32_t* dilatated) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  uint32_t maxVal = 0;
  for (int i = -1 * kernelHalfWidth; i < kernelHalfWidth; i++) {
    for (int j = -1 * kernelHalfHeight; j < kernelHalfHeight; j++) {
      int resx = x + i;
      int resy = y + j;
      if (resx >= 0 && resy >= 0 && resx <= width && resy <= height) {
        maxVal = max(maxVal, *(eroded + resx + resy * width));
      }
    }
  }

  *(dilatated + x + y * width) = maxVal;
}

__global__ void
threshold(const uint32_t* dilatated,
          const int width,
          const int maxVal,
          uint32_t* result) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  *(result + x + y * width) =
      *(dilatated + x + y * width) > 0.5 * maxVal ? UINT32_MAX : 0;
}
