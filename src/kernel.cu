#include <stdint.h>

#include "kernel.cuh"

inline __device__ uint8_t
idx(const uint8_t* a, const unsigned int width, int x, int y) {
  return a[y * width + x];
}

__global__ void
grayscale(const uint8_t* orig, const unsigned int width, uint8_t* gray) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const uint8_t* p = orig + (x + y * width) * 3;
  *(gray + x + y * width) = (*p + *(p + 1) + *(p + 2)) / 3;
}

// Computes the vertical and horizontal sobels and pools the image.
__global__ void
sobelImage(const uint8_t* a,
           const unsigned int w,
           const unsigned int h,
           const unsigned int poolSize,
           const unsigned int pooledWidth,
           uint32_t* pooledX,
           uint32_t* pooledY) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
    int px = x / poolSize;
    int py = y / poolSize;

    int dx =
        abs(-1 * idx(a, w, x - 1, y - 1) + -2 * idx(a, w, x - 1, y) +
            -1 * idx(a, w, x - 1, y + 1) + idx(a, w, x + 1, y - 1) +
            2 * idx(a, w, x + 1, y) + idx(a, w, x + 1, y + 1));
    int dy =
        abs(idx(a, w, x - 1, y - 1) + 2 * idx(a, w, x, y - 1) +
            idx(a, w, x + 1, y - 1) + -1 * idx(a, w, x - 1, y + 1) +
            -2 * idx(a, w, x, y + 1) + -1 * idx(a, w, x + 1, y + 1));

    atomicAdd(pooledX + px + py * pooledWidth, dx);
    atomicAdd(pooledY + px + py * pooledWidth, dy);
  }
}

// Averages the pool values and computes the response.
__global__ void
computeResponse(const uint32_t* pooledY,
                const unsigned int poolSize,
                const unsigned int pooledWidth,
                uint32_t* pooledX) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int d = *(pooledX + x + y * pooledWidth) / poolSize -
          *(pooledY + x + y * pooledWidth) / poolSize;

  *(pooledX + x + y * pooledWidth) = d < 0 ? 0 : d;
}

__global__ void
erode(const uint32_t* response,
      const unsigned int width,
      const unsigned int height,
      const int kernelHalfWidth,
      const int kernelHalfHeight,
      uint8_t* eroded) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  uint8_t minVal = UINT8_MAX;
  for (int i = -1 * kernelHalfWidth; i < kernelHalfWidth; i++) {
    for (int j = -1 * kernelHalfHeight; j < kernelHalfHeight; j++) {
      int resx = x + i;
      int resy = y + j;
      if (resx >= 0 && resy >= 0 && resx <= width && resy <= height) {
        minVal = min(minVal, *(response + resx + resy * width));
      }
    }
  }

  if (minVal != UINT8_MAX) {
    *(eroded + x + y * width) = minVal;
  }
}

__global__ void
dilatate(const uint8_t* eroded,
         const unsigned int width,
         const unsigned int height,
         const int kernelHalfWidth,
         const int kernelHalfHeight,
         uint8_t* dilatated) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  uint8_t maxVal = 0;
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
threshold(const int width,
          const int maxVal,
          uint8_t* dilatated) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  *(dilatated + x + y * width) =
      *(dilatated + x + y * width) > 0.5 * maxVal ? UINT8_MAX : 0;
}
