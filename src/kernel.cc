#include <stdint.h>
#include <algorithm>
#include <iostream>

#include "kernel.hh"

void
grayscale(const uint8_t* orig,
          const unsigned int width,
          const unsigned int height,
          uint8_t* gray) {
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      const uint8_t* p = orig + (x + y * width) * 3;
      *(gray + x + y * width) = (*p + *(p + 1) + *(p + 2)) / 3;
    }
  }
}

uint8_t
idx(const uint8_t* a, const unsigned int width, int x, int y) {
  return a[y * width + x];
}

void
sobelImage(const uint8_t* a,
           const unsigned int w,
           const unsigned int h,
           const unsigned int poolSize,
           const unsigned int pooledWidth,
           uint32_t* pooledX,
           uint32_t* pooledY) {
  for (int x = 1; x < w - 1; x++) {
    for (int y = 1; y < h - 1; y++) {
      int dx = abs(-1 * idx(a, w, x - 1, y - 1) + -2 * idx(a, w, x - 1, y) +
                   -1 * idx(a, w, x - 1, y + 1) + idx(a, w, x + 1, y - 1) +
                   2 * idx(a, w, x + 1, y) + idx(a, w, x + 1, y + 1));
      int dy = abs(idx(a, w, x - 1, y - 1) + 2 * idx(a, w, x, y - 1) +
                   idx(a, w, x + 1, y - 1) + -1 * idx(a, w, x - 1, y + 1) +
                   -2 * idx(a, w, x, y + 1) + -1 * idx(a, w, x + 1, y + 1));
      int px = x / poolSize;
      int py = y / poolSize;

      *(pooledX + px + py * pooledWidth) += dx;
      *(pooledY + px + py * pooledWidth) += dy;

      // *(pooledX + px + py * pooledWidth) += *(a + x + y * w);
    }
  }
}

void
computeResponse(const uint32_t* pooledY,
                const unsigned int poolSize,
                const unsigned int pooledWidth,
                const unsigned int pooledHeight,
                uint32_t* pooledX) {
  for (int x = 0; x < pooledWidth; x++) {
    for (int y = 0; y < pooledHeight; y++) {
      int d = *(pooledX + x + y * pooledWidth) / (poolSize * poolSize) -
              *(pooledY + x + y * pooledWidth) / (poolSize * poolSize);

      *(pooledX + x + y * pooledWidth) = d < 0 ? 0 : d;

      // *(pooledX + x + y * pooledWidth) /= poolSize * poolSize;
    }
  }
}

void
erode(const uint32_t* response,
      const unsigned int width,
      const unsigned int height,
      const int kernelHalfWidth,
      const int kernelHalfHeight,
      uint8_t* eroded) {
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      uint8_t minVal = UINT8_MAX;
      for (int i = -1 * kernelHalfWidth; i < kernelHalfWidth; i++) {
        for (int j = -1 * kernelHalfHeight; j < kernelHalfHeight; j++) {
          int resx = x + i;
          int resy = y + j;
          if (resx >= 0 && resy >= 0 && resx <= width && resy <= height) {
            minVal = std::min((uint32_t)minVal, *(response + resx + resy * width));
          }
        }
      }

      if (minVal != UINT8_MAX) {
        *(eroded + x + y * width) = minVal;
      }
    }
  }
}

void
dilatate(const uint8_t* eroded,
         const unsigned int width,
         const unsigned int height,
         const int kernelHalfWidth,
         const int kernelHalfHeight,
         uint8_t* dilatated) {
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      uint8_t maxVal = 0;
      for (int i = -1 * kernelHalfWidth; i < kernelHalfWidth; i++) {
        for (int j = -1 * kernelHalfHeight; j < kernelHalfHeight; j++) {
          int resx = x + i;
          int resy = y + j;
          if (resx >= 0 && resy >= 0 && resx <= width && resy <= height) {
            maxVal = std::max((uint8_t)maxVal, *(eroded + resx + resy * width));
          }
        }
      }

      *(dilatated + x + y * width) = maxVal;
    }
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

void
threshold(const int width,
          const int height,
          const int maxVal,
          uint8_t* dilatated) {
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {

      *(dilatated + x + y * width) =
          *(dilatated + x + y * width) > 0.5 * maxVal ? UINT8_MAX : 0;
    }
  }
}
