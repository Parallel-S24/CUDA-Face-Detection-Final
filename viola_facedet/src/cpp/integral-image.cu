#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "integral-image.h"
#include "utility.h"
#include "haar-like.h"

IntegralImage::IntegralImage(float inputBuf[], int w, int h, int size, bool squared) {
	data.resize(w, std::vector<float>(h, 0));
	std::vector<float> sumTable(size);
	for (int i = 3; i < size; i += 4) {
		auto vec2 = offsetToVec2(i - 3, w);
		int x = vec2[0];
		int y = vec2[1];
		float yP = y - 1 < 0 ? 0 : sumTable[i - w * 4];
		float xP = x - 1 < 0 ? 0 : this->data[x - 1][y];
		sumTable[i] = !squared ? yP + inputBuf[i] : yP + std::pow(inputBuf[i], 2);
		data[x][y] = xP + sumTable[i];	
	}
}

__device__ float IntegralImage::getRectangleSumDevice(const float* data, int x, int y, int w, int h) {
    float sum;
    if (x != 0 && y != 0) {
        float a = data[(y - 1) * w + (x - 1)];
        float b = data[(y - 1) * w + (x + w - 1)];
        float c = data[(y + h - 1) * w + (x + w - 1)];
        float d = data[(y + h - 1) * w + (x - 1)];
        sum = c + a - (b + d);
    } else if (x == 0 && y != 0) {
        float b = data[(y - 1) * w + (x + w - 1)];
        float c = data[(y + h - 1) * w + (x + w - 1)];
        sum = c - b;
    } else if (y == 0 && x != 0) {
        float c = data[(y + h - 1) * w + (x + w - 1)];
        float d = data[(y + h - 1) * w + (x - 1)];
        sum = c - d;
    } else {
        sum = data[(y + h - 1) * w + (x + w - 1)];
    }
    return sum;
}
__device__ float IntegralImage::computeFeatureDevice(Haarlike h, const float* data, int sx, int sy) {
    float wSum, bSum;
    if (h.type == 1) {
        wSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy, h.w);
        bSum = getRectangleSumDevice(data, h.w, h.x + sx + h.w, h.y + sy, h.w);
    } else if (h.type == 2) {
        wSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy, h.w) + 
               getRectangleSumDevice(data, h.w, h.x + sx + h.w * 2, h.y + sy, h.w);
        bSum = getRectangleSumDevice(data, h.w, h.x + sx + h.w, h.y + sy, h.w);
    } else if (h.type == 3) {
        wSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy, h.w);
        bSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy + h.h, h.w);
    } else if (h.type == 4) {
        wSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy, h.w) + 
               getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy + h.h * 2, h.w);
        bSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy + h.h, h.w);
    } else {
        wSum = getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy, h.w) + 
               getRectangleSumDevice(data, h.w, h.x + sx + h.w, h.y + sy + h.h, h.w);
        bSum = getRectangleSumDevice(data, h.w, h.x + sx + h.w, h.y + sy, h.w) + 
               getRectangleSumDevice(data, h.w, h.x + sx, h.y + sy + h.h, h.w);
    }
    float f = bSum - wSum;
    return f;
}

// __global__ void computeFeatureKernel(Haarlike* h, int w, int sx, int sy, float* results, int numFeatures) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numFeatures) {
//         results[idx] = computeFeatureDevice(h[idx], data, sx, sy);
//     }
// }

// void IntegralImage::computeFeatureBatch(Haarlike* h, int numFeatures, int sx, int sy, float* results) const {
//     int numThreads = 256;
//     int numBlocks = (numFeatures + numThreads - 1) / numThreads;
//     computeFeatureKernel<<<numBlocks, numThreads>>>(h, &data[0][0], data.size(), sx, sy, results, numFeatures);
//     cudaDeviceSynchronize();
// }
