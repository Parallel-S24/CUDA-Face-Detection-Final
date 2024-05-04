#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "integral-image.h"
#include "utility.h"
#include "haar-like.h"

/**
 * Constructor
 * @param inputBuf Pointer to the input buffer containing pixel values.
 * @param w Width of the image.
 * @param h Height of the image.
 * @param size Size of the input buffer.
 * @param squared Flag indicating whether to compute the squared sum (optional, default: false).
 */
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

/**
 * Device function to compute the sum of a rectangle in the integral image.
 * @param data Pointer to the integral image data.
 * @param x X-coordinate of the top-left corner of the rectangle.
 * @param y Y-coordinate of the top-left corner of the rectangle.
 * @param w Width of the rectangle.
 * @param h Height of the rectangle.
 * @return Sum of the rectangle.
 */
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

/**
 * Device function to compute the value of a Haar-like feature.
 * @param h Haar-like object representing the feature.
 * @param data Pointer to the integral image data.
 * @param sx Starting X-coordinate for feature computation.
 * @param sy Starting Y-coordinate for feature computation.
 * @return Value of the Haar-like feature.
 */
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