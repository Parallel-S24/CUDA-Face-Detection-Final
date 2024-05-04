#pragma once

#include <vector>

#include "haar-like.h"
#include "utility.h"
#include <cuda_runtime.h>

class IntegralImage {
	public:
		IntegralImage(float inputBuf[], int w, int h, int size, bool squared);
		__device__ float getRectangleSumDevice(const float* data, int x, int y, int w, int h);
		__device__ float computeFeatureDevice(Haarlike h, const float* data, int sx, int sy);
		void computeFeatureKernel(Haarlike* h, int w, int sx, int sy, float* results, int numFeatures);
		std::vector<std::vector<float>> data;
};

