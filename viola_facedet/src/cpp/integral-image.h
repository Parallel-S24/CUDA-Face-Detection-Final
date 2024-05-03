#pragma once

#include <vector>

#include "haar-like.h"
#include "utility.h"

class IntegralImage {
	public:
		IntegralImage(float inputBuf[], int w, int h, int size, bool squared);
		// float computeFeature(Haarlike& haarlike, int sx, int sy);
		// std::vector<Haarlike> computeEntireFeatureSet(int s, int sx, int sy);
		// float getRectangleSum(int x, int y, int w, int h);
		// std::vector<std::vector<float>> data;
		float getRectangleSumDevice(const float* data, int w, int x, int y, int h, int sx, int sy);
		float computeFeatureDevice(Haarlike h, const float* data, int sx, int sy);
		void computeFeatureKernel(Haarlike* h, int w, int sx, int sy, float* results, int numFeatures);
		std::vector<std::vector<float>> data;
		// void computeFeatureBatch(Haarlike* h, int numFeatures, int sx, int sy, float* results) const;
};

