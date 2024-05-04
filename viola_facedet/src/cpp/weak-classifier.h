#pragma once

#include "haar-like.h"
#include <cuda_runtime.h>

class WeakClassifier {
	public:
		WeakClassifier(Haarlike haarlike, float f, bool label, float weight);
		WeakClassifier();
		__device__ int classify(float featureValue);
		void scale(float factor);
		Haarlike haarlike;
		bool label;
		float weight;
		float threshold;
		int polarity;
		float minErr;
};

bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b);