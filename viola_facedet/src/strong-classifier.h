#pragma once

#include <vector>

#include "weak-classifier.h"
#include "integral-image.h"
#include "cascade-classifier.h"

class CascadeClassifier;

class StrongClassifier {
	public:
		StrongClassifier(float initialThreshold = 0.5);
		void scale(float factor);
		bool classify(IntegralImage& integral, int sx, int sy, float mean, float sd);

		std::vector<WeakClassifier> weakClassifiers;
		std::vector<float> weights;
		float threshold;
};