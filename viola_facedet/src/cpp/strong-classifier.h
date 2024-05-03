#pragma once

#include <vector>

#include "weak-classifier.h"
#include "integral-image.h"
#include "cascade-classifier.h"
// #include "device_launch_parameters.h"

class CascadeClassifier;

class StrongClassifier {
	public:
		StrongClassifier(float initialThreshold = 0.5);
		~StrongClassifier();
		void scale(float factor);
		void addWeakClassifier(WeakClassifier weakClassifier, float weight);
		bool classify(IntegralImage& integral, int sx, int sy, float mean, float sd);
		void optimizeThreshold(std::vector<IntegralImage>& positiveValidationSet, float targetFNR);
		float getFPR(std::vector<IntegralImage>& negativeValidationSet);
		float getFNR(std::vector<IntegralImage>& positiveValidationSet);

		std::vector<WeakClassifier> weakClassifiers;
		std::vector<float> weights;
		float threshold;

		// void StrongClassifier::updateThreshold(float w);
		// float StrongClassifier::getThreshold();
};