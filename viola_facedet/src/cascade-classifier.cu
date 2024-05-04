#include <vector>

#include "cascade-classifier.h"
#include "integral-image.h"
#include "strong-classifier.h"
#include <cstdio>

/**
 * Constructor
 * @param baseResolution Base resolution for the cascade classifier
 * @param sc A set of strong classifiers to add as layers
 */
CascadeClassifier::CascadeClassifier(int baseResolution, std::vector<StrongClassifier> sc) {
	this->baseResolution = baseResolution;
	this->strongClassifiers = sc;
}

/**
 * Destructively scale a cascade classifier relative to its base resolution
 * @param factor The factor by which to scale
 */
void CascadeClassifier::scale(float factor) {
	this->baseResolution *= factor;
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) this->strongClassifiers[i].scale(factor);
}

/**
 * Classify a region of an integral image
 * @param integral The integral image to classify
 * @param sx       Subwindow x offset
 * @param sy       Subwindow y offset
 * @param mean     The mean of the values within the subwindow (for post-normalization)
 * @param sd       The standard deviation of the values within the subwindow (for post normalization)
 * @return True for positive detection, false for negative
 */
bool CascadeClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
	for (int i = 0; i < this->strongClassifiers.size(); i += 1) {
		if (this->strongClassifiers[i].classify(integral, sx, sy, mean, sd) == false) return false;
	}
	return true;
}

/**
 * Get false positive rate for a cascade classifier
 * @param negativeValidationSet A set of negative images to test
 * @return Normalized false positive rate
 */
float CascadeClassifier::getFPR(std::vector<IntegralImage>& negativeValidationSet) {
	int falsePositives = 0;
	for (int i = 0; i < negativeValidationSet.size(); i += 1) {
		if (this->classify(negativeValidationSet[i], 0, 0, 0, 1) == true) falsePositives += 1;
	}
	return (float)falsePositives / negativeValidationSet.size();
}

/**
 * Get false negative rate for a cascade classifier
 * @param positiveValidationSet A set of positive images to test
 * @return Normalized false negative rate
 */
float CascadeClassifier::getFNR(std::vector<IntegralImage>& positiveValidationSet) {
	int falseNegatives = 0;
	for (int i = 0; i < positiveValidationSet.size(); i += 1) {
		if (this->classify(positiveValidationSet[i], 0, 0, 0, 1) == false) falseNegatives += 1;
	}
	return (float)falseNegatives / positiveValidationSet.size();
}