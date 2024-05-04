#include <cuda_runtime.h>
#include "weak-classifier.h"
#include "haar-like.h"

/**
 * Constructor
 * @param haarlike A Haar-like feature
 * @param f The computed feature value
 * @param label True/false corresponds to positive/negative classification
 * @param weight The voting weight associated with this weak classifier
 */
WeakClassifier::WeakClassifier(Haarlike haarlike, float f, bool label, float weight) {
	haarlike = haarlike;
	threshold = f;
	label = label;
	weight = weight;
	minErr = 1;
	polarity = 0;
}

/**
 * Constructor
 */
WeakClassifier::WeakClassifier() {

}

/**
 * Classify a feature value
 * @param  featureValueThe feature value
 * @return 1 is a positive classification, -1 is negative
 */
__device__ int WeakClassifier::classify(float featureValue) {
	if (featureValue * float(this->polarity) < this->threshold * float(this->polarity)) return 1;
	else return -1;
}

/**
 * Scale a weak classifier relative to its base resolution
 * @param factor The factor by which to scale
 */
void WeakClassifier::scale(float factor) {
	this->threshold *= (factor * factor);
	this->haarlike.scale(factor);
}

/**
 * Compare weak classifiers based on their threshold value
 * @param a First weak classifier
 * @param b Second weak classifier
 * @return True if the first weak classifier's threshold is the smaller of the two
 */
bool comparePotentialWeakClassifiers(const WeakClassifier& a, const WeakClassifier& b) {
	return a.threshold < b.threshold;
}