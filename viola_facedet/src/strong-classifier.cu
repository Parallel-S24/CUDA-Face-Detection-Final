#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>

#include "strong-classifier.h"
#include "weak-classifier.h"
#include "integral-image.h"
#include "device_launch_parameters.h"

/**
 * Kernel to evaluate weak classifiers on device.
 * @param classifiers Array of weak classifiers.
 * @param count Number of weak classifiers.
 * @param integral Integral image object.
 * @param d_data Device pointer to the input data.
 * @param sx Starting X-coordinate for feature computation.
 * @param sy Starting Y-coordinate for feature computation.
 * @param mean Mean value for normalization.
 * @param sd Standard deviation for normalization.
 * @param weights Array of weights for the classifiers.
 * @param threshold Threshold for classification.
 * @param results Device pointer to store the classification results.
 */
__global__ void evaluateWeakClassifiersKernel(WeakClassifier* classifiers, int count,IntegralImage integral, float* d_data, int sx, int sy, float mean, float sd, float* weights, float threshold, float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= 0 || idx >= count) {
        return;
    }
    float f = integral.computeFeatureDevice(classifiers[idx].haarlike, d_data, sx, sy);
    if (classifiers[idx].haarlike.type == 2) {
        f += (classifiers[idx].haarlike.w * 3 * classifiers[idx].haarlike.h * mean) / 3;
    } else if (classifiers[idx].haarlike.type == 4) {
        f += (classifiers[idx].haarlike.w * classifiers[idx].haarlike.h * 3 * mean) / 3;
    }
    if (sd != 0) f /= sd;
    results[idx] = classifiers[idx].classify(f) * weights[idx];
}
////////////////////////////////////////////////////////////////////////////////////////

/**
 * Constructor for StrongClassifier class.
 * @param thresh Threshold value for the strong classifier.
 */
StrongClassifier::StrongClassifier(float thresh) : threshold(thresh){}

/**
 * Scale each weak classifier (if applicable).
 * @param factor Scaling factor.
 */
void StrongClassifier::scale(float factor) {
    for (auto& wc : weakClassifiers) {
        wc.scale(factor);
    }
}

/**
 * Classify function for StrongClassifier.
 * @param integral IntegralImage object representing the input image.
 * @param sx Starting X-coordinate for feature computation.
 * @param sy Starting Y-coordinate for feature computation.
 * @param mean Mean value for normalization.
 * @param sd Standard deviation for normalization.
 * @return True if the classification result exceeds the threshold, false otherwise.
 */
bool StrongClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
    float* d_results;
    cudaMalloc(&d_results, weakClassifiers.size() * sizeof(float));

    WeakClassifier* d_classifiers;
    cudaMalloc(&d_classifiers, weakClassifiers.size() * sizeof(WeakClassifier));
    cudaMemcpy(d_classifiers, weakClassifiers.data(), weakClassifiers.size() * sizeof(WeakClassifier), cudaMemcpyHostToDevice);

    float* d_weights;
    cudaMalloc(&d_weights, weights.size() * sizeof(float));
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    // TODO configure this
    int numThreads = 256;
    int numBlocks = (weakClassifiers.size() + numThreads - 1) / numThreads;

    float* d_data;
    float width = integral.data.size();
    float height = integral.data[0].size();
    cudaMalloc(&d_data, width * height * sizeof(float));

    // Copy the data from host to device
    cudaMemcpy(d_data, integral.data[0].data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    evaluateWeakClassifiersKernel<<<numBlocks, numThreads>>>(d_classifiers, weakClassifiers.size(), integral, d_data, sx, sy, mean, sd, d_weights, threshold, d_results);

    float* results = new float[weakClassifiers.size()];
    cudaMemcpy(results, d_results, weakClassifiers.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    float score = 0;
    for (int k = 0; k< 8; k++){
        score += *results++;
    }

    cudaFree(d_results);
    cudaFree(d_classifiers);
    cudaFree(d_weights);
    // TODO do we need to delete results?
        // delete[] results;
    if (score >= threshold) return true;
	else return false;
}
