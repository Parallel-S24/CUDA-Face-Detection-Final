#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>

#include "strong-classifier.h"
#include "weak-classifier.h"
#include "integral-image.h"
#include "device_launch_parameters.h"

// Define a kernel to evaluate weak classifiers in parallel
__global__ void evaluateWeakClassifiersKernel(WeakClassifier* classifiers, int count, IntegralImage integral, float* d_data, int sx, int sy, float mean, float sd, float* weights, float threshold, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // float* dataPtr = integral.data[0].data(); 
        float f = integral.computeFeatureDevice(classifiers[idx].haarlike, d_data, sx, sy);
        // float f = 0.0f;
        if (classifiers[idx].haarlike.type == 2) {
            f += (classifiers[idx].haarlike.w * 3 * classifiers[idx].haarlike.h * mean) / 3;
        } else if (classifiers[idx].haarlike.type == 4) {
            f += (classifiers[idx].haarlike.w * classifiers[idx].haarlike.h * 3 * mean) / 3;
        }
        if (sd != 0) f /= sd;
        float score = classifiers[idx].classify(f) * weights[idx];
        results[idx] = score >= threshold;
    }
}
////////////////////////////////////////////////////////////////////////////////////////

StrongClassifier::StrongClassifier(float thresh) : threshold(thresh){}

StrongClassifier::~StrongClassifier() {
}

// Add a weak classifier
void StrongClassifier::addWeakClassifier(WeakClassifier wc, float w) {
    weakClassifiers.push_back(wc);
    weights.push_back(w);
}
// Scale each weak classifier (if applicable)
void StrongClassifier::scale(float factor) {
    // This would typically involve scaling features used by weak classifiers
    for (auto& wc : weakClassifiers) {
        wc.scale(factor);
    }
}

bool StrongClassifier::classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
    bool* d_results;
    cudaMalloc(&d_results, weakClassifiers.size() * sizeof(bool));

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

    bool* results = new bool[weakClassifiers.size()];
    cudaMemcpy(results, d_results, weakClassifiers.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    bool finalResult = false;
    for (int i = 0; i < weakClassifiers.size(); i++) {
        if (results[i]) {
            finalResult = true;
            break;
        }
    }

    cudaFree(d_results);
    cudaFree(d_classifiers);
    cudaFree(d_weights);
    delete[] results;

    return finalResult;
}
