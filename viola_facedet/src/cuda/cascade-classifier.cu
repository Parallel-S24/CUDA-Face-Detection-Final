#include <vector>
#include <cuda_runtime.h>

#include "cascade-classifier.h"
#include "integral-image.h"
#include "strong-classifier.h"

// Define the kernel to scale strong classifiers in parallel
__global__ void scaleStrongClassifiersKernel(StrongClassifier* classifiers, int count, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        classifiers[idx].scale(factor);
    }
}

// Define the kernel to classify a region in parallel
__global__ void classifyRegionKernel(StrongClassifier* classifiers, int count, IntegralImage integral, int sx, int sy, float mean, float sd, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = classifiers[idx].classify(integral, sx, sy, mean, sd);
    }
}

class CascadeClassifier {
public:
    int baseResolution;
    std::vector<StrongClassifier> strongClassifiers;

    // Constructors
    CascadeClassifier(int baseResolution) : baseResolution(baseResolution) {}

    CascadeClassifier(int baseResolution, std::vector<StrongClassifier> sc) 
        : baseResolution(baseResolution), strongClassifiers(sc) {}

    // Scale the classifier
    void scale(float factor) {
        baseResolution *= factor;
        StrongClassifier* d_classifiers;
        cudaMalloc(&d_classifiers, strongClassifiers.size() * sizeof(StrongClassifier));
        cudaMemcpy(d_classifiers, strongClassifiers.data(), strongClassifiers.size() * sizeof(StrongClassifier), cudaMemcpyHostToDevice);

        int numThreads = 256;
        int numBlocks = (strongClassifiers.size() + numThreads - 1) / numThreads;
        scaleStrongClassifiersKernel<<<numBlocks, numThreads>>>(d_classifiers, strongClassifiers.size(), factor);

        cudaMemcpy(strongClassifiers.data(), d_classifiers, strongClassifiers.size() * sizeof(StrongClassifier), cudaMemcpyDeviceToHost);
        cudaFree(d_classifiers);
    }

    // Add a strong classifier
    void add(StrongClassifier sc) {
        strongClassifiers.push_back(sc);
    }

    // Remove the last strong classifier
    void removeLast() {
        strongClassifiers.pop_back();
    }

    // Classify a region of an integral image
    bool classify(IntegralImage& integral, int sx, int sy, float mean, float sd) {
        bool* results = new bool[strongClassifiers.size()];
        bool* d_results;
        cudaMalloc(&d_results, strongClassifiers.size() * sizeof(bool));
        
        StrongClassifier* d_classifiers;
        cudaMalloc(&d_classifiers, strongClassifiers.size() * sizeof(StrongClassifier));
        cudaMemcpy(d_classifiers, strongClassifiers.data(), strongClassifiers.size() * sizeof(StrongClassifier), cudaMemcpyHostToDevice);

        int numThreads = 256;
        int numBlocks = (strongClassifiers.size() + numThreads - 1) / numThreads;
        classifyRegionKernel<<<numBlocks, numThreads>>>(d_classifiers, strongClassifiers.size(), integral, sx, sy, mean, sd, d_results);

        cudaMemcpy(results, d_results, strongClassifiers.size() * sizeof(bool), cudaMemcpyDeviceToHost);

        // Assume classification requires all classifiers to agree on positive detection
        for (int i = 0; i < strongClassifiers.size(); i++) {
            if (!results[i]) {
                cudaFree(d_results);
                cudaFree(d_classifiers);
                delete[] results;
                return false;
            }
        }

        cudaFree(d_results);
        cudaFree(d_classifiers);
        delete[] results;
        return true;
    }
};