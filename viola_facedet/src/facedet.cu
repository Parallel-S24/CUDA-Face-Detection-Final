#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "../lib/json.hpp"

#include "facedet.h"
#include "utility.h"
#include "integral-image.h"
#include "strong-classifier.h"
#include "cascade-classifier.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

namespace facedet {
/**
 * Compare two pointers based on their dereferenced values
 * @param a First pointer
 * @param b Second pointer
 * @return True if the first pointer's dereferenced value is the smaller of the two
 */
bool compareDereferencedPtrs(int* a, int* b) {
	return *a < *b;
}

/**
 * Apply non-maximum suppression to a set of 1:1 aspect ratio bounding boxes
 * Bounding boxes are represented as [x, y, s] where s = width and height
 * @param boxes   The set of bounding boxes
 * @param thresh  The minimum overlap ratio required for suppression
 * @param nthresh The minimum number of neighboring boxes required for suppression
 * @return The suppressed set of bounding boxes
 */
std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh) {
	int len = boxes.size();
	if (!len) return boxes;

	// Destructure our bounding boxes into arrays representing upper left and lower right coords
	int* x1 = new int[len];
	int* y1 = new int[len];
	int* x2 = new int[len];
	int* y2 = new int[len];
	int* area = new int[len];
	for (int i = 0; i < len; i += 1) {
		x1[i] = boxes[i][0];
		y1[i] = boxes[i][1];
		x2[i] = boxes[i][0] + boxes[i][2] - 1;
		y2[i] = boxes[i][1] + boxes[i][2] - 1;
		area[i] = std::pow(boxes[i][2], 2);
	}

	// Create an array of the indices that would sort our y2 coords
	int** ptrs = new int*[len];
	for (int i = 0; i < len; i += 1) ptrs[i] = &y2[i];
	std::sort(ptrs, ptrs + len, compareDereferencedPtrs);
	std::vector<int> ind(len);
	for (int i = 0; i < len; i += 1) ind[i] = ptrs[i] - &y2[0];

	std::vector<std::pair<int, int>> pick;
	while (ind.size() > 0) {
		int last = ind.size() - 1;
		int n = ind[last]; 
		pick.push_back(std::pair<int, int> (n, 0));  
		// Suppress bounding boxes that overlap 
		int neighborsCount = 0;
		std::vector<int> suppress = {last};
		for (int i = 0; i < last; i += 1) {
			int j = ind[i];
			int xx1 = std::max(x1[n], x1[j]);
			int yy1 = std::max(y1[n], y1[j]);
			int xx2 = std::min(x2[n], x2[j]);
			int yy2 = std::min(y2[n], y2[j]);
			int w = std::max(0, xx2 - xx1 + 1);
			int h = std::max(0, yy2 - yy1 + 1);

			float overlap = float(w * h) / area[j];

			if (overlap > thresh) {
				suppress.push_back(i);
				neighborsCount += 1;
			}
		}
		std::sort(suppress.begin(), suppress.end(), std::greater<int>());

		for (int i = 0; i < suppress.size(); i += 1) 
			ind.erase(ind.begin() + suppress[i]);

		pick.back().second = neighborsCount;
	}

	delete [] x1;
	delete [] y1;
	delete [] x2;
	delete [] y2;
	delete [] area;
	delete [] ptrs;

	// Also suppress boxes that do not have the minimum number of neighbors
	std::vector<std::array<int, 3>> result;
	for (int i = 0; i < pick.size(); i += 1) {
		if (pick[i].second >= nthresh) result.push_back(boxes[pick[i].first]);
	}
	return result;
} 

/**
 * Deserialize and construct a cascade classifier object
 * @param model A serialized cascade classifier object
 * @return A pointer to a new cascade classifier object
 */
CascadeClassifier* create(const char model[]) {
	auto ccJSON = nlohmann::json::parse(model);

	std::vector<StrongClassifier> sc;
	for (int i = 0; i < ccJSON["strongClassifiers"].size(); i += 1) {
		float threshold = ccJSON["strongClassifiers"][i]["threshold"];
		StrongClassifier strongClassifier(threshold);
		for (int j = 0; j < ccJSON["strongClassifiers"][i]["weakClassifiers"].size(); j += 1) {
			WeakClassifier weakClassifier;
			weakClassifier.haarlike.type = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["type"];
			weakClassifier.haarlike.w = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["w"];
			weakClassifier.haarlike.h = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["h"];
			weakClassifier.haarlike.x = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["x"];
			weakClassifier.haarlike.y = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["y"];
			weakClassifier.threshold = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["threshold"];
			weakClassifier.polarity = ccJSON["strongClassifiers"][i]["weakClassifiers"][j]["polarity"];
			strongClassifier.weakClassifiers.push_back(weakClassifier);
			strongClassifier.weights.push_back(ccJSON["strongClassifiers"][i]["weights"][j]);
		}
		sc.push_back(strongClassifier);
	}

	CascadeClassifier* cc = new CascadeClassifier(ccJSON["baseResolution"], sc);
	return cc;
}

/**
 * Kernel to compute rectangle sum for integral image on device.
 * @param integral_data Pointer to the integral image data.
 * @param integral IntegralImage object representing the integral image.
 * @param sum Array to store the computed rectangle sums.
 * @param rect_width Width of the rectangle.
 * @param rect_height Height of the rectangle.
 * @param width Width of the image.
 * @param height Height of the image.
 */
__global__ void getRectangleSumKernel(float* integral_data, IntegralImage integral, float* sum, int rect_width, int rect_height, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y * width;
    
    if (x >= width || y >= height) return;
    
    // Compute the rectangle sum for the specified region
    sum[idx] = integral.getRectangleSumDevice(integral_data, x, y, rect_width, rect_height);    
}

/**
 * Use a cascade classifier to detect objects in an HTML5 ImageData buffer
 * @param frame	 Image input from webcam 
 * @param w        Width of the ImageData object
 * @param h        Height of the ImageData object
 * @param cco      Pointer to a cascade classifier object
 * @param step     Detector scale step to apply
 * @param delta    Detector sweep delta to apply
 * @param pp       True applies post processing
 * @param othresh  Overlap threshold for post processing
 * @param nthresh  Neighbor threshold for post processing
 * @return Pointer to an array of bounding box geometry
 */
uint16_t* detect(cv::Mat frame, int w, int h, CascadeClassifier* cco, 
                                      float step, float delta, bool pp, float othresh, int nthresh) {
	CascadeClassifier* cc = new CascadeClassifier(*cco);
	float* fpgs = toGrayscaleFloat(frame, w, h);
	int byteSize = w * h * 4;
	auto integral = IntegralImage(fpgs, w, h, byteSize, false);
	auto integralSquared = IntegralImage(fpgs, w, h, byteSize, true);
	delete [] fpgs;

	// Sweep and scale the detector over the post-normalized input image and collect detections
	std::vector<std::array<int, 3>> roi;
	float* integral_data;
	float* integralsq_data;
	int width = integral.data.size();
	int height = integralSquared.data[0].size();
	dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	cudaMalloc(&integral_data, width * height * sizeof(float));
	cudaMalloc(&integralsq_data, width * height * sizeof(float));

	cudaMemcpy(integral_data, integral.data[0].data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(integralsq_data, integralSquared.data[0].data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

	float *sum = new float[width * height];
	float *squaredSum = new float[width * height];

	float *d_sum, *d_squaredSum;

	cudaMalloc(&d_sum, width * height * sizeof(float));
	cudaMalloc(&d_squaredSum, width * height * sizeof(float));
	while (cc->baseResolution < w && cc->baseResolution < h) {
			
		getRectangleSumKernel<<<gridDim, blockDim>>>(integral_data, integral, d_sum, cc->baseResolution, cc->baseResolution, width, height);
		getRectangleSumKernel<<<gridDim, blockDim>>>(integralsq_data, integralSquared, d_squaredSum, cc->baseResolution, cc->baseResolution, width, height);
		
		cudaDeviceSynchronize();

		cudaMemcpy(sum, d_sum, width * height * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(squaredSum, d_squaredSum, width * height * sizeof(float), cudaMemcpyDeviceToHost);

		// Process the results and update roi
		for (int y = 0; y < h - cc->baseResolution; y += step * delta) {
			for (int x = 0; x < w - cc->baseResolution; x += step * delta) {
				int idx = x + y * width;
				float area = std::pow(cc->baseResolution, 2);
				float mean = sum[idx] / area;
				float sd = std::sqrt(squaredSum[idx] / area - std::pow(mean, 2));
				bool c = cc->classify(integral, x, y, mean, sd);
				if (c) {
					// printf("found box\n");
					std::array<int, 3> bounding = {x, y, cc->baseResolution};
					roi.push_back(bounding);
				}
			}
		}
		cc->scale(step);
	}
	cudaFree(d_sum);
	cudaFree(d_squaredSum);
	if (pp) roi = nonMaxSuppression(roi, othresh, nthresh);

	int blen = roi.size() * 3 + 1;
	uint16_t* boxes = new uint16_t[blen];
	boxes[0] = blen;
	for (int i = 0, j = 1; i < roi.size(); i += 1, j += 3) {
		boxes[j] = roi[i][0];
		boxes[j + 1] = roi[i][1];
		boxes[j + 2] = roi[i][2];
	}
	
	delete cc;
	delete[] sum;
	delete[] squaredSum;
	return boxes;
}
}
