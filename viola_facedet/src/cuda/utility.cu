#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "utility.h"

// CUDA kernel for converting an image to floating point pseudograyscale
__global__ void toGrayscaleFloatKernel(cv::Vec3b *input, float *output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = (y * w + x);
        cv::Vec3b color = input[idx];
        float r = color[2] * 0.2126f;
        float g = color[1] * 0.7152f;
        float b = color[0] * 0.0722f;
        float luma = r + g + b;

        int outIdx = idx * 4;
        output[outIdx] = r;
        output[outIdx + 1] = g;
        output[outIdx + 2] = b;
        output[outIdx + 3] = 255.0f - luma;
    }
}

// Host function to process the image
float* toGrayscaleFloatCUDA(const cv::Mat& image, int w, int h) {
    // Allocate space for the output
    int size = w * h * 4;
    float* d_output;
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy image data to device memory
    cv::Vec3b *d_input;
    cudaMalloc(&d_input, w * h * sizeof(cv::Vec3b));
    cudaMemcpy(d_input, image.ptr<cv::Vec3b>(), w * h * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    toGrayscaleFloatKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, w, h);
    cudaDeviceSynchronize();

    // Copy result back to host memory
    float* h_output = new float[size];
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return h_output;
}