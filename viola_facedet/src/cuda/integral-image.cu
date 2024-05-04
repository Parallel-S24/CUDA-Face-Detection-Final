#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "integral-image.h"

// CUDA kernel for computing the integral image
__global__ void computeIntegralImageKernel(const float* input, float* output, int width, int height) {
    extern __shared__ float temp[];  // Shared memory for row-wise cumulative sum

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    // Load input into shared memory (assuming each block handles one row)
    temp[threadIdx.x] = input[index];
    __syncthreads();

    // Perform scan operation on the row
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0;
        if (threadIdx.x >= offset)
            val = temp[threadIdx.x - offset];
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }

    // Write the results from shared memory to global memory
    output[index] = temp[threadIdx.x];

    // Add the previous row's cumulative sum to current value (if not the first row)
    if (y > 0) {
        output[index] += output[index - width];
    }
}

// CUDA kernel to compute the sum of values within a rectangular region of an integral image
__global__ void computeRectangleSumKernel(const float* integralImage, float* output, const int* rectangles, int numRectangles, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRectangles) {
        int x = rectangles[idx * 4];
        int y = rectangles[idx * 4 + 1];
        int w = rectangles[idx * 4 + 2];
        int h = rectangles[idx * 4 + 3];

        int topLeft = (y - 1) * width + (x - 1);
        int topRight = (y - 1) * width + (x + w - 1);
        int bottomLeft = (y + h - 1) * width + (x - 1);
        int bottomRight = (y + h - 1) * width + (x + w - 1);

        float A = (x > 0 && y > 0) ? integralImage[topLeft] : 0;
        float B = (y > 0) ? integralImage[topRight] : 0;
        float C = (x > 0) ? integralImage[bottomLeft] : 0;
        float D = integralImage[bottomRight];

        output[idx] = D + A - (B + C);
    }
}

// Host function to setup and invoke CUDA kernels
void processImage(float* h_input, int width, int height) {
    float *d_input, *d_output;
    int size = width * height * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Setup block and grid dimensions
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel
    computeIntegralImageKernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_input, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}