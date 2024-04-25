// #pragma once

// #include <vector>
#include <opencv2/opencv.hpp>

// std::vector<int> offsetToVec2(int offset, int w);
// // unsigned char rgbToLuma(unsigned char r, unsigned char g, unsigned char b);
// // unsigned char* toGrayscale(unsigned char inputBuf[], int w, int h);
// float* toGrayscaleFloat(const cv::Mat& image, int w, int h);
// // float* imageDataToNormalizedBuffer(unsigned char inputBuf[], int w, int h);
#ifndef __UTILITY_H__
#define __UTILITY_H__

void toGrayscaleFloatKernel(cv::Vec3b *input, float *output, int w, int h);
float* toGrayscaleFloatCUDA(const cv::Mat& image, int w, int h);
// void vec2CellNoise(float location[3], float result[2], int index);
// void getNoiseTables(int** permX, int** permY, float** value1D);

#endif
