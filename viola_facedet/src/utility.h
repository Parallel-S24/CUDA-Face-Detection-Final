#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<int> offsetToVec2(int offset, int w);
float* toGrayscaleFloat(const cv::Mat& image, int w, int h);