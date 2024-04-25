#include <vector>
#include <cmath>
// #include <opencv2/opencv.hpp>

#include "utility.h"

/**
 * Convert an HTML5 ImageData offset to a 2D vector
 * @param {Int} offset The offset
 * @param {Int} w      Width of the ImageData object
 */
std::vector<int> offsetToVec2(int offset, int w) {
	std::vector<int> vec2(2);
	int pixelOffset = offset / 4;
	vec2[0] = pixelOffset % w;
	vec2[1] = pixelOffset / w;
	return vec2;
}

/**
 * Convert an HTML5 ImageData buffer to floating point pseudograyscale format (discard RGB and store luma in 4th float)
 * @param  {cv::Mat&} image 
 * @param  {Int}            w        Width of ImageData object
 * @param  {Int}            h        Height of ImageData object
 * @return {Unsigned char*}          Pointer to a new buffer
 */

// float* toGrayscaleFloat(const cv::Mat& image, int w, int h) {
//     int size = w * h *4;
//     float* gs = new float[size];  // Allocate space for grayscale values only

//     for (int y = 0; y < h; ++y) {
//         for (int x = 0; x < w; ++x) {
// 			int index = (y*w +x)*4;
//             // Get pixel at (x, y) and calculate grayscale value
//             cv::Vec3b color = image.at<cv::Vec3b>(y, x);
//             float r = color[2] * 0.2126f;  // Red component
//             float g = color[1] * 0.7152f;  // Green component
//             float b = color[0] * 0.0722f;  // Blue component
//             float luma = r + g + b;  
// 			gs[index] = r;  
// 			gs[index+1] = g;  
// 			gs[index+2] = b;     
//             gs[index+3] = 255.0f - luma;     // Store grayscale value at correct index
//         }
//     }
//     return gs;
// }