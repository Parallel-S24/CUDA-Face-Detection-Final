#pragma once
#include <array>
#include "utility.h"
#include "integral-image.h"
#include "strong-classifier.h"
#include "cascade-classifier.h"

// #include "/Users/ria/Documents/School/15418/emsdk/upstream/emscripten/cache/sysroot/include/emscripten/emscripten.h"
// #include <opencv2/opencv.hpp>

    class CascadeClassifier;

    // bool compareDereferencedPtrs(int* a, int* b);
    // std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh);
    // CascadeClassifier* create(char model[]);
    // void destroy(CascadeClassifier* cc);
    // std::vector<cv::Rect> detect(unsigned char inputBuf[], int w, int h, CascadeClassifier* cco, 
    //                                     float step, float delta, bool pp, float othresh, int nthresh);

    namespace wasmface {
        bool compareDereferencedPtrs(int* a, int* b);
        std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh);
        CascadeClassifier* create(const char model[]);
        void destroy(CascadeClassifier* cc);
        uint16_t* detect(float* fpgs, int w, int h, CascadeClassifier* cco, 
                                    float step, float delta, bool pp, float othresh, int nthresh);
    }   
