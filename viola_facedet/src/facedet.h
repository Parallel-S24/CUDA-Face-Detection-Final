#pragma once
#include <array>
#include "utility.h"
#include "integral-image.h"
#include "strong-classifier.h"
#include "cascade-classifier.h"

// #include <opencv2/opencv.hpp>

    class CascadeClassifier;

    namespace facedet {
        bool compareDereferencedPtrs(int* a, int* b);
        std::vector<std::array<int, 3>> nonMaxSuppression(std::vector<std::array<int, 3>>& boxes, float thresh, int nthresh);
        CascadeClassifier* create(const char model[]);
        uint16_t* detect(float* fpgs, int w, int h, CascadeClassifier* cco, 
                                    float step, float delta, bool pp, float othresh, int nthresh);
    }   
