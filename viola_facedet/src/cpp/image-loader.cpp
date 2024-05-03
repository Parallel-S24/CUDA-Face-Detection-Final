#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "wasmface.h"
#include "cascade-classifier.h"
#include <array>

// g++ -std=c++11 -o image-loader image-loader.cpp wasmface.cpp ../cuda/cascade-classifier.cu ../cuda/integral-image.cu ../cuda-strong-classifier.cu weak-classifier.cpp haar-like.cpp utility.cpp

// g++ -std=c++11 -o image-loader image-loader.cpp wasmface.cpp cascade-classifier.cpp integral-image.cpp strong-classifier.cu weak-classifier.cpp haar-like.cpp utility.cpp 

float* readFloatArrayFromFile(const std::string& filename, int& size) {
    std::ifstream file("output.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading." << std::endl;
        size = 0;
        return nullptr;  // Return nullptr if file couldn't be opened
    }

    std::vector<float> tempArray;
    float value;
    while (file >> value) {
        tempArray.push_back(value);
    }
    file.close();

    size = tempArray.size();
    if (size == 0) return nullptr;  // No data read, return nullptr

    float* array = new float[size];
    for (int i = 0; i < size; ++i) {
        array[i] = tempArray[i];
    }

    return array;
}
std::string readModelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    // Example usage
    int size = 1920*1080*3;
    float* gray = readFloatArrayFromFile("output.txt",size);
    std::string jsonContent = readModelFile("../../models/human-face.json");
    CascadeClassifier* face_cascade = wasmface::create(jsonContent.c_str());
    if (!face_cascade) {
        std::cerr << "Failed to load the model." << std::endl;
        return 1;
    }
    uint16_t* boxes = wasmface::detect(gray, 1920, 1080, face_cascade, 2.0, 2.0, true, 0.3, 15);
    printf("SUCCESS!");
    // for (int i = 0; i < size; ++i) {
    //     std::cout << boxes[i];
    //     if (i < size - 1)
    //         std::cout << ", ";
    // }
    // std::cout << std::endl;

    return 0;
}
