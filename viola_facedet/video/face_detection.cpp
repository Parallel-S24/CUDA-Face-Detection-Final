#include <opencv2/opencv.hpp>
#include <fstream> 

//Compile: g++ -std=c++11 -o face_detection face_detection.cpp ../src/cpp/utility.cpp ../src/cpp/facedet.cpp ../src/cpp/integral-image.cpp ../src/cpp/haar-like.cpp ../src/cpp/weak-classifier.cpp ../src/cpp/strong-classifier.cpp ../src/cpp/cascade-classifier.cpp `pkg-config --cflags --libs opencv4`
//Run: ./face_detection

#include "../src/cpp/facedet.h"
#include "../src/cpp/cascade-classifier.h"
#include <iostream>
#include <fstream>

void saveFloatArrayToFile(const float* array, int size, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        file << array[i];
        if (i < size - 1) {
            file << "\n";  // New line for each element, you can use ' ' for space separated
        }
    }

    file.close();
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
float* toGrayscaleFloat(const cv::Mat& image, int w, int h) {
    int size = w * h *4;
    float* gs = new float[size];  // Allocate space for grayscale values only

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
			int index = (y*w +x)*4;
            // Get pixel at (x, y) and calculate grayscale value
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            float r = color[2] * 0.2126f;  // Red component
            float g = color[1] * 0.7152f;  // Green component
            float b = color[0] * 0.0722f;  // Blue component
            float luma = r + g + b;  
			gs[index] = r;  
			gs[index+1] = g;  
			gs[index+2] = b;     
            gs[index+3] = 255.0f - luma;     // Store grayscale value at correct index
        }
    }
    return gs;
}

int main() {
    // Load the face detection model
    std::string jsonContent = readModelFile("../models/human-face.json");
    // std::vector<int> jsonArray = intToStringArray(jsonContent);
    CascadeClassifier* face_cascade = facedet::create(jsonContent.c_str());
    if (!face_cascade) {
        std::cerr << "Failed to load the model." << std::endl;
        return 1;
    }
    // if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
    //     std::cerr << "Error loading cascade file. Exiting!" << std::endl;
    //     return -1;
    // }

    // Open the default video camera
    cv::VideoCapture webcam(0);
    if (!webcam.isOpened()) {
        std::cerr << "Error opening video camera. Exiting!" << std::endl;
        return -1;
    }

    cv::namedWindow("web cam", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Gray video", cv::WINDOW_AUTOSIZE);
    // int frame_num = 0;
    while (true) {
        cv::Mat frame;
        // Read a new frame from the video camera
        bool ret = webcam.read(frame);
        if (!ret) {
            std::cerr << "Failed to read from camera. Exiting!" << std::endl;
            break;
        }

        // Convert to grayscale
        // cv::Mat gray;
        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        int height = frame.rows;
        int width = frame.cols;
        // cv::imshow("Gray video", gray);

        // Face detection
        std::vector<cv::Rect> faces;
        auto gray = toGrayscaleFloat(frame, width, height);
        int size = width * height *4;
        // if(frame_num == 0){
        //     saveFloatArrayToFile(gray, size, "output.txt");
        // }

        uint16_t* boxes = facedet::detect(gray, width, height, face_cascade, 2.0, 2.0, true, 0.3, 15);
        int num_boxes = (boxes[0] - 1) / 3;
    
        for (int i = 0; i < num_boxes; i++) {
            int index = 1 + i * 3;  // Calculate the index for each box (skip the first element)
            int x = boxes[index];
            int y = boxes[index + 1];
            int width = boxes[index + 2];
            int height = width;  // Assuming height equals width, change if necessary

            faces.push_back(cv::Rect(x, y, width, height));  // Create a cv::Rect and add it to the vector
        }

        // Draw rectangles around detected faces
        for (const auto& rect : faces) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 240), 2);
        }

        // Display the resulting frame
        cv::imshow("web cam", frame);
        // frame_num ++;

        // Break loop on 'd' key press
        if (cv::waitKey(1) == 'd') {
            break;
        }
    }

    // Release the video capture object
    webcam.release();
    // Destroy all windows
    cv::destroyAllWindows();

    return 0;
}
