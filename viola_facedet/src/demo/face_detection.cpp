#include <opencv2/opencv.hpp>
#include "../wasmface.h"
#include "../cascade-classifier.h"
#include <iostream>
#include <fstream>

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
    // Load the face detection model
    std::string jsonContent = readModelFile("../models/human-face.json");
    CascadeClassifier* face_cascade = wasmface::create(jsonContent.c_str());
    if (!face_cascade) {
        std::cerr << "Failed to load the model." << std::endl;
        return 1;
    }

    // Open the default video camera
    cv::VideoCapture webcam(0);
    if (!webcam.isOpened()) {
        std::cerr << "Error opening video camera. Exiting!" << std::endl;
        return -1;
    }

    cv::namedWindow("web cam", cv::WINDOW_AUTOSIZE);

    // Allocate memory for grayscale image outside the loop
    int width = webcam.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = webcam.get(cv::CAP_PROP_FRAME_HEIGHT);
    float* grayscaleImage = new float[width * height * 4];

    while (true) {
        cv::Mat frame;
        bool ret = webcam.read(frame);
        if (!ret) {
            std::cerr << "Failed to read from camera. Exiting!" << std::endl;
            break;
        }

        // // Convert to grayscale
        // toGrayscaleFloat(frame, grayscaleImage);

        // Face detection
        uint16_t* boxes = wasmface::detect(frame, width, height, face_cascade, 2.0, 2.0, true, 0.3, 15);
        int num_boxes = (boxes[0] - 1) / 3;

        std::vector<cv::Rect> faces;
        for (int i = 0; i < num_boxes; i++) {
            int index = 1 + i * 3;
            int x = boxes[index];
            int y = boxes[index + 1];
            int width = boxes[index + 2];
            int height = width;  // Assuming height equals width, change if necessary
            faces.push_back(cv::Rect(x, y, width, height));
        }

        // Draw rectangles around detected faces
        for (const auto& rect : faces) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 240), 2);
        }

        // Display the resulting frame
        cv::imshow("web cam", frame);

        // Break loop on 'd' key press
        if (cv::waitKey(1) == 'd') {
            break;
        }
    }

    // Release resources
    webcam.release();
    cv::destroyAllWindows();
    delete[] grayscaleImage;

    return 0;
}
