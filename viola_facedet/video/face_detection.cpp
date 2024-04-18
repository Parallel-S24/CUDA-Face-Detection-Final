#include <opencv2/opencv.hpp>
#include <fstream> 

//Compile: g++ -std=c++11 -o face_detection face_detection.cpp ../src/cpp/wasmface.cpp ../src/cpp/utility.cpp ../src/cpp/integral-image.cpp ../src/cpp/haar-like.cpp ../src/cpp/weak-classifier.cpp ../src/cpp/strong-classifier.cpp ../src/cpp/cascade-classifier.cpp `pkg-config --cflags --libs opencv4`
//Run: ./face_detection

#include "../src/cpp/wasmface.h"
#include "../src/cpp/cascade-classifier.h"

std::string readModelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
std::vector<int> intToStringArray(const std::string& input) {
    std::vector<int> result;
    for (char c : input) {
        result.push_back(static_cast<int>(c));  // Cast char to int to store ASCII value
    }
    return result;
}

int main() {
    // Load the face detection model
    std::string jsonContent = readModelFile("../models/human-face.json");
    // std::vector<int> jsonArray = intToStringArray(jsonContent);
    CascadeClassifier* face_cascade = wasmface::create(jsonContent.c_str());
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
    cv::namedWindow("Gray video", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame;
        // Read a new frame from the video camera
        bool ret = webcam.read(frame);
        if (!ret) {
            std::cerr << "Failed to read from camera. Exiting!" << std::endl;
            break;
        }

        // Convert to grayscale
        cv::Mat gray;
        int height = gray.rows;
        int width = gray.cols;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::imshow("Gray video", gray);

        // Face detection
        std::vector<cv::Rect> faces;
        faces = wasmface::detect(frame, width, height, face_cascade, 2.0, 2.0, true, 0.3, 5);

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

    // Release the video capture object
    webcam.release();
    // Destroy all windows
    cv::destroyAllWindows();

    return 0;
}

// #include <opencv2/opencv.hpp>

// // g++ -std=c++11 -o face_detection face_detection.cpp `pkg-config --cflags --libs opencv4`

// int main() {
//     // Load the face detection model
//     cv::CascadeClassifier face_cascade;
//     if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
//         std::cerr << "Error loading cascade file. Exiting!" << std::endl;
//         return -1;
//     }

//     // Open the default video camera
//     cv::VideoCapture webcam(0);
//     if (!webcam.isOpened()) {
//         std::cerr << "Error opening video camera. Exiting!" << std::endl;
//         return -1;
//     }

//     cv::namedWindow("web cam", cv::WINDOW_AUTOSIZE);
//     cv::namedWindow("Gray video", cv::WINDOW_AUTOSIZE);

//     while (true) {
//         cv::Mat frame;
//         // Read a new frame from the video camera
//         bool ret = webcam.read(frame);
//         if (!ret) {
//             std::cerr << "Failed to read from camera. Exiting!" << std::endl;
//             break;
//         }

//         // Convert to grayscale
//         cv::Mat gray;
//         cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//         cv::imshow("Gray video", gray);

//         // Face detection
//         std::vector<cv::Rect> faces;
//         face_cascade.detectMultiScale(gray, faces, 1.1, 5);

//         // Draw rectangles around detected faces
//         for (const auto& rect : faces) {
//             cv::rectangle(frame, rect, cv::Scalar(0, 255, 240), 2);
//         }

//         // Display the resulting frame
//         cv::imshow("web cam", frame);

//         // Break loop on 'd' key press
//         if (cv::waitKey(1) == 'd') {
//             break;
//         }
//     }

//     // Release the video capture object
//     webcam.release();
//     // Destroy all windows
//     cv::destroyAllWindows();

//     return 0;
// }