#include <opencv2/opencv.hpp>

// g++ -std=c++11 -o face_detection face_detection.cpp `pkg-config --cflags --libs opencv4`

int main() {
    // Load the face detection model
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading cascade file. Exiting!" << std::endl;
        return -1;
    }

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
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::imshow("Gray video", gray);

        // Face detection
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5);

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
