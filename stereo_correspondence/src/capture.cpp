#include <iostream>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    VideoCapture cap(0);
    VideoCapture cap1(1);

    if (!cap.isOpened()) {
        LOG(ERROR) << "cloud not open camera0";
        return -1;
    }
    if (!cap1.isOpened()) {
        LOG(ERROR) << "cloud not open camera1";
        return -2;
    }

    Mat frame, frame1;
    bool stop = false;
    while (!stop) {
        static int count = 0;
        cap.read(frame);
        cap1.read(frame1);

        int delay = 30;
        if (delay >= 0 && waitKey(delay) > 0) {
            waitKey(0);
        }
        count++;
        imwrite("../capture/left/left" + to_string(count) + ".png", frame);
        imwrite("../capture/right/right" + to_string(count) + ".png", frame1);
    }
    cap.release();
    cap1.release();
    return 0;
}