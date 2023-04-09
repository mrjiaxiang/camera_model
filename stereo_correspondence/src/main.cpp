#include "function.h"
#include "stereo_match.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/home/melody/slam/camera_model/stereo_correspondence/Log";
    FLAGS_alsologtostderr = 1;

    string left_dir = "/home/melody/slam/camera_model/stereo/data/left";
    string right_dir = "/home/melody/slam/camera_model/stereo/data/right";

    std::vector<cv::Mat> image_left;
    std::vector<cv::Mat> image_right;

    readImage(left_dir, image_left);
    readImage(right_dir, image_right);
    if (image_left.empty() || image_right.empty()) {
        LOG(ERROR) << "image buff empty.";
        return -1;
    }
    SAD sad_test(7, 30);
    for (int i = 0; i < image_left.size(); i++) {
        cv::Mat image = sad_test.computerSAD(image_left[i], image_right[i]);
    }

    return 0;
}
