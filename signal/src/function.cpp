#include "../include/function.h"
#include <glog/logging.h>

void readImage(const std::string &path, std::vector<cv::Mat> &image_buff) {
    std::vector<cv::String> result;
    // 读取文件中的路径名
    cv::glob(path, result, false);
    for (size_t i = 0; i < result.size(); i++) {
        cv::Mat img = cv::imread(result[i], -1);
        image_buff.push_back(img);
    }
    LOG(INFO) << "read finsh.";
    return;
}

void grabCorner(cv::Mat &image, std::vector<cv::Point2f> &points_per_image) {
    static std::string path = "/home/melody/slam/camera_model/signal/result/";
    static int image_num = 0;
    image_num++;
    cv::Size image_size;
    if (image_num == 1) {
        image_size.width = image.cols;
        image_size.height = image.rows;
        LOG(INFO) << "image size "
                  << "width " << image_size.width << "  "
                  << "height" << image_size.height;
    }

    cv::cvtColor(image, image, CV_BGR2GRAY);

    bool success = cv::findChessboardCorners(
        image, cv::Size(points_per_row, points_per_col), points_per_image);
    if (!success) {
        LOG(ERROR) << "grab default.";
    } else {
        cv::find4QuadCornerSubpix(image, points_per_image, cv::Size(5, 5));

        cv::drawChessboardCorners(image,
                                  cv::Size(points_per_row, points_per_col),
                                  points_per_image, success);

        cv::imwrite(path + std::to_string(image_num) + ".png", image);
        // cv::waitKey(0);
    }
}
