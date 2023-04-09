#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

using namespace std;
using namespace cv;

#define points_per_row 9
// 每行的内点数
#define points_per_col 6
// 每列的内点数

void readImage(const std::string &path, std::vector<cv::Mat> &image_buff);

void stereoCalib(const vector<cv::Mat> &image_left_list,
                 const vector<cv::Mat> &image_right_list,
                 const Size &board_size);
