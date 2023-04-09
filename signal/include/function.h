#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#define points_per_row 11
// 每行的内点数
#define points_per_col 8
// 每列的内点数

void readImage(const std::string &path, std::vector<cv::Mat> &image_buff);

void grabCorner(cv::Mat &image, std::vector<cv::Point2f> &points_per_image);
