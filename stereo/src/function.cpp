#include "../include/function.h"
#include <glog/logging.h>

void readImage(const std::string &path, std::vector<cv::Mat> &image_buff) {
    std::vector<cv::String> result;
    // 读取文件中的路径名
    cv::glob(path, result, false);
    for (size_t i = 0; i < result.size(); i++) {
        cv::Mat img = cv::imread(result[i]);
        image_buff.push_back(img);
    }
    LOG(INFO) << "read finsh.";
    return;
}

void stereoCalib(const vector<cv::Mat> &image_left_list,
                 const vector<cv::Mat> &image_right_list,
                 const Size &board_size) {

    vector<vector<Point2f>> image_points[2];
    vector<vector<Point3f>> object_points;

    Size image_size;
    int n_images = image_left_list.size();
    image_points[0].resize(n_images);
    image_points[1].resize(n_images);
    object_points.resize(n_images);
    LOG(INFO) << "start";
    for (size_t i = 0; i < n_images; i++) {
        bool found_left = false, found_right = false;
        vector<Point2f> corners_left;  //= image_points[0][i];
        vector<Point2f> corners_right; //= image_points[1][i];

        cv::Mat left_image = image_left_list[i];
        cv::Mat right_image = image_right_list[i];

        cv::cvtColor(left_image, left_image, CV_BGR2GRAY);
        cv::cvtColor(right_image, right_image, CV_BGR2GRAY);

        LOG(INFO) << "findChessboardCorners.";
        found_left = findChessboardCorners(
            image_left_list[i], board_size, corners_left,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        found_right = findChessboardCorners(
            image_right_list[i], board_size, corners_right,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found_left == false || found_right == false) {
            LOG(ERROR) << "FALSE";
            break;
        } else {
            LOG(INFO) << "cornerSubPix.";
            find4QuadCornerSubpix(left_image, corners_left, Size(11, 11));
            find4QuadCornerSubpix(right_image, corners_right, Size(11, 11));
            image_points[0].push_back(corners_left);
            image_points[1].push_back(corners_right);
            LOG(INFO) << "drawChessboardCorners.";
            drawChessboardCorners(left_image, board_size, corners_left,
                                  found_left);
            drawChessboardCorners(right_image, board_size, corners_right,
                                  found_right);
            cv::imwrite("/home/melody/slam/camera_model/stereo/result/left/" +
                            std::to_string(i) + ".png",
                        left_image);
            cv::imwrite("/home/melody/slam/camera_model/stereo/result/right/" +
                            std::to_string(i) + ".png",
                        right_image);
        }
    }
    Size block_size(21, 21);
    for (size_t i = 0; i < n_images; i++) {
        for (size_t j = 0; j < block_size.height; j++)
            for (size_t k = 0; k < block_size.width; k++)
                object_points[i].push_back(
                    Point3f(k * block_size.width, j * block_size.height, 0));
    }

    image_size.width = image_left_list[0].cols;
    image_size.height = image_left_list[0].rows;

    Mat cameraMatrix[2], distCoeffs[2];
    LOG(INFO) << "initCameraMatrix2D.";
    cameraMatrix[0] =
        initCameraMatrix2D(object_points, image_points[0], image_size, 0);
    cameraMatrix[1] =
        initCameraMatrix2D(object_points, image_points[1], image_size, 0);

    Mat R, T, E, F;
    LOG(INFO) << "stereoCalibrate.";
    double rms = stereoCalibrate(
        object_points, image_points[0], image_points[1], cameraMatrix[0],
        distCoeffs[0], cameraMatrix[1], distCoeffs[1], image_size, R, T, E, F,
        CALIB_FIX_ASPECT_RATIO + CALIB_ZERO_TANGENT_DIST +
            CALIB_USE_INTRINSIC_GUESS + CALIB_SAME_FOCAL_LENGTH +
            CALIB_RATIONAL_MODEL + CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    LOG(INFO) << "done with RMS error=" << rms << endl;
}