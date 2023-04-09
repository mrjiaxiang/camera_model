#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>

#include "../include/function.h"

using namespace std;
using namespace cv;

const int image_width = 640;
const int image_height = 480;

const int board_width = 9;
const int board_height = 6;
const int board_corner = board_width * board_height;
const int frame_number = 13;
const int square_size = 25; // 标定板黑白格子的大小 单位mm
const Size board_size = Size(board_width, board_height);
Size image_size = Size(image_width, image_height);

Mat R, T, E, F;    // R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<Mat> rvecs; // 旋转向量
vector<Mat> tvecs; // 平移向量
vector<vector<Point2f>> imagePointL; // 左边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR; // 右边摄像机所有照片角点的坐标集合
vector<vector<Point3f>> objRealPoint; // 各副图像的角点的实际物理坐标集合

vector<Point2f> cornerL; // 左边摄像机某一照片角点坐标集合

vector<Point2f> cornerR; // 右边摄像机某一照片角点坐标集合
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat Rl, Rr, Pl, Pr,
    Q; // 校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）
Mat mapLx, mapLy, mapRx, mapRy; // 映射表
Rect validROIL,
    validROIR; // 图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域

Mat cameraMatrixL = (Mat_<double>(3, 3) << 462.279595, 0, 312.781587, 0,
                     460.220741, 208.225803, 0, 0, 1);
Mat distCoeffL =
    (Mat_<double>(5, 1) << -0.054929, 0.224509, 0.000386, 0.001799, -0.302288);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 463.923124, 0, 322.783959, 0,
                     462.203276, 256.100655, 0, 0, 1);
Mat distCoeffR =
    (Mat_<double>(5, 1) << -0.049056, 0.229945, 0.001745, -0.001862, -0.321533);

void calRealPoint(vector<vector<Point3f>> &obj, int boardwidth, int boardheight,
                  int imgNumber, int squaresize) {
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardheight; rowIndex++) {
        for (int colIndex = 0; colIndex < boardwidth; colIndex++) {
            imgpoint.push_back(
                Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++) {
        obj.push_back(imgpoint);
    }
}

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/home/melody/slam/camera_model/stereo/Log";
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

    int goodFrameCount = 0;

    while (goodFrameCount < frame_number) {
        cvtColor(image_left[goodFrameCount], grayImageL, CV_BGR2GRAY);
        cvtColor(image_right[goodFrameCount], grayImageR, CV_BGR2GRAY);

        bool isFindL = false, isFindR = false;
        isFindL = findChessboardCorners(grayImageL, board_size, cornerL);
        isFindR = findChessboardCorners(grayImageR, board_size, cornerR);

        if (isFindL && isFindR) {
            cornerSubPix(
                grayImageL, cornerL, Size(5, 5), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(image_left[goodFrameCount], board_size,
                                  cornerL, isFindL);
            imagePointL.push_back(cornerL);

            cornerSubPix(
                grayImageR, cornerR, Size(5, 5), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(image_right[goodFrameCount], board_size,
                                  cornerR, isFindR);
            imagePointR.push_back(cornerR);
            goodFrameCount++;
        }
    }
    calRealPoint(objRealPoint, board_width, board_height, frame_number,
                 square_size);

    double rms = stereoCalibrate(
        objRealPoint, imagePointL, imagePointR, cameraMatrixL, distCoeffL,
        cameraMatrixR, distCoeffR, Size(image_width, image_height), R, T, E, F,
        CALIB_USE_INTRINSIC_GUESS,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    LOG(INFO) << "stereo calibration done with rms error = " << rms;

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
                  image_size, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1,
                  image_size, &validROIL, &validROIR);

    LOG(INFO) << "start rectify";
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, image_size,
                            CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, image_size,
                            CV_32FC1, mapRx, mapRy);

    Mat rectifyImageL, rectifyImageR;
    cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
    cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
    LOG(INFO) << "after remap 图像已经共面且已经对准";

    Mat rectifyImageL2, rectifyImageR2;
    remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
    remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);

    

    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / max(image_size.width, image_size.height);
    w = cvRound(image_size.width * sf);
    h = cvRound(image_size.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    /*左图像画到画布上*/
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h)); // 得到画布的一部分
    resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0,
           INTER_AREA); // 把图像缩放到跟canvasPart一样大小
    Rect vroiL(cvRound(validROIL.x * sf),
               cvRound(validROIL.y * sf), // 获得被截取的区域
               cvRound(validROIL.width * sf), cvRound(validROIL.height * sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8); // 画上一个矩形

    cout << "Painted ImageL" << endl;

    /*右图像画到画布上*/
    canvasPart = canvas(Rect(w, 0, w, h)); // 获得画布的另一部分
    resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
               cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

    cout << "Painted ImageR" << endl;

    /*画上对应的线条*/
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1,
             8);
    imwrite("/home/melody/slam/camera_model/stereo/result/image.png", canvas);
    return 0;
}