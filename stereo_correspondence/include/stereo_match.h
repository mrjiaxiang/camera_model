#pragma once
// https://blog.csdn.net/jin739738709/article/details/124145268
// https://blog.csdn.net/liulina603/article/details/53302168/

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class SAD {
  public:
    SAD() : winSize(7), DSR(30) {}
    SAD(int _winSize, int _DSR) : winSize(_winSize), DSR(_DSR) {}
    Mat computerSAD(Mat &L, Mat &R); // 计算SAD
  private:
    int winSize; // 卷积核的尺寸
    int DSR;     // 视差搜索范围
};

IplImage mat2IplImage(Mat image) {
    IplImage *ipl_img = cvCreateImage(cvSize(image.cols, image.rows),
                                      IPL_DEPTH_8U, image.channels());
    memcpy(ipl_img->imageData, image.data,
           ipl_img->height * ipl_img->widthStep);
    return *ipl_img;
}

// http://blog.csdn.net/wqvbjhc/article/details/6260844
Mat BM(cv::Mat &left, cv::Mat &right);

// http://www.opencv.org.cn/forum.php?mod=viewthread&tid=23854
cv::Mat SGBM(cv::Mat &left, cv::Mat &right);

//
cv::Mat GC(cv::Mat &left, cv::Mat &right);