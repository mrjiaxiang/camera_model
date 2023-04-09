#include "stereo_match.h"
#include <glog/logging.h>

Mat SAD::computerSAD(Mat &L, Mat &R) {
    int Height = L.rows;
    int Width = L.cols;
    Mat Kernel_L(Size(winSize, winSize), CV_8U, Scalar::all(0));
    Mat Kernel_R(Size(winSize, winSize), CV_8U, Scalar::all(0));
    Mat Disparity(Height, Width, CV_8U, Scalar(0)); // 视差图

    for (int i = 0; i < Width - winSize; i++) // 左图从DSR开始遍历
    {
        for (int j = 0; j < Height - winSize; j++) {
            Kernel_L = L(Rect(i, j, winSize, winSize));
            Mat MM(1, DSR, CV_32F, Scalar(0)); //

            for (int k = 0; k < DSR; k++) {
                int x = i - k;
                if (x >= 0) {
                    Kernel_R = R(Rect(x, j, winSize, winSize));
                    Mat Dif;
                    absdiff(Kernel_L, Kernel_R, Dif); //
                    Scalar ADD = sum(Dif);
                    float a = ADD[0];
                    MM.at<float>(k) = a;
                }
            }
            Point minLoc;
            minMaxLoc(MM, NULL, NULL, &minLoc, NULL);

            int loc = minLoc.x;
            // int loc=DSR-loc;
            Disparity.at<char>(j, i) = loc * 16;
        }
        double rate = double(i) / (Width);
        // LOG(INFO) << "已完成" << std::setprecision(2) << rate * 100 << "%"
        //           << endl; // 处理进度
    }
    return Disparity;
}

Mat BM(cv::Mat &left, cv::Mat &right) {
    int SADWindowSize = 11;
    int numberOfDisparities = 64;
    int uniquenessRation = 0;
    // Rect validROIL, validROIR;

    cv::Ptr<cv::StereoBM> bm =
        cv::StereoBM::create(numberOfDisparities, SADWindowSize);
    // bm->setROI1(validROIL);
    // bm->setROI2(validROIR);
    bm->setPreFilterCap(15);
    bm->setMinDisparity(0);
    // 最小视差，默认值为0, 可以是负值，int型
    bm->setNumDisparities(numberOfDisparities * 16 + 16);
    // 视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(uniquenessRation);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);

    Mat disp, disp8;
    bm->compute(left, right, disp);
    disp.convertTo(disp8, CV_8U, 255 / ((numberOfDisparities * 16 + 16) * 16.));

    return disp8;
    IplImage *img1;
    IplImage *img2;
    // *img1 = mat2IplImage("left.png", 0);
    // *img2 = mat2IplImage("right.png", 0);
    // CvStereoBMState *BMState = cvCreateStereoBMState();
    // assert(BMState);
    // BMState->preFilterSize = 9;
    // BMState->preFilterCap = 31;
    // BMState->SADWindowSize = 15;
    // BMState->minDisparity = 0;
    // BMState->numberOfDisparities = 64;
    // BMState->textureThreshold = 10;
    // BMState->uniquenessRatio = 15;
    // BMState->speckleWindowSize = 100;
    // BMState->speckleRange = 32;
    // BMState->disp12MaxDiff = 1;

    // CvMat *disp = cvCreateMat(img1->height, img1->width, CV_16S);
    // CvMat *vdisp = cvCreateMat(img1->height, img1->width, CV_8U);
    // int64 t = getTickCount();
    // cvFindStereoCorrespondenceBM(img1, img2, disp, BMState);
    // t = getTickCount() - t;
    // cout << "Time elapsed:" << t * 1000 / getTickFrequency() << endl;
    // cvSave("disp.xml", disp);
    // cvNormalize(disp, vdisp, 0, 255, CV_MINMAX);
}

cv::Mat SGBM(cv::Mat &left, cv::Mat &right) {
    // cv::StereoSGBM sgbm;
    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;
    int numberOfDisparities = 64;
    cv::Ptr<cv::StereoSGBM> sgbm =
        cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
    int P1 = 8 * left.channels() * SADWindowSize * SADWindowSize;
    int P2 = 32 * left.channels() * SADWindowSize * SADWindowSize;
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setPreFilterCap(15);
    sgbm->setUniquenessRatio(6);
    sgbm->setSpeckleRange(2);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setDisp12MaxDiff(1);
    // sgbm->setNumDisparities(1);
    sgbm->setMode(cv::StereoSGBM::MODE_HH);
    Mat disp, disp8U;
    sgbm->compute(left, right, disp);
    disp.convertTo(disp, CV_32F, 1.0 / 16);
    disp8U = Mat(disp.rows, disp.cols, CV_8UC1); // 显示
    // normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

    disp.convertTo(disp8U, CV_8U, 255 / (numberOfDisparities * 16.));
    // reprojectImageTo3D(disp, xyz, Q);
    // xyz = xyz * 16;
    return disp8U;
}

cv::Mat GC(cv::Mat &left, cv::Mat &right) {
    IplImage *img_left, *img_right;
    *img_left = mat2IplImage(left);
    *img_right = mat2IplImage(right);
    // 这和代码在opencv找不到？？？
    CvStereoGCState *GCState = cvCreateStereoGCState(64, 3);
    assert(GCState);
    cout << "start matching using GC" << endl;
    CvMat *gcdispleft = cvCreateMat(img_left->height, img_left->width, CV_16S);
    CvMat *gcdispright =
        cvCreateMat(img_right->height, img_right->width, CV_16S);
    CvMat *gcvdisp = cvCreateMat(img_left->height, img_left->width, CV_8U);
    int64 t = getTickCount();
    // 这和代码在opencv找不到？？？
    cvFindStereoCorrespondenceGC(img_left, img_right, gcdispleft, gcdispright,
                                 GCState);
    t = getTickCount() - t;
    cout << "Time elapsed:" << t * 1000 / getTickFrequency() << endl;
    // cvNormalize(gcdispleft,gcvdisp,0,255,CV_MINMAX);
    // cvSaveImage("GC_left_disparity.png",gcvdisp);
    cvNormalize(gcdispright, gcvdisp, 0, 255, CV_MINMAX);

    return cv::cvarrToMat(gcvdisp).setTo(true);
}
