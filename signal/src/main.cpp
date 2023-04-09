#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>

#include "../include/function.h"

using namespace std;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/home/melody/slam/camera_model/signal/Log";
    FLAGS_alsologtostderr = 1;

    string dir = "/home/melody/slam/camera_model/signal/data";
    std::vector<cv::Mat> image;

    readImage(dir, image);
    if (image.empty()) {
        LOG(ERROR) << "image buff empty.";
        return -1;
    }

    vector<vector<cv::Point2f>> points_all_images;

    for (int i = 0; i < image.size(); i++) {
        vector<cv::Point2f> points_per_image;
        grabCorner(image[i], points_per_image);

        points_all_images.push_back(points_per_image);
    }
    cv::destroyAllWindows();

    cv::Size block_size(21,
                        21); // 每个小方格实际大小, 只会影响最后求解的平移向量t
    cv::Mat camera_K(3, 3, CV_32FC1, cv::Scalar::all(0));   // 内参矩阵3*3
    cv::Mat distCoeffs(1, 5, CV_32FC1, cv::Scalar::all(0)); // 畸变矩阵1*5
    vector<cv::Mat> rotationMat;                            // 旋转矩阵
    vector<cv::Mat> translationMat;                         // 平移矩阵
    // 初始化角点三维坐标,从左到右,从上到下!!!
    vector<cv::Point3f> points3D_per_image;
    for (int i = 0; i < cv::Size(points_per_row, points_per_col).height; i++) {
        for (int j = 0; j < cv::Size(points_per_row, points_per_col).width;
             j++) {
            points3D_per_image.push_back(
                cv::Point3f(block_size.width * j, block_size.height * i, 0));
        }
    }

    vector<vector<cv::Point3f>> points3D_all_images(
        image.size(), points3D_per_image); // 保存所有图像角点的三维坐标, z=0

    int point_counts =
        cv::Size(points_per_row, points_per_col).area(); // 每张图片上角点个数
    //! 标定
    /**
     * points3D_all_images: 真实三维坐标
     * points_all_images: 提取的角点
     * image_size: 图像尺寸
     * camera_K : 内参矩阵K
     * distCoeffs: 畸变参数
     * rotationMat: 每个图片的旋转向量
     * translationMat: 每个图片的平移向量
     * */
    // step4 标定

    cv::Size image_size;
    image_size.width = image[0].cols;
    image_size.height = image[0].rows;
    cv::calibrateCamera(points3D_all_images, points_all_images, image_size,
                        camera_K, distCoeffs, rotationMat, translationMat, 0);

    std::string undistort_path =
        "/home/melody/slam/camera_model/signal/undistort/";

    // undistort
    for (int i = 0; i < image.size(); i++) {
        cv::Mat result;
        cv::undistort(image[i], result, camera_K, distCoeffs);
        cv::imwrite(undistort_path + std::to_string(i) + ".png", result);
    }

    double total_err = 0.0;               // 所有图像平均误差总和
    double err = 0.0;                     // 每幅图像的平均误差
    vector<cv::Point2f> points_reproject; // 重投影点
    cout << "\n\t每幅图像的标定误差:\n";
    LOG(INFO) << "每幅图像的标定误差：\n";
    for (int i = 0; i < image.size(); i++) {
        vector<cv::Point3f> points3D_per_image = points3D_all_images[i];
        // 通过之前标定得到的相机内外参，对三维点进行重投影
        cv::projectPoints(points3D_per_image, rotationMat[i], translationMat[i],
                          camera_K, distCoeffs, points_reproject);
        // 计算两者之间的误差
        vector<cv::Point2f> detect_points =
            points_all_images[i]; // 提取到的图像角点
        cv::Mat detect_points_Mat =
            cv::Mat(1, detect_points.size(),
                    CV_32FC2); // 变为1*70的矩阵,2通道保存提取角点的像素坐标
        cv::Mat points_reproject_Mat =
            cv::Mat(1, points_reproject.size(),
                    CV_32FC2); // 2通道保存投影角点的像素坐标
        for (int j = 0; j < detect_points.size(); j++) {
            detect_points_Mat.at<cv::Vec2f>(0, j) =
                cv::Vec2f(detect_points[j].x, detect_points[j].y);
            points_reproject_Mat.at<cv::Vec2f>(0, j) =
                cv::Vec2f(points_reproject[j].x, points_reproject[j].y);
        }
        err = cv::norm(points_reproject_Mat, detect_points_Mat,
                       cv::NormTypes::NORM_L2);
        total_err += err /= point_counts;
        cout << "第" << i + 1 << "幅图像的平均误差为： " << err << "像素"
             << endl;
        LOG(INFO) << "第" << i + 1 << "幅图像的平均误差为： " << err << "像素"
                  << endl;
    }
    cout << "总体平均误差为： " << total_err / image.size() << "像素" << endl;
    LOG(INFO) << "总体平均误差为： " << total_err / image.size() << "像素"
              << endl;
    cout << "评价完成！" << endl;

    // 将标定结果写入txt文件
    cv::Mat rotate_Mat =
        cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); // 保存旋转矩阵
    cout << "\n相机内参数矩阵:" << endl;
    cout << camera_K << endl << endl;
    LOG(INFO) << "\n相机内参数矩阵:" << endl;
    LOG(INFO) << camera_K << endl << endl;
    cout << "畸变系数：\n";
    cout << distCoeffs << endl << endl << endl;
    LOG(INFO) << "畸变系数：\n";
    LOG(INFO) << distCoeffs << endl << endl << endl;
    for (int i = 0; i < image.size(); i++) {
        cv::Rodrigues(rotationMat[i],
                      rotate_Mat); // 将旋转向量通过罗德里格斯公式转换为旋转矩阵
        LOG(INFO) << "第" << i + 1 << "幅图像的旋转矩阵为：" << endl;
        LOG(INFO) << rotate_Mat << endl;
        LOG(INFO) << "第" << i + 1 << "幅图像的平移向量为：" << endl;
        LOG(INFO) << translationMat[i] << endl << endl;
    }

    return 0;
}