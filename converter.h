#ifndef CONVERTER_H
#define CONVERTER_H

#include <Eigen/Dense>
#include <opencv2/core.hpp>

class Converter {
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& Descriptors);

    static cv::Mat toHomogeneous(const cv::Mat& r, const cv::Mat& t);

    static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4>& m);
    static cv::Mat toCvMat(const Eigen::Matrix<float, 4, 4>& m);
    static cv::Mat toCvMat(const Eigen::Matrix3d& m);
    static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1>& m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3>& R,
        const Eigen::Matrix<double, 3, 1>& t);

    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat& cvVector);
    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f& cvPoint);
    static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat& cvMat3);
    static Eigen::Matrix<float, 4, 4> toMatrix4f(const cv::Mat& cvMat4);

    static std::vector<float> toQuaternion(const cv::Mat& M);
};
#endif // CONVERTER_H