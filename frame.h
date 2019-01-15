#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Frame {
public:
    Frame();

    Frame(cv::Mat& imColor, cv::Mat& imDepth, double timestamp);

    void SetPose(cv::Mat& Tcw);

    void DetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor);

    void CreateCloud();

public:
    double timestamp;
    cv::Mat im, depth;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    std::vector<cv::Point3f> kps3Dc;

    // Camera pose.
    cv::Mat mTcw;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc (Camera center)
};

#endif // FRAME_H
