#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Landmark;

class Frame {
public:
    Frame(cv::Mat& imColor, cv::Mat& imDepth, double mTimestamp);

    Frame(cv::Mat& imColor);

    void SetPose(cv::Mat& Tcw);

    void DetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor);

    void GridDetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor, int gridRows, int gridCols);

    void Detect(cv::Ptr<cv::FeatureDetector> pDetector);

    void Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor);

    void CreateCloud();
    void VoxelGridFilterCloud(float resolution);
    void StatisticalOutlierRemovalFilterCloud(int meanK, double stddev);

    void AddLandmark(Landmark* pLandmark, const size_t& idx);

    // Iaicp test
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Mat2Cloud();

public:
    cv::Mat mIm, mDepth;
    double mTimestamp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mpCloud;
    std::vector<cv::KeyPoint> mvKps;
    cv::Mat mDescriptors;
    std::vector<cv::Point3f> mvKps3Dc;

    // Landmark to associated keypoint
    std::vector<Landmark*> mvpLandmarks;

    // Number of keypoints
    int N;

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc (Camera center)
};

#endif // FRAME_H
