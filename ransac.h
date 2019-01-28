#ifndef RANSAC_H
#define RANSAC_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class Frame;

class Ransac {
public:
    Ransac();

    Ransac(int iters, uint minInlierTh, float maxMahalanobisDist, uint sampleSize);

    ~Ransac() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    bool Iterate(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& m12);

    // Return mean residual error
    float TransformSourcePointCloud();

    void SetIterations(int iters);
    void SetMaxMahalanobisDistance(float dist);
    void SetSampleSize(uint sampleSize);
    void SetInlierThreshold(uint th);
    void CheckDepth(bool check);

private:
    std::vector<cv::DMatch> SampleMatches(const std::vector<cv::DMatch>& vMatches);

    Eigen::Matrix4f GetTransformFromMatches(const std::vector<cv::DMatch>& vMatches, bool& valid);

    double ComputeInliersAndError(const std::vector<cv::DMatch>& m12, const Eigen::Matrix4f& transformation4f,
        std::vector<cv::DMatch>& vInlierMatches);

    double ErrorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation);

    double DepthCovariance(double depth);

    double DepthStdDev(double depth);

    // Ransac parameters
    int mIterations;
    uint mMinInlierTh;
    float mMaxMahalanobisDistance;
    uint mSampleSize;

    bool mCheckDepth;

    Frame* mpSourceFrame;
    Frame* mpTargetFrame;

public:
    float rmse;
    std::vector<cv::DMatch> mvInliers;
    Eigen::Matrix4f mT12;

    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTransformedCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceInlierCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetInlierCloud;
};

#endif // RANSAC_H
