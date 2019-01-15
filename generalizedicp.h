#ifndef GENERALIZEDICP_H
#define GENERALIZEDICP_H

#include <opencv2/opencv.hpp>
#include <pcl/registration/gicp.h>
#include <vector>

class Frame;

class GeneralizedICP {
public:
    GeneralizedICP();

    GeneralizedICP(size_t iters, double maxCorrespondenceDist);

    ~GeneralizedICP() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    bool Compute(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12, Eigen::Matrix4f& guess);

    bool ComputeSubset(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    bool Align(const Eigen::Matrix4f& guess);

    Eigen::Matrix4f GetTransformation() const;

    double GetScore() const;

private:
    void CreateCloudsFromMatches(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    std::vector<cv::DMatch> GetSubset(const std::vector<cv::DMatch>& vMatches12);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> mGicp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mSrcCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTgtCloud;

    size_t mIterations;
    double mMaxCorrespondenceDistance;
    Eigen::Matrix4f mTransformation;
    double mScore;
};

#endif // GENERALIZEDICP_H
