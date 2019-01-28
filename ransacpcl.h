#ifndef RANSACPCL_H
#define RANSACPCL_H

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <vector>

class Frame;

class RansacPCL {
public:
    RansacPCL();

    void Iterate(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    void CreateCloudFromMatches(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    void GetInliersDMatch(std::vector<cv::DMatch>& vInliers);

    float ResidualError();

    void SetInlierThreshold(double thresh);

    void SetMaximumIterations(int iters);

public:
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>::Ptr mpSampleConsensus;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTransformedCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceInlierCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetInlierCloud;
    pcl::Correspondences mCorrespondences;
    std::vector<cv::DMatch> mvDMatches;
    pcl::Correspondences mInliersCorrespondences;
    Eigen::Matrix4f mT12;
};

#endif // RANSACPCL_H
