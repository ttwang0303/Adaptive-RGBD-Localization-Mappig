#ifndef UTILS_H
#define UTILS_H

#include "featureadjuster.h"
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

class Frame;

void LoadImages(const std::string& associationFilename, std::vector<std::string>& vImageFilenamesRGB,
    std::vector<std::string>& vImageFilenamesD, std::vector<double>& vTimestamps);

std::vector<cv::DMatch> Match(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher);

void DrawMatches(Frame* pF1, Frame* pF2, const std::vector<cv::DMatch>& m12, const int delay = 1);

void DrawKeyPoints(Frame* pF1, const int delay = 1);

cv::Ptr<cv::DescriptorExtractor> CreateDetector(const std::string& detector);

cv::Ptr<cv::FeatureDetector> CreateDescriptor(const std::string& descriptor);

bool FindHomography(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12, cv::Mat& H, std::vector<uchar>& vRansacStatus);
cv::Point2f ApplyHomography(const cv::Point2f& pt, const cv::Mat& H);
cv::Mat DistanceFiler(const Frame* pF1, const Frame* pF2, std::vector<cv::DMatch>& vMatches12, const double thresh = 2.0);

std::vector<std::pair<double, double>> TestRecallPrecision(Frame* pF1, Frame* pF2, cv::Ptr<cv::DescriptorMatcher> pMatcher, std::vector<cv::DMatch>& vMatches12);

// ---------- Cloud Processing ----------
void AddNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud, int k = 15);

template <typename PointT>
void VoxelGridFilterCloud(pcl::PointCloud<PointT>& cloud, float resolution)
{
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.setInputCloud(boost::make_shared<pcl::PointCloud<PointT>>(cloud));
    voxel.filter(cloud);
}

// New Features functions
StatefulFeatureDetector* AdjusterWrapper(cv::Ptr<DetectorAdjuster> detadj, int min, int max);
StatefulFeatureDetector* AdjustedGridWrapper(cv::Ptr<DetectorAdjuster> detadj);
cv::Feature2D* CreateDetector2(const std::string& detectorName);
cv::Ptr<cv::DescriptorExtractor> CreateDescriptor2(const std::string& descriptorName);

#endif // UTILS_H
