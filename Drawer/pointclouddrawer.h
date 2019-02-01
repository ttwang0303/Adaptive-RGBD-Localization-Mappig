#ifndef POINTCLOUDDRAWER_H
#define POINTCLOUDDRAWER_H

#include <mutex>
#include <opencv2/core.hpp>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudDrawer {
public:
    PointCloudDrawer();

    void DrawPointCloud();

    void UpdateSourceCloud(pcl::PointCloud<pcl::PointXYZ>& pSource);

    void UpdateTargetCloud(pcl::PointCloud<pcl::PointXYZ>& pTarget);

    void UpdateMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pMapCloud, const cv::Mat& pose);

private:
    float mPointSize;
    float mLineWidth;

    std::mutex mMutexPointCloud;
    std::mutex mMutexPose;

    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mpMapCloud;
    std::vector<cv::Mat> mvPoses;
    float r, g, b;
};

#endif // POINTCLOUDDRAWER_H
