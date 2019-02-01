#ifndef POINTCLOUDDRAWER_H
#define POINTCLOUDDRAWER_H

#include <mutex>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudDrawer {
public:
    PointCloudDrawer();

    void DrawPointCloud();

    void UpdateSourceCloud(pcl::PointCloud<pcl::PointXYZ>& pSource);

    void UpdateTargetCloud(pcl::PointCloud<pcl::PointXYZ>& pTarget);

private:
    float mPointSize;
    float mLineWidth;

    std::mutex mMutexPointCloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
};

#endif // POINTCLOUDDRAWER_H
