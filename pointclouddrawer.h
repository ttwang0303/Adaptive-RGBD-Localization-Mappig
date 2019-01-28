#ifndef POINTCLOUDDRAWER_H
#define POINTCLOUDDRAWER_H

#include <mutex>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudDrawer {
public:
    PointCloudDrawer();

    void DrawPointCloud();

    void AssignSourceCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pSource);

    void AssignTargetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pTarget);

private:
    float mPointSize;
    float mLineWidth;

    std::mutex mMutexPointCloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
};

#endif // POINTCLOUDDRAWER_H
