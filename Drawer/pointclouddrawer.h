#ifndef POINTCLOUDDRAWER_H
#define POINTCLOUDDRAWER_H

#include <mutex>
#include <opencv2/core.hpp>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <set>

class Landmark;
class Map;
class Frame;

class PointCloudDrawer {
public:
    PointCloudDrawer(Map* pMap);

    void DrawPointCloud(bool drawLandmarks = true, bool drawDenseCloud = true, bool drawKFs = true);

private:
    void DrawLandmarks();
    void DrawDenseCloud(Frame* pKF);
    void DrawPoseKF(Frame* pKF);
    void DrawConnections(std::vector<Frame*>& vpOrderedKFs);

    Map* mpMap;

    float mPointSize;
    float mLineWidth;
    float mLineWidthGraph;

    std::mutex mMutexPointCloud;
    std::mutex mMutexPose;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mpMapCloud;
    std::set<int> mKFids;
};

#endif // POINTCLOUDDRAWER_H
