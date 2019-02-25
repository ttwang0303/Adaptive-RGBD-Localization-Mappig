#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include <mutex>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <set>

class Landmark;
class Map;
class KeyFrame;

class MapDrawer {
public:
    MapDrawer(Map* pMap);

    size_t DrawLandmarks();
    // void DrawDensePoints();
    // void DrawOctoMap();
    size_t DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    // void DrawCurrentCamera(pangolin::OpenGlMatrix& Twc);
    // void SetCurrentCameraPose(const cv::Mat& Tcw);
    // void SetReferenceKeyFrame(KeyFrame* pKF);
    // void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);

private:
    Map* mpMap;

    float mPointSize;
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;

    std::mutex mMutexPointCloud;
    std::mutex mMutexPose;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mpMapCloud;
    std::set<int> mKFids;
};

#endif
