#ifndef FRAME_H
#define FRAME_H

#include "DBoW3/DBoW3.h"
#include <mutex>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Landmark;
class Extractor;
class Map;

class Frame {
public:
    typedef pcl::PointCloud<pcl::PointXYZRGB> DenseCloud;

public:
    Frame();

    Frame(cv::Mat& imColor, cv::Mat& imDepth, double mTimestamp);

    Frame(cv::Mat& imColor);

    // Pose functions
    void SetPose(cv::Mat Tcw);
    cv::Mat GetPose();
    cv::Mat GetRotationInv();
    cv::Mat GetCameraCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Feature extraction
    void ExtractFeatures(Extractor* pExtractor);
    void Detect(cv::Ptr<cv::FeatureDetector> pDetector);
    void Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor);

    void ComputeBoW(DBoW3::Vocabulary* pVoc);

    // Point cloud processing
    void CreateCloud();
    void VoxelGridFilterCloud(float resolution);
    void StatisticalOutlierRemovalFilterCloud(int meanK, double stddev);

    // Landmark observation functions
    void AddLandmark(Landmark* pLandmark, const size_t& idx);
    std::set<Landmark*> GetLandmarks();
    std::vector<Landmark*> GetLandmarksMatched();
    Landmark* GetLandmark(const size_t& idx);

    // Backprojects a keypoint (if depth info available) into 3D world coordinates
    cv::Mat UnprojectWorld(const size_t& i);

    long unsigned int GetId();

    // Outlier/Inlier feature association
    void SetOutlier(const size_t& idx);
    void SetInlier(const size_t& idx);
    bool IsOutlier(const size_t& idx);
    bool IsInlier(const size_t& idx);

    // Copy operator
    const Frame& operator=(Frame& frame);

public:
    cv::Mat mImColor;
    cv::Mat mImGray;
    cv::Mat mImDepth;

    double mTimestamp;

    DenseCloud::Ptr mpCloud = nullptr;

    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<cv::Point3f> mvKeys3Dc;

        cv::Mat mDescriptors;

    std::vector<bool> mvbOutlier;

    // Bag of Words structures
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;

    // Number of keypoints
    size_t N;

    // Current and Next Frame id.
    static long unsigned int nNextId;

protected:
    long unsigned int mnId;

    // Camera pose.
    cv::Mat mTcw;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc (Camera center)

    // Landmark to associated keypoint
    std::vector<Landmark*> mvpLandmarks;

    std::mutex mMutexPose;
    std::mutex mMutexFeatures;
    std::mutex mMutexId;
};
#endif // FRAME_H
