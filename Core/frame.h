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
class KeyFrame;

class Frame {
public:
    typedef pcl::PointCloud<pcl::PointXYZRGB> DenseCloud;

public:
    Frame();

    Frame(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp);

    Frame(cv::Mat& imColor);

    void Initialize(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp);

    // Pose functions
    void SetPose(cv::Mat Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInv();
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
    std::set<Landmark*> GetLandmarkSet();
    std::vector<Landmark*> GetLandmarks();
    Landmark* GetLandmark(const size_t& idx);
    void EraseLandmark(const size_t& idx);
    void ReplaceLandmark(const size_t& idx, Landmark* pLM);

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

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

protected:
    long unsigned int mnId;

    // Camera pose.
    cv::Mat mTcw;
    cv::Mat Twc;

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
