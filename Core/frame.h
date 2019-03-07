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

    virtual ~Frame() {}

    // Pose functions
    virtual void SetPose(cv::Mat Tcw);
    virtual cv::Mat GetPose();
    virtual cv::Mat GetPoseInv();
    virtual cv::Mat GetRotationInv();
    virtual cv::Mat GetCameraCenter();
    virtual cv::Mat GetRotation();
    virtual cv::Mat GetTranslation();
    void UpdatePoseMatrices();

    bool isInFrustum(Landmark* pLM);

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
    virtual void AddLandmark(Landmark* pLandmark, const size_t& idx);
    virtual void ReleaseLandmark(const size_t& idx);
    virtual std::vector<Landmark*> GetLandmarks();
    virtual Landmark* GetLandmark(const size_t& idx);

    // Backprojects a keypoint (if depth info available) into 3D world coordinates
    virtual cv::Mat UnprojectWorld(const size_t& i);

    std::vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r) const;

    // Outlier/Inlier feature association
    void SetOutlier(const size_t& idx);
    void SetInlier(const size_t& idx);
    bool IsOutlier(const size_t& idx);
    bool IsInlier(const size_t& idx);
    std::vector<bool> GetOutliers();

public:
    cv::Mat mImColor;
    cv::Mat mImGray;
    cv::Mat mImDepth;

    double mTimestamp;

    cv::Mat mK;
    cv::Mat mDistCoef;

    DenseCloud::Ptr mpCloud = nullptr;

    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<cv::Point3f> mvKeys3Dc;
    std::vector<float> mvuRight;

    cv::Mat mDescriptors;

    // Bag of Words structures
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;

    // Number of keypoints
    size_t N;

    // Current and Next Frame id.
    static long unsigned int nNextFrameId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

protected:
    void UndistortKeyPoints();

    void ComputeImageBounds();

    // Camera pose.
    cv::Mat mTcw;
    cv::Mat Twc;

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc (Camera center)

    std::vector<bool> mvbOutlier;

    // Landmark to associated keypoint
    std::vector<Landmark*> mvpLandmarks;
};
#endif // FRAME_H
