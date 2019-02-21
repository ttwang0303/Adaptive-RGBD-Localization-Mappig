#include "frame.h"
#include "Features/extractor.h"
#include "Utils/common.h"
#include "Utils/converter.h"
#include "landmark.h"
#include <boost/make_shared.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

long unsigned int Frame::nNextId = 0;

Frame::Frame() {}

Frame::Frame(cv::Mat& imColor, cv::Mat& imDepth, double timestamp)
    : mImColor(imColor)
    , mTimestamp(timestamp)
    , mpCloud(nullptr)
{
    // Frame ID
    unique_lock<mutex> lock(mMutexId);
    mnId = nNextId++;

    cv::cvtColor(mImColor, mImGray, cv::COLOR_BGR2GRAY);
    imDepth.convertTo(mImDepth, CV_32F, Calibration::depthFactor);
}

Frame::Frame(cv::Mat& imColor)
    : mImColor(imColor)
    , mImDepth(cv::Mat())
    , mTimestamp(0)
    , mpCloud(nullptr)
{
    cv::cvtColor(mImColor, mImGray, cv::COLOR_BGR2GRAY);

    // Frame ID
    unique_lock<mutex> lock(mMutexId);
    mnId = nNextId++;
}

void Frame::SetPose(cv::Mat Tcw)
{
    unique_lock<mutex> lock(mMutexPose);
    mTcw = Tcw.clone();

    // Update pose matrices
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;

    // Transform from camera into world frame
    Twc = cv::Mat::eye(4, 4, mTcw.type());
    mRwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    mOw.copyTo(Twc.rowRange(0, 3).col(3));
}

cv::Mat Frame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.clone();
}

cv::Mat Frame::GetPoseInv()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat Frame::GetRotationInv()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRwc.clone();
}

cv::Mat Frame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOw.clone();
}

cv::Mat Frame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mRcw.clone();
}

cv::Mat Frame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mtcw.clone();
}

void Frame::ExtractFeatures(Extractor* pExtractor)
{
    pExtractor->Extract(mImColor, cv::Mat(), mvKeys, mDescriptors);

    N = mvKeys.size();
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));
    }

    mvKeys3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));
    mvbOutlier = vector<bool>(N, false);

    for (size_t i = 0; i < N; ++i) {
        const float v = mvKeys[i].pt.y;
        const float u = mvKeys[i].pt.x;

        const float z = mImDepth.at<float>(v, u);
        if (z > 0) {
            // KeyPoint in Camera coordinates
            const float x = (u - Calibration::cx) * z * Calibration::invfx;
            const float y = (v - Calibration::cy) * z * Calibration::invfy;
            mvKeys3Dc[i] = cv::Point3f(x, y, z);
        }
    }
}

void Frame::Detect(cv::Ptr<cv::FeatureDetector> pDetector)
{
    pDetector->detect(mImColor, mvKeys);
    N = mvKeys.size();
}

void Frame::Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDescriptor->compute(mImColor, mvKeys, mDescriptors);
}

void Frame::ComputeBoW(DBoW3::Vocabulary* pVoc)
{
    if (mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::CreateCloud()
{
    if (mpCloud)
        return;

    mpCloud = boost::make_shared<DenseCloud>();

    for (int i = 0; i < mImDepth.rows; ++i) {
        for (int j = 0; j < mImDepth.cols; ++j) {
            float z = mImDepth.at<float>(i, j);
            if (z > 0) {
                DenseCloud::PointType p;
                p.z = z;
                p.x = (j - Calibration::cx) * z * Calibration::invfx;
                p.y = (i - Calibration::cy) * z * Calibration::invfy;

                p.b = mImColor.ptr<uchar>(i)[j * 3];
                p.g = mImColor.ptr<uchar>(i)[j * 3 + 1];
                p.r = mImColor.ptr<uchar>(i)[j * 3 + 2];

                mpCloud->points.push_back(p);
            }
        }
    }
}

void Frame::VoxelGridFilterCloud(float resolution)
{
    if (!mpCloud)
        return;

    pcl::VoxelGrid<DenseCloud::PointType> voxel;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.setInputCloud(mpCloud);
    voxel.filter(*mpCloud);
}

void Frame::StatisticalOutlierRemovalFilterCloud(int meanK, double stddev)
{
    if (!mpCloud)
        return;

    pcl::StatisticalOutlierRemoval<DenseCloud::PointType> sor;
    sor.setInputCloud(mpCloud);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddev);
    sor.filter(*mpCloud);
}

void Frame::AddLandmark(Landmark* pLandmark, const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = pLandmark;
}

set<Landmark*> Frame::GetLandmarkSet()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<Landmark*> s;
    for (Landmark* pLMK : mvpLandmarks) {
        if (!pLMK)
            continue;

        s.insert(pLMK);
    }
    return s;
}

vector<Landmark*> Frame::GetLandmarks()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks;
}

Landmark* Frame::GetLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks[idx];
}

void Frame::EraseLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = static_cast<Landmark*>(nullptr);
}

void Frame::ReplaceLandmark(const size_t& idx, Landmark* pLM)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpLandmarks[idx] = pLM;
}

cv::Mat Frame::UnprojectWorld(const size_t& i)
{
    const cv::Point3f& p3Dc = mvKeys3Dc[i];
    if (p3Dc.z > 0) {
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

        unique_lock<mutex> lock1(mMutexPose);
        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}

unsigned long Frame::GetId()
{
    unique_lock<mutex> lock(mMutexId);
    return mnId;
}

void Frame::SetOutlier(const size_t& idx)
{
    mvbOutlier[idx] = true;
}

void Frame::SetInlier(const size_t& idx)
{
    mvbOutlier[idx] = false;
}

bool Frame::IsOutlier(const size_t& idx)
{
    return mvbOutlier[idx] == true;
}

bool Frame::IsInlier(const size_t& idx)
{
    return mvbOutlier[idx] == false;
}

const Frame& Frame::operator=(Frame& frame)
{
    if (&frame != this) {
        mImColor = frame.mImColor;
        mImGray = frame.mImGray;
        mImDepth = frame.mImDepth;
        mTimestamp = frame.mTimestamp;
        mvKeys = frame.mvKeys;
        mDescriptors = frame.mDescriptors;
        mvKeys3Dc = frame.mvKeys3Dc;
        mvbOutlier = frame.mvbOutlier;
        mBowVec = frame.mBowVec;
        mFeatVec = frame.mFeatVec;
        N = frame.N;
        mnId = frame.GetId();

        if (frame.mpCloud)
            mpCloud = frame.mpCloud;

        cv::Mat framePose = frame.GetPose();
        if (!framePose.empty())
            SetPose(framePose);

        mvpLandmarks = frame.GetLandmarks();
    }

    return *this;
}
