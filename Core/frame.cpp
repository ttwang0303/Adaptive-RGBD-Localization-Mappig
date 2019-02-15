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
    mnId = nNextId++;

    imDepth.convertTo(mImDepth, CV_32F, depthFactor);
}

Frame::Frame(cv::Mat& imColor)
    : mImColor(imColor)
    , mImDepth(cv::Mat())
    , mTimestamp(0)
    , mpCloud(nullptr)
{
    // Frame ID
    mnId = nNextId++;
}

void Frame::SetPose(cv::Mat& Tcw)
{
    unique_lock<mutex> lock1(mMutexPose);
    mTcw = Tcw.clone();

    // Update pose matrices
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;
}

cv::Mat Frame::GetPose()
{
    unique_lock<mutex> lock1(mMutexPose);
    return mTcw.clone();
}

cv::Mat Frame::GetRotationInv()
{
    unique_lock<mutex> lock1(mMutexPose);
    return mRwc.clone();
}

cv::Mat Frame::GetCameraCenter()
{
    unique_lock<mutex> lock1(mMutexPose);
    return mOw.clone();
}

cv::Mat Frame::GetRotation()
{
    unique_lock<mutex> lock1(mMutexPose);
    return mRcw.clone();
}

cv::Mat Frame::GetTranslation()
{
    unique_lock<mutex> lock1(mMutexPose);
    return mtcw.clone();
}

void Frame::ExtractFeatures(Extractor* pExtractor)
{
    pExtractor->Extract(mImColor, cv::Mat(), mvKps, mDescriptors);

    N = mvKps.size();
    mvKps3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));
    }

    for (size_t i = 0; i < N; ++i) {
        const float v = mvKps[i].pt.y;
        const float u = mvKps[i].pt.x;

        const float z = mImDepth.at<float>(v, u);
        if (z > 0) {
            // KeyPoint in Camera coordinates
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            mvKps3Dc[i] = cv::Point3f(x, y, z);
        }
    }
}

void Frame::Detect(cv::Ptr<cv::FeatureDetector> pDetector)
{
    pDetector->detect(mImColor, mvKps);
    N = mvKps.size();
}

void Frame::Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDescriptor->compute(mImColor, mvKps, mDescriptors);
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
                p.x = (j - cx) * z * invfx;
                p.y = (i - cy) * z * invfy;

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

set<Landmark*> Frame::GetLandmarks()
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

vector<Landmark*> Frame::GetLandmarksMatched()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks;
}

Landmark* Frame::GetLandmark(const size_t& idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpLandmarks[idx];
}

cv::Mat Frame::UnprojectWorld(const size_t& i)
{
    const cv::Point3f& p3Dc = mvKps3Dc[i];
    if (p3Dc.z > 0) {
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

        unique_lock<mutex> lock1(mMutexPose);
        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}

const Frame& Frame::operator=(Frame& frame)
{
    if (&frame != this) {
        mImColor = frame.mImColor;
        mImDepth = frame.mImDepth;
        mTimestamp = frame.mTimestamp;
        mvKps = frame.mvKps;
        mDescriptors = frame.mDescriptors;
        mvKps3Dc = frame.mvKps3Dc;
        mBowVec = frame.mBowVec;
        mFeatVec = frame.mFeatVec;
        N = frame.N;
        mnId = frame.mnId;

        if (frame.mpCloud)
            mpCloud = frame.mpCloud;

        cv::Mat framePose = frame.GetPose();
        if (!framePose.empty())
            SetPose(framePose);

        mvpLandmarks = frame.GetLandmarksMatched();
    }

    return *this;
}
