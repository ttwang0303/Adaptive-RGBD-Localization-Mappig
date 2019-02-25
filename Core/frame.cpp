#include "frame.h"
#include "Features/extractor.h"
#include "Utils/common.h"
#include "Utils/converter.h"
#include "landmark.h"
#include <boost/make_shared.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

long unsigned int Frame::nNextFrameId = 0;

Frame::Frame() {}

Frame::Frame(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp)
    : mImColor(imColor)
    , mTimestamp(timestamp)
    , mpCloud(nullptr)
{
    cv::cvtColor(mImColor, mImGray, cv::COLOR_BGR2GRAY);
    imDepth.convertTo(mImDepth, CV_32F, Calibration::depthFactor);

    mnId = nNextFrameId++;
}

Frame::Frame(cv::Mat& imColor)
    : mImColor(imColor)
    , mImDepth(cv::Mat())
    , mTimestamp(0)
    , mpCloud(nullptr)
{
    cv::cvtColor(mImColor, mImGray, cv::COLOR_BGR2GRAY);

    mnId = nNextFrameId++;
}

void Frame::SetPose(cv::Mat Tcw)
{
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

cv::Mat Frame::GetPose() { return mTcw.clone(); }

cv::Mat Frame::GetPoseInv() { return Twc.clone(); }

cv::Mat Frame::GetRotationInv() { return mRwc.clone(); }

cv::Mat Frame::GetCameraCenter() { return mOw.clone(); }

cv::Mat Frame::GetRotation() { return mRcw.clone(); }

cv::Mat Frame::GetTranslation() { return mtcw.clone(); }

void Frame::UpdatePoseMatrices()
{
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

bool Frame::isInFrustum(Landmark* pLM)
{
    pLM->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pLM->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw * P + mtcw;
    const float& PcX = Pc.at<float>(0);
    const float& PcY = Pc.at<float>(1);
    const float& PcZ = Pc.at<float>(2);

    if (PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u = Calibration::fx * PcX * invz + Calibration::cx;
    const float v = Calibration::fy * PcY * invz + Calibration::cy;

    if (u < 0.0f || u > mImGray.cols)
        return false;
    if (v < 0.0f || v > mImGray.rows)
        return false;

    // Data used by the tracking
    pLM->mbTrackInView = true;
    pLM->mTrackProjX = u;
    pLM->mTrackProjY = v;

    return true;
}

void Frame::ExtractFeatures(Extractor* pExtractor)
{
    pExtractor->Extract(mImColor, cv::Mat(), mvKeys, mDescriptors);

    N = mvKeys.size();
    mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));

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

void Frame::AddLandmark(Landmark* pLandmark, const size_t& idx) { mvpLandmarks[idx] = pLandmark; }

vector<Landmark*> Frame::GetLandmarks() { return mvpLandmarks; }

Landmark* Frame::GetLandmark(const size_t& idx) { return mvpLandmarks[idx]; }

cv::Mat Frame::UnprojectWorld(const size_t& i)
{
    const cv::Point3f& p3Dc = mvKeys3Dc[i];
    if (p3Dc.z > 0) {
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

        return mRwc * x3Dc + mOw;
    } else
        return cv::Mat();
}

vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y, const float& r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        const cv::KeyPoint& kp = mvKeys[i];

        const float distx = kp.pt.x - x;
        const float disty = kp.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r)
            vIndices.push_back(i);
    }

    return vIndices;
}

void Frame::SetOutlier(const size_t& idx) { mvbOutlier[idx] = true; }

void Frame::SetInlier(const size_t& idx) { mvbOutlier[idx] = false; }

bool Frame::IsOutlier(const size_t& idx) { return mvbOutlier[idx] == true; }

bool Frame::IsInlier(const size_t& idx) { return mvbOutlier[idx] == false; }

std::vector<bool> Frame::GetOutliers() { return mvbOutlier; }
