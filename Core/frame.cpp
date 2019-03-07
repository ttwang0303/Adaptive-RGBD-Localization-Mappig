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
bool Frame::mbInitialComputations = true;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;

Frame::Frame() {}

Frame::Frame(const cv::Mat& imColor, const cv::Mat& imDepth, const double& timestamp)
    : mImColor(imColor)
    , mTimestamp(timestamp)
    , mpCloud(nullptr)
{
    cv::cvtColor(mImColor, mImGray, cv::COLOR_BGR2GRAY);
    imDepth.convertTo(mImDepth, CV_32F, Calibration::depthFactor);

    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = Calibration::fx;
    mK.at<float>(1, 1) = Calibration::fy;
    mK.at<float>(0, 2) = Calibration::cx;
    mK.at<float>(1, 2) = Calibration::cy;

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = Calibration::k1;
    DistCoef.at<float>(1) = Calibration::k2;
    DistCoef.at<float>(2) = Calibration::p1;
    DistCoef.at<float>(3) = Calibration::p2;
    const float k3 = Calibration::k3;
    if (k3 != 0.0f) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

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

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    // Data used by the tracking
    pLM->mbTrackInView = true;
    pLM->mTrackProjX = u;
    pLM->mTrackProjXR = u - Calibration::mbf * invz;
    pLM->mTrackProjY = v;

    return true;
}

void Frame::ExtractFeatures(Extractor* pExtractor)
{
    pExtractor->Extract(mImGray, cv::Mat(), mvKeys, mDescriptors);

    N = mvKeys.size();

    UndistortKeyPoints();

    mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));
    mvKeys3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));
    mvbOutlier = vector<bool>(N, false);
    mvuRight = vector<float>(N, -1);

    for (size_t i = 0; i < N; ++i) {
        const cv::KeyPoint& kp = mvKeys[i];
        const cv::KeyPoint& kpU = mvKeysUn[i];

        const float v = kp.pt.y;
        const float u = kp.pt.x;

        const float z = mImDepth.at<float>(v, u);
        if (z > 0) {
            mvuRight[i] = kpU.pt.x - Calibration::mbf / z;

            // KeyPoint in Camera coordinates
            const float x = (kpU.pt.x - Calibration::cx) * z * Calibration::invfx;
            const float y = (kpU.pt.y - Calibration::cy) * z * Calibration::invfy;
            mvKeys3Dc[i] = cv::Point3f(x, y, z);
        }
    }

    if (mbInitialComputations) {
        ComputeImageBounds();
        mbInitialComputations = false;
    }
}

void Frame::Detect(cv::Ptr<cv::FeatureDetector> pDetector)
{
    pDetector->detect(mImGray, mvKeys);
    N = mvKeys.size();
}

void Frame::Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDescriptor->compute(mImGray, mvKeys, mDescriptors);
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

void Frame::ReleaseLandmark(const size_t& idx) { mvpLandmarks[idx] = static_cast<Landmark*>(nullptr); }

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
        const cv::KeyPoint& kpU = mvKeysUn[i];

        const float distx = kpU.pt.x - x;
        const float disty = kpU.pt.y - y;

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

void Frame::UndistortKeyPoints()
{
    if (mDistCoef.at<float>(0) == 0.0f) {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);
    for (size_t i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++) {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeImageBounds()
{
    if (mDistCoef.at<float>(0) != 0.0f) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = mImGray.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = mImGray.rows;
        mat.at<float>(3, 0) = mImGray.cols;
        mat.at<float>(3, 1) = mImGray.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

    } else {
        mnMinX = 0.0f;
        mnMaxX = mImGray.cols;
        mnMinY = 0.0f;
        mnMaxY = mImGray.rows;
    }
}
