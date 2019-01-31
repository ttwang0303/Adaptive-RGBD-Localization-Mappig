#include "frame.h"
#include "Utils/constants.h"
#include "landmark.h"
#include <boost/make_shared.hpp>

using namespace std;

long unsigned int Frame::nNextId = 0;

Frame::Frame(cv::Mat& imColor, cv::Mat& imDepth, double timestamp)
    : mIm(imColor)
    , mTimestamp(timestamp)
    , mpCloud(nullptr)
{
    // Frame ID
    mnId = nNextId++;

    imDepth.convertTo(mDepth, CV_32F, depthFactor);
}

Frame::Frame(cv::Mat& imColor)
    : mIm(imColor)
    , mDepth(cv::Mat())
    , mTimestamp(0)
    , mpCloud(nullptr)
{
    // Frame ID
    mnId = nNextId++;
}

void Frame::SetPose(cv::Mat& Tcw)
{
    mTcw = Tcw.clone();

    // Update pose matrices
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRcw.t() * mtcw;
}

void Frame::DetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDetector->detect(mIm, mvKps);
    if (mvKps.size() > nFeatures)
        cv::KeyPointsFilter::retainBest(mvKps, nFeatures);

    pDescriptor->compute(mIm, mvKps, mDescriptors);

    N = mvKps.size();
    mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));

    //    mvKps3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));

    //    for (int i = 0; i < N; ++i) {
    //        const float v = mvKps[i].pt.y;
    //        const float u = mvKps[i].pt.x;

    //        const float z = mDepth.at<float>(v, u);
    //        if (z > 0) {
    //            // KeyPoint in Camera coordinates
    //            const float x = (u - cx) * z * invfx;
    //            const float y = (v - cy) * z * invfy;
    //            mvKps3Dc[i] = cv::Point3f(x, y, z);
    //        }
    //    }
}

void Frame::Detect(cv::Ptr<cv::FeatureDetector> pDetector)
{
    pDetector->detect(mIm, mvKps);
    N = mvKps.size();
}

void Frame::Compute(cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDescriptor->compute(mIm, mvKps, mDescriptors);
}

void Frame::CreateCloud()
{
    if (mpCloud)
        return;

    mpCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();

    for (int i = 0; i < mDepth.rows; ++i) {
        for (int j = 0; j < mDepth.cols; ++j) {
            float z = mDepth.at<float>(i, j);
            if (z > 0) {
                pcl::PointXYZRGBA p;
                p.z = z;
                p.x = (j - cx) * z * invfx;
                p.y = (i - cy) * z * invfy;

                p.b = mIm.ptr<uchar>(i)[j * 3];
                p.g = mIm.ptr<uchar>(i)[j * 3 + 1];
                p.r = mIm.ptr<uchar>(i)[j * 3 + 2];

                mpCloud->points.push_back(p);
            }
        }
    }
}

void Frame::AddLandmark(Landmark* pLandmark, const size_t& idx)
{
    mvpLandmarks[idx] = pLandmark;
}
