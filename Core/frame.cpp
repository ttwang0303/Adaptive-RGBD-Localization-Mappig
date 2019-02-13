#include "frame.h"
#include "Utils/constants.h"
#include "dbscan.h"
#include "landmark.h"
#include <boost/make_shared.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

long unsigned int Frame::nNextId = 0;

Frame::Frame(cv::Mat& imColor, cv::Mat& imDepth, double timestamp)
    : mbIsKeyFrame(false)
    , mIm(imColor)
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

void Frame::DetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor)
{
    pDetector->detect(mIm, mvKps);
    if (mvKps.size() > nFeatures)
        cv::KeyPointsFilter::retainBest(mvKps, nFeatures);

    pDescriptor->compute(mIm, mvKps, mDescriptors);

    N = mvKps.size();
    mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));

    mvKps3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));

    for (size_t i = 0; i < N; ++i) {
        const float v = mvKps[i].pt.y;
        const float u = mvKps[i].pt.x;

        const float z = mDepth.at<float>(v, u);
        if (z > 0) {
            // KeyPoint in Camera coordinates
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            mvKps3Dc[i] = cv::Point3f(x, y, z);
        }
    }
}

void Frame::GridDetectAndCompute(cv::Ptr<cv::FeatureDetector> pDetector, cv::Ptr<cv::DescriptorExtractor> pDescriptor, int gridRows, int gridCols)
{
    cv::Mat grayImage;
    cv::cvtColor(mIm, grayImage, CV_BGR2GRAY);

    std::vector<cv::KeyPoint> raw_keypoints;
    int grayImageWidth = grayImage.cols;
    int grayImageHeight = grayImage.rows;

    int maximalFeaturesInROI = nFeatures * 3 / (gridCols * gridRows);

    // Let's divide image into boxes/rectangles
    for (int k = 0; k < gridCols; k++) {
        for (int i = 0; i < gridRows; i++) {

            std::vector<cv::KeyPoint> keypointsInROI;
            cv::Mat roiBGR(grayImage, cv::Rect(k * grayImageWidth / gridCols, i * grayImageHeight / gridRows, grayImageWidth / gridCols, grayImageHeight / gridRows));

            pDetector->detect(roiBGR, keypointsInROI);

            // Sorting keypoints by the response to choose the bests
            std::sort(keypointsInROI.begin(), keypointsInROI.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
                return p1.response > p2.response;
            });

            // Adding to final keypoints
            for (size_t j = 0; j < keypointsInROI.size() && j < maximalFeaturesInROI; j++) {
                keypointsInROI[j].pt.x += float(k * grayImageWidth / gridCols);
                keypointsInROI[j].pt.y += float(i * grayImageHeight / gridRows);
                raw_keypoints.push_back(keypointsInROI[j]);
            }
        }
    }

    // It is better to have them sorted according to their response strength
    std::sort(raw_keypoints.begin(), raw_keypoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
        return p1.response > p2.response;
    });

    if (raw_keypoints.size() > nFeatures)
        raw_keypoints.resize(nFeatures);

    mvKps = raw_keypoints;
    double epsilon = 1.0;
    DBScan dbscan(epsilon);
    dbscan.run(mvKps);

    pDescriptor->compute(grayImage, mvKps, mDescriptors);

    N = mvKps.size();
    mvpLandmarks = vector<Landmark*>(N, static_cast<Landmark*>(nullptr));

    mvKps3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));

    for (int i = 0; i < N; ++i) {
        const float v = mvKps[i].pt.y;
        const float u = mvKps[i].pt.x;

        const float z = mDepth.at<float>(v, u);
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

    mpCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    for (int i = 0; i < mDepth.rows; ++i) {
        for (int j = 0; j < mDepth.cols; ++j) {
            float z = mDepth.at<float>(i, j);
            if (z > 0) {
                pcl::PointXYZRGB p;
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Frame::Mat2Cloud()
{
    if (mpCloud)
        return mpCloud;

    mpCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    mpCloud->points.resize(width * height);

    for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < height; j++) {
            pcl::PointXYZRGB pt;
            if (!(mDepth.at<float>(j, i) == mDepth.at<float>(j, i))) {
                pt.z = 0.f / 0.f;
                mpCloud->points.at(j * width + i) = pt;
                continue;
            }

            pt.z = mDepth.at<float>(j, i);
            pt.x = (float(i) - cx) * pt.z * invfx;
            pt.y = (float(j) - cy) * pt.z * invfy;

            cv::Vec3b color = mIm.at<cv::Vec3b>(j, i);
            pt.r = (int)color.val[0];
            pt.g = (int)color.val[1];
            pt.b = (int)color.val[2];

            mpCloud->points.at(j * width + i) = pt;
        }
    }

    return mpCloud;
}

void Frame::VoxelGridFilterCloud(float resolution)
{
    if (!mpCloud)
        return;

    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.setInputCloud(mpCloud);
    voxel.filter(*mpCloud);
}

void Frame::StatisticalOutlierRemovalFilterCloud(int meanK, double stddev)
{
    if (!mpCloud)
        return;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(mpCloud);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddev);
    sor.filter(*mpCloud);
}

void Frame::AddLandmark(Landmark* pLandmark, const size_t& idx)
{
    mvpLandmarks[idx] = pLandmark;
}

cv::Mat Frame::UnprojectWorld(const size_t& i)
{

    const cv::Point3f& p3Dc = mvKps3Dc[i];
    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << p3Dc.x, p3Dc.y, p3Dc.z);

    unique_lock<mutex> lock1(mMutexPose);
    return mRwc * x3Dc + mOw;
}
