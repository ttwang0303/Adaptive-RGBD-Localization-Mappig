#include "frame.h"
#include "Utils/constants.h"
#include <pcl/filters/filter.h>

using namespace std;

Frame::Frame()
    : cloud(new pcl::PointCloud<pcl::PointXYZRGBA>)
{
    cv::Mat init = cv::Mat::eye(4, 4, CV_32F);
    SetPose(init);
}

Frame::Frame(cv::Mat& imColor, cv::Mat& imDepth, double timestamp)
    : cloud(new pcl::PointCloud<pcl::PointXYZRGBA>)
{
    im = imColor;
    imDepth.convertTo(depth, CV_32F, depthFactor);
    this->timestamp = timestamp;

    cv::Mat init = cv::Mat::eye(4, 4, CV_32F);
    SetPose(init);
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
    pDetector->detect(im, kps);
    if (kps.size() > 1000)
        cv::KeyPointsFilter::retainBest(kps, 1000);

    pDescriptor->compute(im, kps, desc);

    N = kps.size();

    kps3Dc = vector<cv::Point3f>(N, cv::Point3f(0, 0, 0));

    for (int i = 0; i < N; ++i) {
        const float v = kps[i].pt.y;
        const float u = kps[i].pt.x;

        const float z = depth.at<float>(v, u);
        if (z > 0) {
            // KeyPoint in Camera coordinates
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            kps3Dc[i] = cv::Point3f(x, y, z);
        }
    }
}

void Frame::CreateCloud()
{
    for (int i = 0; i < depth.rows; ++i) {
        for (int j = 0; j < depth.cols; ++j) {
            float z = depth.at<float>(i, j);
            if (z > 0) {
                pcl::PointXYZRGBA p;
                p.z = z;
                p.x = (j - cx) * z * invfx;
                p.y = (i - cy) * z * invfy;

                p.b = im.ptr<uchar>(i)[j * 3];
                p.g = im.ptr<uchar>(i)[j * 3 + 1];
                p.r = im.ptr<uchar>(i)[j * 3 + 2];

                cloud->points.push_back(p);
            }
        }
    }

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
}
