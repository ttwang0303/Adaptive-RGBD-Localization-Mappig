#include "pointclouddrawer.h"
#include <boost/make_shared.hpp>
#include <pangolin/pangolin.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

PointCloudDrawer::PointCloudDrawer()
{
    mpSourceCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTargetCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpMapCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    mPointSize = 2.0f;
    mLineWidth = 0.9f;
    srand((unsigned int)time(NULL));

    r = (double)rand() / (double)RAND_MAX;
    g = (double)rand() / (double)RAND_MAX;
    b = (double)rand() / (double)RAND_MAX;
}

void PointCloudDrawer::DrawPointCloud()
{
    unique_lock<mutex> lock(mMutexPointCloud);

    if (!mpSourceCloud->points.empty() && !mpTargetCloud->points.empty()) {
        size_t N = mpSourceCloud->points.size();

        for (size_t i = 0; i < N; ++i) {
            glPointSize(mPointSize);
            glBegin(GL_POINTS);

            // Red for source
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);

            // Blue for target
            glColor3f(0.0f, 0.0f, 1.0f);
            glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

            glEnd();

            // Draw connector
            glLineWidth(mLineWidth);
            glColor3f(0.0f, 0.75f, 0.0f);
            glBegin(GL_LINES);

            glVertex3f(mpSourceCloud->points[i].x, mpSourceCloud->points[i].y, mpSourceCloud->points[i].z);
            glVertex3f(mpTargetCloud->points[i].x, mpTargetCloud->points[i].y, mpTargetCloud->points[i].z);

            glEnd();
        }

        glEnd();
    } else {
        if (mpMapCloud->points.empty())
            return;

        size_t N = mpMapCloud->points.size();

        glPointSize(mPointSize);
        glBegin(GL_POINTS);

        for (size_t i = 0; i < N; ++i) {
            glColor3f(mpMapCloud->points[i].r / 255.0f, mpMapCloud->points[i].g / 255.0f, mpMapCloud->points[i].b / 255.0f);
            glVertex3f(mpMapCloud->points[i].x, mpMapCloud->points[i].y, mpMapCloud->points[i].z);
        }

        glEnd();

        // Draw poses
        for (int i = 1; i < mvPoses.size(); ++i) {
            glPointSize(mPointSize * 3);
            glBegin(GL_POINTS);
            glColor3f(r, g, b);
            glVertex3f(mvPoses[i].at<float>(0, 3), mvPoses[i].at<float>(1, 3), mvPoses[i].at<float>(2, 3));
            glEnd();

            if (i < mvPoses.size() - 1) {
                glBegin(GL_LINES);
                glVertex3f(mvPoses[i].at<float>(0, 3), mvPoses[i].at<float>(1, 3), mvPoses[i].at<float>(2, 3));
                glVertex3f(mvPoses[i + 1].at<float>(0, 3), mvPoses[i + 1].at<float>(1, 3), mvPoses[i + 1].at<float>(2, 3));

                glEnd();
            }
        }
    }
}

void PointCloudDrawer::UpdateSourceCloud(pcl::PointCloud<pcl::PointXYZ>& pSource)
{
    unique_lock<mutex> lock(mMutexPointCloud);
    pcl::copyPointCloud(pSource, *mpSourceCloud);
}

void PointCloudDrawer::UpdateTargetCloud(pcl::PointCloud<pcl::PointXYZ>& pTarget)
{
    unique_lock<mutex> lock(mMutexPointCloud);
    pcl::copyPointCloud(pTarget, *mpTargetCloud);
}

void PointCloudDrawer::UpdateMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pMapCloud, const cv::Mat& pose)
{
    unique_lock<mutex> lock(mMutexPointCloud);

    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    float resolution = 0.02f;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.setInputCloud(pMapCloud);
    voxel.filter(*pMapCloud);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(pMapCloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*pMapCloud);

    for (int i = 0; i < pMapCloud->points.size(); ++i)
        mpMapCloud->points.push_back(pMapCloud->points[i]);

    mvPoses.push_back(pose.clone());

    r = (double)rand() / (double)RAND_MAX;
    g = (double)rand() / (double)RAND_MAX;
    b = (double)rand() / (double)RAND_MAX;
}
